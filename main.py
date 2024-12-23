from transformers import AutoTokenizer, AutoModelForCausalLM
from src.asr import SLAM_ASR
import torch
import argparse
# from rwkv.model import RWKV
from src.rwkv.model import RWKV


####################################### Parameters #################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Audio tasks with freezed RWKV')
parser.add_argument('--multiturns', action='store_true', help='Enable multi-turn conversation mode')
parser.add_argument('--rwkv_path',type=str,     default='model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',    help='Path to RWKV model weights')
parser.add_argument('--weights_path',type=str,  default='model/rwkv-adapter-speechQA-VoiceAssistant-final.pth',  help='Path to trained weights')

args = parser.parse_args()

MULTITURNS = args.multiturns
MODEL = args.rwkv_path
WIEGHT = args.weights_path

################################# Initialization ###################################

print("CUDA version:", torch.version.cuda)
print("CUDA avaliable:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA must be available to run this script."

lm_model = RWKV(model=MODEL, strategy='cuda bf16')

model_state_dict = torch.load(WIEGHT)

state = []
for i in range(32):
    state.append(torch.zeros(2560).to("cuda",dtype=torch.bfloat16))
    state.append(model_state_dict[f"language_model.blocks.{i}.att.time_state"].to("cuda",dtype=torch.bfloat16))
    state.append(torch.zeros(2560).to("cuda",dtype=torch.bfloat16))
 
layer = 1   
for param_name, param_tensor in model_state_dict.items():
    if('adapter' in param_name and '4' in param_name):
        layer = 2
        break

lm_model = lm_model.to("cuda", dtype=torch.bfloat16)

model = SLAM_ASR(
    "microsoft/wavlm-large",
    lm_model,
    layer = layer
).to("cuda", dtype=torch.bfloat16)

for param_name, param_tensor in model_state_dict.items():
    if 'adapter' in param_name:
        model.state_dict()[param_name].copy_(param_tensor)
        
#################################### Inference ####################################

import pyaudio
import numpy as np
import time
import librosa
import scipy.io.wavfile as wavfile
import os
import io
import soundfile as sf

# 设置参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
THRESHOLD = 1000

# 初始化pyaudio
audio = pyaudio.PyAudio()

# 打开流
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("start recording...")
frames = []
recording = False
silence_start = None
SILENCE_DURATION = 1  # 设置沉默时间阈值（秒）

model.eval()

if(not MULTITURNS):
    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.max(audio_data) > THRESHOLD:
            if not recording:
                # print("audio start...")
                recording = True
            frames.append(audio_data)
            silence_start = None  # 重置沉默计时器
        elif recording:
            frames.append(audio_data)
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_DURATION:
                # print("audio end...")
                frames = np.hstack(frames)
                frames = np.pad(frames, (5000, 3000), 'constant', constant_values=0.0)
                
                # wavfile.write("this.wav", 16000, frames.astype(np.int16))
                # audio, sr = librosa.load("this.wav", sr=None)
                # audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # 在内存中处理音频数据
                with io.BytesIO() as buffer:
                    sf.write(buffer, frames.astype(np.int16), 16000, format='WAV')
                    buffer.seek(0)
                    audio, sr = sf.read(buffer, dtype='int16')
                
                if(len(audio) /16000 * 50 > 5):
                    # print(f"audio length:{len(audio)}:{len(audio)/16000}")
                    
                    with torch.no_grad():
                        output = model.generate(audio, state.copy(),stop='<s>', stream=True)

                    print()
                frames = []

                recording = False
else:
    
    audios = []
    answers = []
    
    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.max(audio_data) > THRESHOLD:
            if not recording:
                # print("audio start...")
                recording = True
            frames.append(audio_data)
            silence_start = None  # 重置沉默计时器
        elif recording:
            frames.append(audio_data)
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_DURATION:
                # print("audio end...")
                frames = np.hstack(frames)
                frames = np.pad(frames, (5000, 3000), 'constant', constant_values=0.0)
                
                # 在内存中处理音频数据
                with io.BytesIO() as buffer:
                    sf.write(buffer, frames.astype(np.int16), 16000, format='WAV')
                    buffer.seek(0)
                    audio, sr = sf.read(buffer, dtype='int16')
                
                if(len(audio) /16000 * 50 > 5):
                    # print(f"audio length:{len(audio)}:{len(audio)/16000}")
                    
                    with torch.no_grad():
                        if(len(audios) != 0):
                            history_tensors = model.preprocess(audios, answers)
                            output = model.generate(audio, state.copy(),stop='<s>', stream=True, history = history_tensors)
                        else:
                            output = model.generate(audio, state.copy(),stop='<s>', stream=True)

                    audios.append(audio)
                    answers.append("".join(output))
                    print()
                frames = []

                recording = False