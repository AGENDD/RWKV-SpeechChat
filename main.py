from transformers import AutoTokenizer, AutoModelForCausalLM
from src.asr import SLAM_ASR
import torch

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
print("CUDA 版本:", torch.version.cuda)
print("CUDA 可用:", torch.cuda.is_available())
lm_model = RWKV(model='model/RWKV-x060-World-3B-v2.1-20240417-ctx4096', strategy='cuda bf16')
# pipe+line = PIPELINE(model, "rwkv_vocab_v20230424")
# out, state = lm_model.forward([187, 510, 1563, 310, 247], None)
# print(out.shape)
# print(len(state))
# for i in range(len(state)):
#     print(state[i].shape)
lm_model = lm_model.to(dtype=torch.bfloat16)

model = SLAM_ASR(
    "microsoft/wavlm-large",
    lm_model
).to("cuda", dtype=torch.bfloat16)

model_path = "model/rwkv-adapter-speechQA-VoiceAssistant-final.pth"
model_state_dict = torch.load(model_path)

state = []
# 打印参数名称和形状
for param_name, param_tensor in model_state_dict.items():
    if 'adapter' in param_name:
        model.state_dict()[param_name].copy_(param_tensor)
        
for i in range(32):
    state.append(torch.zeros(2560).to("cuda",dtype=torch.bfloat16))
    state.append(model_state_dict[f"language_model.blocks.{i}.att.time_state"].to("cuda",dtype=torch.bfloat16))
    state.append(torch.zeros(2560).to("cuda",dtype=torch.bfloat16))
# state = torch.stack(state)
# print(state.shape)
# for param_name, param_tensor in model_state_dict.items():
#     print(f"参数名称: {param_name}, 参数形状: {param_tensor.shape}")
# print("\n\n")
# for param_name, param_tensor in model.state_dict().items():
#     print(f"模型参数名称: {param_name}, 参数形状: {param_tensor.shape}")

########################################################################################

# import librosa
# import time

# audio, sr = librosa.load("this.wav", sr=None)
# audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
# model = model.to("cuda", dtype=torch.bfloat16)
# # model = model.to("cuda", dtype=torch.bfloat16)

# start_time = time.time()
# output= model.generate(audio, state.copy())
# output = ''.join(output)
# end_time = time.time()

# # print(f"audio: {args.file_path}")
# print(f"predict: {output}")
# print(f"Response time: {end_time - start_time} seconds")
# exit(0)
#########################################################################################

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
                # output = ''.join(output)
                # print(f"predict: {output}")
                print()
            frames = []
            # print(f"output: {output}")
            recording = False
            # os.remove("this.wav")
            # exit(0)