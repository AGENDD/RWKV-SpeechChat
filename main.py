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
WEIGHT = args.weights_path

################################# Initialization ###################################

print("CUDA version:", torch.version.cuda)
print("CUDA avaliable:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA must be available to run this script."
print(f"Multiturns: {True if MULTITURNS else False}")
print(f"Trained weights: {WEIGHT}")

lm_model = RWKV(model=MODEL, strategy='cuda bf16')

model_state_dict = torch.load(WEIGHT)
print(f"Loaded trained weight:{WEIGHT}")

state = []
for i in range(32):
    state.append(torch.zeros(2560).to("cuda",dtype=torch.bfloat16))
    state.append(model_state_dict[f"language_model.blocks.{i}.att.time_state"].to("cuda",dtype=torch.bfloat16))
    state.append(torch.zeros(2560).to("cuda",dtype=torch.bfloat16))
 
layer = 1   
lora = False
loralayer = set()
lorarank = -1
import re

for param_name, param_tensor in model_state_dict.items():
    if('adapter' in param_name and '4' in param_name):
        layer = 2
    
    if("LoRA" in param_name):
        lora = True
        ll = numbers = re.findall(r'\d+', param_name)
        loralayer.add(ll[0])
        lorarank = min(param_tensor.shape[0], param_tensor.shape[1])
###################
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=64):
        super(LoRALayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, out_features))
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        self.r = r

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B)

def replace_linear_with_lora(model, r=64, layer = None):
    
    def change(model_):
        for name, module in model_.named_children():
            if isinstance(module, nn.Linear):
                print(f"\t{name}")
                in_features = module.in_features
                out_features = module.out_features
                lora_layer = LoRALayer(in_features, out_features, r)
                lora_layer.linear.weight = module.weight
                lora_layer.linear.bias = module.bias
                setattr(module, name, lora_layer)
    
    blocks = model.get_submodule("blocks")
    for name, module in blocks.named_children():
        
        if(blocks in layer):
            print(name)
            att = module.get_submodule("att")
            ffn = module.get_submodule("ffn")
            change(att)
            change(ffn)
    return model

if(lora):
    print("Changing to LoRA:")
    lm_model = replace_linear_with_lora(lm_model,r=lorarank, layer=ll)


###################

lm_model = lm_model.to("cuda", dtype=torch.bfloat16)

model = SLAM_ASR(
    "microsoft/wavlm-large",
    lm_model,
    layer = layer
).to("cuda", dtype=torch.bfloat16)

print("Loading weights:")
for param_name, param_tensor in model_state_dict.items():
    if 'state' not in param_name:
        print(f"\t{param_name}")
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
import keyboard


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

print("Inference start")
frames = []
recording = False
# silence_start = None
# SILENCE_DURATION = 1  # 设置沉默时间阈值（秒）

model.eval()

if(not MULTITURNS):
    while True:
        data = stream.read(CHUNK)
        if not recording:
            if keyboard.is_pressed('space'):
                print("start recording >...",end="")
                recording = True
                time_start = time.time()
                # frames.append(audio_data)
            silence_start = None  # 重置沉默计时器
        elif recording:
            audio_data = np.frombuffer(data, dtype=np.int16)
            frames.append(audio_data)
            # if silence_start is None:
            #     silence_start = time.time()
            # elif time.time() - silence_start > SILENCE_DURATION:
            if time.time() - time_start > 1 and keyboard.is_pressed('space'):
                print("< recording ends")
                print()
                frames = np.hstack(frames)
                # frames = np.pad(frames, (5000, 3000), 'constant', constant_values=0.0)
                
                # wavfile.write("this.wav", 16000, frames.astype(np.int16))
                # audio, sr = librosa.load("this.wav", sr=None)
                # audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                # 
                # 在内存中处理音频数据
                with io.BytesIO() as buffer:
                    sf.write(buffer, frames.astype(np.int16), 16000, format='WAV')
                    buffer.seek(0)
                    audio, sr = sf.read(buffer, dtype='int16')
                
                if(len(audio) /16000 * 50> 3):
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
        
        if not recording:
            if keyboard.is_pressed('space'):
                print("start recording >...",end="")
                recording = True
                time_start = time.time()
        elif recording:
            audio_data = np.frombuffer(data, dtype=np.int16)
            frames.append(audio_data)
            # if silence_start is None:
            #     silence_start = time.time()
            if time.time() - time_start > 1 and keyboard.is_pressed('space'):
                print("< recording ends")
                frames = np.hstack(frames)
                # frames = np.pad(frames, (5000, 3000), 'constant', constant_values=0.0)
                
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