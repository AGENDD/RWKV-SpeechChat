from transformers import AutoTokenizer, AutoModelForCausalLM
from src.asr import SLAM_ASR
import torch

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
# print("CUDA 版本:", torch.version.cuda)
# print("CUDA 可用:", torch.cuda.is_available())
lm_model = RWKV(model='model/RWKV-x060-World-3B-v2.1-20240417-ctx4096', strategy='cuda fp32')
# pipe+line = PIPELINE(model, "rwkv_vocab_v20230424")
# out, state = lm_model.forward([187, 510, 1563, 310, 247], None)
# print(out.shape)
# print(len(state))
# for i in range(len(state)):
#     print(state[i].shape)


model = SLAM_ASR(
    "microsoft/wavlm-large",
    lm_model
).to("cuda")

model_path = "model/rwkv-adapter-25100.pth"
model_state_dict = torch.load(model_path)


state = []
# 打印参数名称和形状
for param_name, param_tensor in model_state_dict.items():
    if 'adapter' in param_name:
        model.state_dict()[param_name].copy_(param_tensor)
for i in range(32):
    state.append(torch.zeros(2560).to("cuda"))
    state.append(model_state_dict[f"language_model.blocks.{i}.att.time_state"].to("cuda"))
    state.append(torch.zeros(2560).to("cuda"))
# state = torch.stack(state)
# print(state.shape)

# for param_name, param_tensor in model_state_dict.items():
#     print(f"参数名称: {param_name}, 参数形状: {param_tensor.shape}")
# print("\n\n")
# for param_name, param_tensor in model.state_dict().items():
#     print(f"模型参数名称: {param_name}, 参数形状: {param_tensor.shape}")

import pyaudio
import numpy as np
import time

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

while True:
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)
    if np.max(audio_data) > THRESHOLD:
        if not recording:
            print("audio start...")
            recording = True
        frames.append(audio_data)
        silence_start = None  # 重置沉默计时器
    elif recording:
        if silence_start is None:
            silence_start = time.time()
        elif time.time() - silence_start > SILENCE_DURATION:
            print("audio end...")
            frames = np.hstack(frames)
            frames = frames.astype(np.float32) / 32768.0
            # print(frames)
            output = model.generate(frames, state.copy())
            output = ''.join(output)
            print(f"predict: {output}")
            frames = []
            # print(f"output: {output}")
            recording = False