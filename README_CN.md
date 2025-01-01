# RWKV-SpeechChat

- [中文说明](README_CN.md)
- [English](README.md)

RWKV-SpeechChat 是一个基于冻结的 3B RWKV base模型、训练好的适配器和初始状态的实时对话脚本。相应的训练框架可在此处找到：https://github.com/AGENDD/RWKV-ASR，
当中提供了更详细的描述。可以应用各种训练好的权重来执行一系列音频任务，包括自动语音识别（ASR）、语音翻译、语音问答（QA）等。

## 特点

- **多音频任务支持**：支持多种音频任务，包括自动语音识别（ASR）、语音翻译、语音问答（QA）等，未来将支持更多任务。
- **本地部署**：可以在具有至少 6GB 显存的 GPU 的 PC 上运行。
- **实时对话**：支持与模型的实时对话，类似于 GPT-4。

## 演示

```/veidos``` 中是一些英语和中文语音问答任务的视频演示。


https://github.com/user-attachments/assets/6eae2e3d-ef07-4fc6-81c1-88fdf64bd3b1



https://github.com/user-attachments/assets/056a679f-7e6a-448e-b017-2a425eb220b0




https://github.com/user-attachments/assets/a0bd2174-7699-4dee-bbad-583d67911749



## 安装

1. 克隆仓库：
    ```bash
    git clone https://github.com/AGENDD/RWKV-SpeechChat.git
    cd RWKV-SpeechChat
    ```

2. 下载 RWKV 模型权重：

    从以下地址下载 RWKV 模型权重：
    https://huggingface.co/BlinkDL/rwkv-6-world/tree/main
    
    该项目目前仅支持 "RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth"。将权重放置在 `model` 目录中。

3. 下载训练好的权重：

    下载对应音频任务的训练权重：

    - ASR: https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/ASR
    - ST: https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/ST
    - SpeechQA: https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/SpeechQA

    将权重放置在 `model` 目录中。

## 使用方法

### 命令行参数

- `--multiturns`：启用多轮对话模式（移除此参数以禁用多轮对话）。
- `--rwkv_path`：RWKV 模型权重的路径（默认是 `model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth`）。
- `--weights_path`：训练权重的路径（默认是 `model/rwkv-adapter-speechQA-VoiceAssistant-final.pth`）。

### 运行脚本

你可以使用以下命令运行脚本：

```bash
python main.py --multiturns --rwkv_path path/to/your/model/weights.pth --weights_path path/to/your/trained/weights.pth
```

或者使用默认参数：

```bash
python main.py
```

请注意，多轮对话目前仅支持语音问答。当看到“Inference start”，可通过按空格键开始和停止录音。
