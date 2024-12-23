Here's a polished version of your GitHub repository introduction:

---

# RWKV-SpeechChat

RWKV-SpeechChat is a real-time dialogue script based on a frozen 3B RWKV model with trained adapters and initial states. The corresponding training framework is available here: https://github.com/AGENDD/RWKV-ASR, providing more detailed descriptions. Various trained weights can be applied to perform a range of audio tasks, including automatic speech recognition (ASR), speech translation, speech question answering (QA), and more.

## Features

- **Multi Audio Task Support**: Supports multiple audio tasks, including automatic speech recognition (ASR), speech translation, speech question answering (QA), and more coming soon.
- **Local Deployment**: Can be run on a PC with a GPU that has at least 6GB of video memory.
- **Real-time Conversation**: Supports real-time conversation with the model, similar to GPT-4.

## Demonstration

Here are some video demonstartion of speech QA task in English and Chinese.

<video width="320" height="240" src = "videos/video1.mp4" controls>
  Your browser does not support the video tag.
</video>
<video width="320" height="240" src = "videos/video2.mp4" controls>
  Your browser does not support the video tag.
</video>
<video width="320" height="240" src = "videos/video3.mp4" controls>
  Your browser does not support the video tag.
</video>



## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AGENDD/RWKV-SpeechChat.git
    cd RWKV-SpeechChat
    ```

2. Download RWKV model weights:

    Download the RWKV model weights from:
    https://huggingface.co/BlinkDL/rwkv-6-world/tree/main
    
    This project currently supports "RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth" only. Place the weights in the `model` directory.

3. Download trained weights:

    Download the trained weights corresponding to audio tasks:

    - ASR: https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/ASR
    - ST: https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/ST
    - SpeechQA: https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/SpeechQA

    Place the weights in the `model` directory.

## Usage

### Command-line Arguments

- `--multiturns`: Enable multi-turn conversation mode (remove this to disable multi-turn conversation).
- `--rwkv_path`: Path to RWKV model weights (default is `model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth`).
- `--weights_path`: Path to trained weights (default is `model/rwkv-adapter-speechQA-VoiceAssistant-final.pth`).

### Running the Script

You can run the script with the following command:

```bash
python main.py --multiturns --rwkv_path path/to/your/model/weights.pth --weights_path path/to/your/trained/weights.pth
```

Or use the default parameters:

```bash
python main.py
```

Note that multi-turn conversation currently only supports speech QA.

