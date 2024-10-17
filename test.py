from transformers import AutoTokenizer, AutoModelForCausalLM
from src.asr import SLAM_ASR
import torch
from torch.nn import functional as F

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from datasets import load_dataset

lm_model = RWKV(model='model/RWKV-x060-World-3B-v2.1-20240417-ctx4096', strategy='cuda bf16')
lm_model = lm_model.to(dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-6-world-1b6",trust_remote_code=True)


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy")['validation'].shuffle()
print(ds)

with torch.no_grad():
    for data in ds:
        transcript = data["text"].lower()

        prompt = tokenizer.encode(f"User:{transcript}\n\nAssistant:")
        out, state = lm_model.forward(tokens=prompt ,state=None)    
        MAX_LENGTH = 100
        true_output = []
        # print("character:",end="")
        for i in range(MAX_LENGTH):
            # print(f"logit:{out.shape}")
            probabilities = F.softmax(out, dim=-1)
            _, top_idx = probabilities.topk(1, dim=-1)
            # print(f"token:{top_idx}")
            decoded_token = tokenizer.decode(top_idx)
            # print(f"{decoded_token}",end="")
            if decoded_token == '<s>':
                break
            else:
                true_output.append(decoded_token)
        
            out, state = lm_model.forward(tokens=top_idx,state=state)
        
        print("transcript:")
        print(transcript)
        print("response:")
        print(''.join(true_output))
        print("\n\n")
        
        
