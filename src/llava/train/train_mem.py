import sys
import os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
# from llava.train.train import train
from src.llava.train.train import train


if __name__ == "__main__":
    try:
        import flash_attn
        attn_implementation = 'flash_attention_2'
    except ModuleNotFoundError:
        attn_implementation = 'sdpa'
        print('Cannot import flash_attn. Using SDPA attention')
        # cuDNN flash SDP kernel fails during gradient checkpointing recomputation;
        # disable it so SDPA falls back to the memory-efficient kernel.
        import torch
        torch.backends.cuda.enable_flash_sdp(False)

    train(attn_implementation=attn_implementation)
