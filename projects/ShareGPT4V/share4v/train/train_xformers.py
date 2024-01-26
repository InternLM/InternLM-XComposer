# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.

# Need to call this before importing transformers.
from share4v.train.train import train
from share4v.train.llama_xformers_attn_monkey_patch import \
    replace_llama_attn_with_xformers_attn

replace_llama_attn_with_xformers_attn()


if __name__ == "__main__":
    train()
