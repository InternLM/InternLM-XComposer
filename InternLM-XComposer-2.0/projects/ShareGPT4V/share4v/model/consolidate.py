"""
Usage:
python3 -m share4v.model.consolidate --src ~/model_weights/share4v-7b --dst ~/model_weights/share4v-7b_consolidate
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from share4v.model import *
from share4v.model.utils import auto_upgrade


def consolidate_ckpt(src_path, dst_path):
    print("Loading model")
    auto_upgrade(src_path)
    src_model = AutoModelForCausalLM.from_pretrained(
        src_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    src_tokenizer = AutoTokenizer.from_pretrained(src_path, use_fast=False)
    src_model.save_pretrained(dst_path)
    src_tokenizer.save_pretrained(dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)

    args = parser.parse_args()

    consolidate_ckpt(args.src, args.dst)
