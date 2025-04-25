import pandas as pd
from collections import defaultdict
import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


torch.set_grad_enabled(False)

ckpt_path = "internlm/internlm-xcomposer2d5-7b-reward"
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model.tokenizer = tokenizer

random_seed = 0
set_seed(random_seed)

path = './filtered-00000-of-00001.parquet'
data = pd.read_parquet(path)


SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

REVERSED_SUBSET_MAPPING = {}

for key, values in SUBSET_MAPPING.items():
    for value in values:
        REVERSED_SUBSET_MAPPING[value] = key

correct = defaultdict(int)
count = defaultdict(int)
results = []

for ind in tqdm(range(len(data))):
    prompt = data['prompt'][ind]
    chosen = data['chosen'][ind]
    rejected = data['rejected'][ind]
    subset = REVERSED_SUBSET_MAPPING[data['subset'][ind]]

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        chat_1 = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        chat_2 = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected}
        ]
        score = model.get_scores([chat_1, chat_2], [[]] * 2)

        if score[0] > score[1]:
            correct[subset] += 1
            correct['all'] += 1
        
        count[subset] += 1
        count['all'] += 1

        result = {
            'score_chosen': score[0],
            'score_rejected': score[1],
        }
        results.append(result)

        with open(f'./results.json', 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

for key in correct.keys():
    accuracy = correct[key] / count[key]
    print(key, accuracy)