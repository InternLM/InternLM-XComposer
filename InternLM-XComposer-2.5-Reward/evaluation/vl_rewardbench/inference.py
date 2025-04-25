import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import json
import random


torch.set_grad_enabled(False)

ckpt_path = "internlm/internlm-xcomposer2d5-7b-reward"

model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model.tokenizer = tokenizer

random_seed = 0
set_seed(random_seed)

jsonl_path = './combined_data_tagged.jsonl'
data = []
with open(jsonl_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

random_numbers = []
out_all = []
out_hallucination = []
out_reasoning = []
out_general = []

max_length = 16384
hd_num = 9

for ind in tqdm(range(len(data))):
    image = [data[ind]['image_path']]

    random_number = random.choice([0, 1])
    random_numbers.append(random_number)

    if random_number == 0:
        response_1 = data[ind]['response'][0]
        response_2 = data[ind]['response'][1]
        answer = data[ind]['human_ranking']
    else:
        response_1 = data[ind]['response'][1]
        response_2 = data[ind]['response'][0]
        answer = data[ind]['human_ranking'][::-1]

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        chat_1 = [
            {"role": "user", "content": data[ind]['query']},
            {"role": "assistant", "content": response_1}
        ]
        chat_2 = [
            {"role": "user", "content": data[ind]['query']},
            {"role": "assistant", "content": response_2}
        ]
        rank_res = model.rank([chat_1, chat_2], [image, image], max_length, hd_num)

    if rank_res == answer:
        out = 1
    else:
        out = 0

    out_all.append(out)
    if 'reasoning_tasks' in data[ind]['image_path']:
        out_reasoning.append(out)
    elif 'vlfeedback' in data[ind]['image_path'] or 'wildvision-battle' in data[ind]['image_path']:
        out_general.append(out)
    elif 'povid' in data[ind]['image_path'] or 'rlhf-v' in data[ind]['image_path'] or 'rlaif-v' in data[ind]['image_path']:
        out_hallucination.append(out)
    else:
        raise NotImplementedError

    output_data = {
        "out_all": out_all,
        "out_hallucination": out_hallucination,
        "out_reasoning": out_reasoning,
        "out_general": out_general,
        "random_numbers": random_numbers,
    }

    with open(f'./results.json', 'w') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


out_all = output_data['out_all']
out_general = output_data['out_general']
out_hallucination = output_data['out_hallucination']
out_reasoning = output_data['out_reasoning']

overall_acc = sum(out_all) / len(out_all)
general_acc = sum(out_general) / len(out_general)
hallucination_acc = sum(out_hallucination) / len(out_hallucination)
reasoning_acc = sum(out_reasoning) / (len(out_reasoning) + 1e-7)
macro_acc = (general_acc + reasoning_acc + hallucination_acc) / 3

print(len(out_all), len(out_general), len(out_hallucination), len(out_reasoning))
print(f'General: {general_acc} Hallu: {hallucination_acc} Reasoning: {reasoning_acc} Overall: {overall_acc} Macro: {macro_acc}')