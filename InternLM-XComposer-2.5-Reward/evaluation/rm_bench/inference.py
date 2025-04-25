import os
import json
import numpy as np
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    # results is a list of dictionaries, each dictionary contains the following keys:
    # score_chosen: [float, float, float], the scores of the chosen responses
    # score_rejected: [float, float, float], the scores of the rejected responses
    # the scores are in the order of [concise, detailed_plain, detailed_markdown]
    # we will compare the scores of chosen responses and rejected responses iteratively
    # formatted as a 3x3 matrix, where the rows represent the scores of chosen responses
    # and the columns represent the scores of rejected responses
    MATRIX_SIZE = 3 # the column and row size of the matrix
    acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for result in results:
        for i in range(len(result["score_chosen"])):
            for j in range(len(result["score_rejected"])):
                if result["score_chosen"][i] > result["score_rejected"][j]:
                    acc_matrix[i][j] += 1
    
    # compute the accuracy by dividing the number of correct comparisons by the total number of comparisons
    acc_matrix /= len(results)
    # compute the hard,normal,easy accuracy
    # hard accuracy: the average of the upper-right triangle of the matrix
    # namely chosen responses with less fancy style compared to rejected responses with more fancy style
    upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count
    # normal accuracy: the average of the diagonal of the matrix
    # namely chosen responses with the same style compared to rejected responses with the same style
    normal_acc = np.mean(np.diag(acc_matrix))
    # easy accuracy: the average of the lower-left triangle of the matrix
    # namely chosen responses with more fancy style compared to rejected responses with less fancy style
    lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count
    
    return {
        "hard_acc": hard_acc,
        "normal_acc": normal_acc,
        "easy_acc": easy_acc,
        "avg_acc": (easy_acc + normal_acc + hard_acc) / 3,
    }


torch.set_grad_enabled(False)

ckpt_path = "internlm/internlm-xcomposer2d5-7b-reward"
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model.tokenizer = tokenizer

random_seed = 0
set_seed(random_seed)

json_path = './total_dataset.json'
with open(json_path, 'r') as f:
    data = json.load(f)
print(len(data))

max_length = 16384
hd_num = 9

results = []
for ind in tqdm(range(len(data))):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        chat_1_1 = [
            {"role": "user", "content": data[ind]['prompt']},
            {"role": "assistant", "content": data[ind]['chosen'][0]}
        ]
        chat_1_2 = [
            {"role": "user", "content": data[ind]['prompt']},
            {"role": "assistant", "content": data[ind]['chosen'][1]}
        ]
        chat_1_3 = [
            {"role": "user", "content": data[ind]['prompt']},
            {"role": "assistant", "content": data[ind]['chosen'][2]}
        ]
        chat_2_1 = [
            {"role": "user", "content": data[ind]['prompt']},
            {"role": "assistant", "content": data[ind]['rejected'][0]}
        ]
        chat_2_2 = [
            {"role": "user", "content": data[ind]['prompt']},
            {"role": "assistant", "content": data[ind]['rejected'][1]}
        ]
        chat_2_3 = [
            {"role": "user", "content": data[ind]['prompt']},
            {"role": "assistant", "content": data[ind]['rejected'][2]}
        ]

        scores = model.get_scores([chat_1_1, chat_1_2, chat_1_3, chat_2_1, chat_2_2, chat_2_3], [[]] * 6, max_length, hd_num)

    result = {
        'score_chosen': scores[:3],
        'score_rejected': scores[3:],
    }
    results.append(result)

    with open(f'./results.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

accuracy = compute_accuracy(results)
print(accuracy)