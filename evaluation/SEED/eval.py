import os
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import SeedDataset, model_gen

seed_bench = SeedDataset('../data/SEED-Bench-image',
                        '../data/SEED-Bench.json')

tgt_dir = 'internlm/internlm-xcomposer2-vl-7b'
tokenizer = AutoTokenizer.from_pretrained(tgt_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(tgt_dir, trust_remote_code=True)
model.cuda().eval().half()
model.tokenizer = tokenizer

answer_list = []

for sample in tqdm(seed_bench):
    image = sample['img']
    text = sample['text']
    with torch.cuda.amp.autocast():
        with torch.no_grad(): 
            response = model_gen(model, text, image) 
        
    answer_record = {
            'question_id': sample['index'] ,
            'prediction': response,
            'answer': sample['answer'] ,
    }
    answer_list.append(answer_record)
    #if len(answer_list) > 200:
    #    break

score = 0
for i, answer_record in enumerate(answer_list):
    if answer_record['answer'] == answer_record['prediction']:
        score+=1
print(score/len(answer_list))
