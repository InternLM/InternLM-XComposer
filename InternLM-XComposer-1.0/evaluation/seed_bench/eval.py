import torch
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import generate_answer_with_ppl, SeedDataset
seed_bench = SeedDataset('PATH TO SEED IMAGE')

tgt_dir = 'PATH TO MODEL'
hf_tokenizer = AutoTokenizer.from_pretrained(tgt_dir, trust_remote_code=True)
hf_model = AutoModel.from_pretrained(tgt_dir, trust_remote_code=True)
hf_model.cuda()
hf_model.eval()
for n, p in hf_model.named_parameters():
    p.requires_grad = False
hf_model.tokenizer = hf_tokenizer
model = hf_model

answer_list = []

for sample in tqdm(seed_bench):
    image = sample['img']
    text = sample['text_input']

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response = generate_answer_with_ppl(model, text, image)
        
    answer_record = {
            'question_id': sample['index'] ,
            'prediction': response,
            'answer': sample['answer'] ,
    }
    answer_list.append(answer_record)
    
score = 0
for i, answer_record in enumerate(answer_list):
    if answer_record['answer'] == answer_record['prediction']:
        score+=1
print(score/len(answer_list))
