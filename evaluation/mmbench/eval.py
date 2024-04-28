import torch
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import model_gen, MMDump, MMBenchDataset

mmbench = MMBenchDataset('data/mmbench_test_20230712.tsv')
mm_dump = MMDump(save_path = '../Output/submit_test.xlsx')

tgt_dir = 'internlm/internlm-xcomposer2-vl-7b'
tokenizer = AutoTokenizer.from_pretrained(tgt_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(tgt_dir, trust_remote_code=True)
model.cuda().eval().half()
model.tokenizer = tokenizer

for sample in tqdm(mmbench):
    image = sample['img']
    text = sample['text']
    with torch.cuda.amp.autocast():
        with torch.no_grad(): 
            response = model_gen(model, text, image) 
            #print (response)
    sample['pred_answer'] = response
    mm_dump.process(sample)
mm_dump.save_results()


