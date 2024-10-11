import torch
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import generate_answer, MMDump, MMBenchDataset

mmbench = MMBenchDataset('PATH TO MMBENCH TSV')
mm_dump = MMDump(save_path = './test.xlsx')

tgt_dir = 'PATH TO MODEL'
hf_tokenizer = AutoTokenizer.from_pretrained(tgt_dir, trust_remote_code=True)
hf_model = AutoModel.from_pretrained(tgt_dir, trust_remote_code=True)
hf_model.cuda()
hf_model.eval()
for n, p in hf_model.named_parameters():
    p.requires_grad = False
hf_model.tokenizer = hf_tokenizer
model = hf_model

for sample in tqdm(mmbench):
    image = sample['img']
    text = sample['input_text']
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response = generate_answer(model, text, image)
    sample['pred_answer'] = response
    mm_dump.process(sample)
mm_dump.save_results()
