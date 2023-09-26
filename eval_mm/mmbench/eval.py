import torch
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import generate_answer, MMDump, MMBenchDataset

mmbench = MMBenchDataset('/mnt/petrelfs/share_data/liuyuan/data/mm/mmbench/mmbench_dev_20230712.tsv')
mm_dump = MMDump(save_path = '/mnt/petrelfs/dongxiaoyi/dataset/eval_tool/submit_dev_final.xlsx')

tgt_dir = '/mnt/petrelfs/share_data/dongxiaoyi/share_models/release_performance'
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
