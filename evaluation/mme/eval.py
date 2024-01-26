import torch
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import model_gen

tgt_dir = 'internlm/internlm-xcomposer2-vl-7b'
tokenizer = AutoTokenizer.from_pretrained(tgt_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(tgt_dir, trust_remote_code=True)
model.cuda().eval().half()
model.tokenizer = tokenizer

eval_tool_path = '/mnt/petrelfs/dongxiaoyi/dataset/eval_tool/'
mme_data_path = '/mnt/petrelfs/dongxiaoyi/dataset/MME_Benchmark_release'

root = os.path.join(eval_tool_path, 'Your_Results')
output = os.path.join(eval_tool_path, 'InternLM_XComposer2_VL')

os.makedirs(output, exist_ok=True)


for filename in os.listdir(root):
    with open(os.path.join(root, filename), 'r') as fin, open(os.path.join(output, filename), 'w') as fout:
        lines = fin.read().splitlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            try:
                img_path = os.path.join(mme_data_path, filename, img)
                assert os.path.exists(img_path), img_path
            except:
                img_path = os.path.join(mme_data_path, filename, 'images', img)
                assert os.path.exists(img_path), img_path
            with torch.cuda.amp.autocast():
                response = model_gen(model, question, img_path)
            print(img, question, gt, response, sep='\t', file=fout)
