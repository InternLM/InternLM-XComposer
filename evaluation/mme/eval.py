import torch
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import generate_answer

tgt_dir = 'PATH TO MODEL'
hf_tokenizer = AutoTokenizer.from_pretrained(tgt_dir, trust_remote_code=True)
hf_model = AutoModel.from_pretrained(tgt_dir, trust_remote_code=True)
hf_model.cuda()
hf_model.eval()
for n, p in hf_model.named_parameters():
    p.requires_grad = False

hf_model.tokenizer = hf_tokenizer
model = hf_model


root = 'MME_PATH/eval_tool/Your_Results'
output = 'MME_PATH/eval_tool/InternLM-XComposer-VL'
os.makedirs(output, exist_ok=True)


for filename in os.listdir(root):
    with open(os.path.join(root, filename), 'r') as fin, open(os.path.join(output, filename), 'w') as fout:
        lines = fin.read().splitlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            try:
                img_path = os.path.join('MME_IMG_PATH', filename, img)
                assert os.path.exists(img_path), img_path
            except:
                img_path = os.path.join('MME_IMG_PATH', filename, 'images', img)
                assert os.path.exists(img_path), img_path
            text = f' <|User|>:<ImageHere> {question} Answer this question briefly' + hf_model.eoh + ' <|Bot|>:'
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    response = generate_answer(model, text, img_path)
            print(img, question, gt, response, sep='\t', file=fout)
