from torch.utils.data import Dataset
import base64
import os.path as osp
import torch
import numpy as np
import torchvision
from PIL import Image
import re
pattern = re.compile(r'[A-D]')

class SeedDataset(Dataset):
    def __init__(self,
                 im_path,
                 json_path,
                 ):
        self.im_path = im_path
        temps = json.load(open(json_path, 'r'))
        self.samples = [temp for temp in temps['questions'] if temp['data_type'] == 'image']
        self.q_types = {}
        for k, v in temps['question_type'].items():
            self.q_types[v] = k

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        index = sample['question_id']
        q_index = sample['question_type_id']
        q_type = self.q_types[q_index]

        data_path = osp.join(self.im_path, sample['data_id'])

        question = sample['question']
        answer = sample['answer']

        options = [sample['choice_a'], sample['choice_b'], ]
        options_prompt = f'A. {options[0]} B. {options[1]} '  # noqa
        if 'choice_c' in sample:
            options.append(sample['choice_c'])
            options_prompt = options_prompt + f'C. {options[2]} '
        if 'choice_d' in sample:
            options.append(sample['choice_d'])
            options_prompt = options_prompt + f'D. {options[3]} '
        if 'choice_e' in sample:
            options.append(sample['choice_e'])
            options_prompt = options_prompt + f'E. {options[4]} '

        img_prompt = '[UNUSED_TOKEN_146]user\n'
        context = 'N/A'
        options_prompt = options_prompt.strip()
        mid_prompt = 'Question: ' + question + '\nContext: ' + context + '\nOptions: ' + options_prompt
        ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'
        text = img_prompt + mid_prompt + ans_prompt

        data = {
            'img': data_path,
            'text': text,
            'answer': answer,
        }
        return data

def model_gen( model, text, images, need_bos=True):
    pt1 = 0
    embeds = []
    im_mask = []
    images = [images]
    images_loc = [0]
    for i, pts in enumerate(images_loc + [len(text)]):
        subtext = text[pt1:pts]
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            need_bos = False
        if i < len(images):
            image = Image.open(images[i]).convert('RGB')
            image = model.vis_processor(image).unsqueeze(0).cuda()
            image_embeds = model.encode_img(image)
            embeds.append(image_embeds)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                        temperature=1.0, max_new_tokens=5, num_beams=5,
                        do_sample=False, repetition_penalty=1.0)

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0]
    res = pattern.findall(output_text)
    if len(res) == 0:
        print('Error:', output_text); res = 'E'
    return res[0]
