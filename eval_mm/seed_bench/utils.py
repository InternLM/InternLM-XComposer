import io
import base64
import torch
import json
from transformers import StoppingCriteria, StoppingCriteriaList
import os
from typing import Optional
import pandas as pd
from PIL import Image
import numpy as np

import pandas as pd
from torch.utils.data import Dataset
import torchvision
import os.path as osp

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

stop_words_ids = [
                  torch.tensor([103027]).cuda(), ### end of human
                  torch.tensor([103028]).cuda(), ### end of bot
                 ]
stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)])

def generate_answer_with_ppl(model, base_prompt, image):
    choice_mapping = ['A.', 'B.', 'C.', 'D.']
    img_embeds = model.encode_img(image)
    prompt = base_prompt
    prompt_segs = prompt.split('<ImageHere>')
    prompt_seg_tokens = [
        model.tokenizer(seg,
                             return_tensors='pt',
                             add_special_tokens=i == 0).
        to(model.internlm_model.model.embed_tokens.weight.device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]
    prompt_seg_embs = [
        model.internlm_model.model.embed_tokens(seg)
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
    prompt_embs = torch.cat(prompt_seg_embs, dim=1)

    im_targets = torch.ones(img_embeds.shape[0], img_embeds.shape[1], dtype=torch.long).to(img_embeds.device) * model.tokenizer.pad_token_id
    tars = torch.cat([prompt_seg_tokens[0], im_targets, prompt_seg_tokens[1]], dim=1)
    atts_mask = torch.ones(prompt_embs.size()[:-1], dtype=torch.long).to(model.device)

    len_prompt = tars.shape[1] - 1
    candis = choice_mapping
    op_tokens = model.tokenizer(candis, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    op_embeds = model.internlm_model.model.embed_tokens(op_tokens.input_ids)
    tars = torch.cat([tars.repeat(4,1), op_tokens.input_ids], dim=1)
    atts_mask = torch.cat([atts_mask.repeat(4,1), op_tokens.attention_mask], dim=1)
    prompt_embs = torch.cat([prompt_embs.repeat(4,1,1), op_embeds], dim=1)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = model.internlm_model(
                inputs_embeds=prompt_embs,
                attention_mask=atts_mask,
                return_dict=True,
                labels=None,
            )
    outputs = outputs.logits
    shift_logits = outputs[..., len_prompt:-1, :].contiguous()
    shift_labels = tars[..., 1+len_prompt:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    loss_fct = torch.nn.CrossEntropyLoss(
        reduction='none', ignore_index=model.tokenizer.pad_token_id)
    sf = torch.nn.functional.softmax(shift_logits, dim=-1)

    loss = loss_fct(shift_logits, shift_labels.view(-1)).reshape(shift_labels.shape[0], shift_labels.shape[1])
    mask_length = None

    lens = (shift_labels !=
            model.tokenizer.pad_token_id).sum(-1).cpu().numpy()
    ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
    outs = ce_loss
    idx = np.argmin(ce_loss)
    output_text = choice_mapping[idx]
    return output_text[0]

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


class SeedDataset(Dataset):
    def __init__(self,
                 cc3m_path,
                 ):
        self.cc3m_path = cc3m_path
        temps = json.load(open('/mnt/petrelfs/share_data/fangyixiao/mm_data_protocol/SEED-Bench/data/SEED-Bench.json', 'r'))
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

        data_path = osp.join(self.cc3m_path, sample['data_id'])

        question = sample['question']
        answer = sample['answer']

        options = [sample['choice_a'], sample['choice_b'], ]
        options_prompt = f'A. {options[0]}\nB. {options[1]}\n'  # noqa
        if 'choice_c' in sample:
            options.append(sample['choice_c'])
            options_prompt = options_prompt + f'C. {options[2]}\n'
        if 'choice_d' in sample:
            options.append(sample['choice_d'])
            options_prompt = options_prompt + f'D. {options[3]}\n'
        if 'choice_e' in sample:
            options.append(sample['choice_e'])
            options_prompt = options_prompt + f'E. {options[4]}\n'

        img_prompt = ' <|User|>:<ImageHere>'
        context = 'N/A'
        
        mid_prompt = 'Context: ' + context + '\nQuestion: ' + question + '\nOptions: []\n' + options_prompt
        ans_prompt = ' <|Bot|>: Answer: The answer is'
        text = img_prompt + mid_prompt + '<TOKENS_UNUSED_0>' + ans_prompt

        data = {
            'img': data_path,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'index': index,
            'context': None,
            'category': q_type,
            'text_input': text
        }
        return data

