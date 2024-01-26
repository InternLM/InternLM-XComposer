import io
import base64
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import os
from typing import Optional
import xlsxwriter
import pandas as pd
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset
import torchvision

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

def generate_answer(model, text, image_path):
    image = Image.open(image_path).convert("RGB")
    image = model.vis_processor(image).unsqueeze(0).to(model.device)
    img_embeds = model.encode_img(image)
    prompt_segs = text.split('<ImageHere>')
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
    
    outputs = model.internlm_model.generate(
        inputs_embeds=prompt_embs,
        max_new_tokens=5,
        num_beams=5,
        do_sample=False,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1.0,
        stopping_criteria=stopping_criteria,
    )
    #print (outputs)
    output_token = outputs[0]
    if output_token[0] == 0:
        output_token = output_token[1:]
    if output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token,
                                              add_special_tokens=False)

    output_text = output_text.split(model.eoa)[0]
    output_text = output_text.split('<|Bot|>')[-1].strip()
    return output_text

class MMDump():
    def __init__(self,
                 save_path: str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        self.save_path = save_path
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.results = []

    def process(self, data_sample) -> None:
        result = dict()

        result['question'] = data_sample.get('question')
        result['answer'] = data_sample.get('answer')
        result['options'] = data_sample.get('options')
        result['prediction'] = data_sample.get('pred_answer')
        if data_sample.get('category') is not None:
            result['category'] = data_sample.get('category')
        if data_sample.get('l2-category') is not None:
            result['l2-category'] = data_sample.get('l2-category')
        result['index'] = data_sample.get('index')
        self.results.append(result)

    def save_results(self) -> dict:
        df = pd.DataFrame(self.results)
        with pd.ExcelWriter(self.save_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        return {}

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = io.BytesIO(image_data)
    return image


class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,):
        self.df = pd.read_csv(data_file, sep='\t')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)

        question = self.df.iloc[idx]['question']
        try:
            answer = self.df.iloc[idx]['answer']
        except:
            answer = ''

        catetory = self.df.iloc[idx]['category']
        try:
            l2_catetory = self.df.iloc[idx]['l2-category']
        except:
            l2_catetory = 'CN'
        options = [self.df.iloc[idx]['A'], self.df.iloc[idx]['B']]
        options_prompt = f'A. {options[0]}\nB. {options[1]}\n'  # noqa
        if 'C' in self.df.iloc[idx]:
            options.append(self.df.iloc[idx]['C'])
            options_prompt = options_prompt + f'C. {options[2]}\n'
        if 'D' in self.df.iloc[idx]:
            options.append(self.df.iloc[idx]['D'])
            options_prompt = options_prompt + f'D. {options[3]}\n'
        if 'E' in self.df.iloc[idx]:
            options.append(self.df.iloc[idx]['E'])
            options_prompt = options_prompt + f'E. {options[4]}\n'

        try:
            hint = None if pd.isna(
                self.df.iloc[idx]['hint']) else self.df.iloc[idx]['hint']  # noqa
        except:
            hint = None

        img_prompt = ' <|User|>:<ImageHere>'
        txt_prompt = 'Please answer this question by choosing the correct choice.'
        context = 'N/A' if hint is None else hint
        mid_prompt = 'Context: ' + context + '\nQuestion: ' + question + '\nOptions: ' + options_prompt
        ans_prompt = ' <|Bot|>: Answer: The answer is'
        text = img_prompt + txt_prompt + mid_prompt + '<TOKENS_UNUSED_0>' + ans_prompt
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_list': options,
            'index': index,
            'context': hint,
            'input_text': text,
        }
        return data
