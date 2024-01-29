from typing import Optional
from torch.utils.data import Dataset
import base64
import os.path as osp
import pandas as pd
import xlsxwriter
from PIL import Image
import io
import re
pattern = re.compile(r'[A-D]')

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
        options_prompt = f'A. {options[0]} B. {options[1]} '  # noqa
        if 'C' in self.df.iloc[idx]:
            options.append(self.df.iloc[idx]['C'])
            options_prompt = options_prompt + f'C. {options[2]} '
        if 'D' in self.df.iloc[idx]:
            options.append(self.df.iloc[idx]['D'])
            options_prompt = options_prompt + f'D. {options[3]} '
        if 'E' in self.df.iloc[idx]:
            options.append(self.df.iloc[idx]['E'])
            options_prompt = options_prompt + f'E. {options[4]} '

        try:
            hint = None if pd.isna(
                self.df.iloc[idx]['hint']) else self.df.iloc[idx]['hint']  # noqa
        except:
            hint = None
        
        img_prompt = '[UNUSED_TOKEN_146]user\n'
        context = 'N/A' if hint is None else hint
        options_prompt = options_prompt.strip()
        mid_prompt = 'Question: ' + question + '\nContext: ' + context + '\nOptions: ' + options_prompt
        ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'
        text = img_prompt + mid_prompt + ans_prompt
        
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
            'text_input': text,
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
