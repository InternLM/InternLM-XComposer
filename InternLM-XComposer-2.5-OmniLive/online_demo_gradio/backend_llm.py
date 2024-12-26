import base64
import io
import time
import os
import socket
import re
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from queue import Queue
import nest_asyncio
nest_asyncio.apply()
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import torchvision.transforms as transforms


hostname = socket.gethostname()
ip_addr = socket.gethostbyname(hostname)
os.environ["no_proxy"] = f"localhost,127.0.0.1,::1,{ip_addr}"


class Item(BaseModel):
    query: str
    time_dict: dict


def img_process(imgs):
    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w/h
        if w > h:
            new_w = 560 * 2
            new_h = int(560 * 2 / scale)
        else:
            new_w = int(560 * 2 * scale)
            new_h = 560 * 2
        img = transforms.functional.resize(img, [new_h, new_w],)
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w,h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        font = ImageFont.truetype("SimHei.ttf", pad)
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w,h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h ), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(0, pad +curr_h + h +5), (new_w, pad +curr_h + h +5)], fill = 'black', width=2)
            curr_h += h + 10 + pad
        #print (new_w, new_h)
    else:
        for im in imgs:
            w,h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        font = ImageFont.truetype("SimHei.ttf", pad)
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w,h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill = 'black', width=2)
            curr_w += w + 10
    return new_img


class IXC_Client():
    def __init__(self, tp=1):
        hf_model = 'internlm-xcomposer2d5-ol-7b/merge_lora'
        if tp > 1:
            backend_config = TurbomindEngineConfig(tp=tp)
            self.pipe = pipeline(hf_model, backend_config=backend_config)
        else:
            self.pipe = pipeline(hf_model)
        self.pipe.chat_template.meta_instruction = """You are an 人工智能 assistant whose name is 浦语·灵笔.
- 浦语·灵笔 is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- 浦语·灵笔 can understand and communicate fluently in the language chosen by the user such as English and 中文.
- 浦语·灵笔 is capable of comprehending and articulating responses effectively based on the provided image.
"""
        #     pipe.chat_template.meta_instruction = """你是一个多模态人工智能助手，名字叫浦语·灵笔。
        # - 浦语·灵笔是由上海人工智能实验室开发的一个多模态对话模型，是一个有用，真实且无害的模型。
        # - 浦语·灵笔可以根据看到和听到的内容，流利的同用户进行交流，并使用用户使用的语言（中文或英文）进行回复。
        # """
        print('IXC init finished!!!')
        self.gen_config = GenerationConfig(top_k=50, top_p=0.8, temperature=0.1)

        self.querys = Queue()
        #self.anwsers = Queue()
        llm_inference = threading.Thread(target=self.run)
        llm_inference.start()

    def run(self):
        while True:
            if self.querys.empty():
                time.sleep(0.15)
            else:
                item = self.querys.get()
                query, time_dict = item.query, item.time_dict

                st = time.time()
                query_list = query.split('<##>')
                if len(query_list) == 1:
                    query = query_nopunc = query_list[0]
                elif len(query_list) == 2:
                    query_nopunc = query_list[1]
                    query = query_list[0] + query_list[1]

                inst_yes = False
                query_nopunc = query_nopunc[:-1] if query_nopunc[-1] in ['?', '？'] else query_nopunc
                mid_prompt = f'''Is the following string an instruction? "{query_nopunc}"'''
                inst_resp = self.pipe.stream_infer((mid_prompt, []), gen_config=self.gen_config)
                if inst_resp:
                    for chunk in inst_resp:
                        if chunk.text:
                            if "Yes" in chunk.text:
                                inst_yes = True
                                break

                time_dict['instruct_t'] = time.time() - st
                st = time.time()
                print(f"model instruct output:  ========={inst_yes}=========")

                if True: #inst_yes:
                    folder = self.prepare_grounding(query_nopunc)
                    imgs = torch.load(os.path.join(folder, 'temp_lol_img.pth'))
                    imgs = imgs[::2]

                    if len(imgs) > 0:
                        img = img_process(imgs)
                        img = [img]
                    else:
                        img = []

                    response = self.pipe.stream_infer((query, img), gen_config=self.gen_config)
                    if response:
                        output = ''
                        start_idx = 0

                        for chunk in response:
                            print(chunk)
                            if chunk.text:
                                output += chunk.text

                                match = re.search(r'[.\n。？！?!]', output[start_idx:])
                                if match:
                                    match_idx = start_idx + match.start()
                                    temp_text = output[start_idx:match_idx + 1]
                                    start_idx = match_idx + 1
                                    if 'llm_first_chunck' not in time_dict:
                                        time_dict['llm_first_chunck'] = time.time() - st
                                    #self.anwsers.put(temp_text)
                                    self.send_answer(temp_text)
                        if start_idx < len(output):
                            #self.anwsers.put(output[start_idx:] + '。')
                            self.send_answer(output[start_idx:] + '。')
                        print(f"model output: {output}")

                        if 'llm_first_chunck' not in time_dict:
                            time_dict['llm_first_chunck'] = 0

                        print(f"==============time start==================")
                        print('''
                        ast_t:{:.2f}s
                        instruct_t:{:.2f}s
                        llm_first_chunck:{:.2f}s'''.format(time_dict['ast_t'], time_dict['instruct_t'], time_dict['llm_first_chunck'],))
                        print(f"==============time end==================")

    def prepare_grounding(self, query):
        query = {'query': query}
        response = requests.post(f"http://{ip_addr}:8002/send_query", params=query, timeout=30)
        folder = eval(response.content.decode('utf-8'))
        return folder


    def send_answer(self, text):
        query = {'text': text}
        requests.post(f"http://{ip_addr}:8000/recv_llm", params=query)


ixc_client = IXC_Client()

# Initialize FastAPI app
app = FastAPI()


@app.post("/send_ixc")
async def enqueue_query(item: Item):
    ixc_client.querys.put(item)
    return True


if __name__ == "__main__":
    uvicorn.run(app, host=ip_addr, port=8001)