import os
import re
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import argparse
import gradio as gr
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), 'tmp')
import copy
import time
from datetime import datetime
import hashlib
import shutil
import requests
from PIL import Image, ImageFile
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

from demo_asset.assets.css_html_js import custom_css
from demo_asset.serve_utils import Stream, Iteratorize
from demo_asset.conversation import CONV_VISION_INTERN2
from demo_asset.download import download_image_thread
from examples.utils import get_stopping_criteria, set_random_seed


meta_instruction = """You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
chat_meta = """You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.
"""



max_section = 60
chat_stream_output = True
article_stream_output = True


def get_urls(caption, exclude):
    headers = {'Content-Type': 'application/json'}
    json_data = {'caption': caption, 'exclude': exclude, 'need_idxs': True}
    response = requests.post('https://lingbi.openxlab.org.cn/image/similar',
                             headers=headers,
                             json=json_data)
    urls = response.json()['data']['image_urls']
    idx = response.json()['data']['indices']
    return urls, idx


class ImageGroup(object):
    def __init__(self, cap, paths, pts=0):
        #assert len(paths) == 1 or len(paths) == 4, f"ImageGroup only support 1 or 4 images, not {len(paths)} images"
        self.cap = cap
        self.paths = paths
        self.pts = pts

    def __str__(self):
        return f"cap: {self.cap}; paths:{self.paths}; pts:{self.pts}"


class ImageProcessor:
    def __init__(self, image_size=224):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        if isinstance(item, str):
            item = Image.open(item).convert('RGB')
        return self.transform(item)

class Demo_UI:
    def __init__(self, code_path, num_gpus=1):
        self.code_path = code_path
        self.reset()

        tokenizer = AutoTokenizer.from_pretrained(code_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(code_path, device_map='cuda', trust_remote_code=True).half().eval()
        self.model.tokenizer = tokenizer
        self.model.vit.resize_pos()

        self.vis_processor = ImageProcessor()

        stop_words_ids = [92397]
        #stop_words_ids = [92542]
        self.stopping_criteria = get_stopping_criteria(stop_words_ids)
        set_random_seed(1234)
        self.r2 = re.compile(r'<Seg[0-9]*>')
        self.withmeta = False

    def reset(self):
        self.pt = 0
        self.img_pt = 0
        self.texts_imgs = []
        self.open_edit = False
        self.hash_folder = '12345'
        self.instruction = ''

    def text2instruction(self, text):
        if self.withmeta:
            return f"[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"
        else:
            return f"[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"

    def get_images_xlab(self, caption, pt, exclude):
        urls, idxs = get_urls(caption.strip()[:53], exclude)
        print(urls[0])
        print('download image with url')
        download_image_thread(urls,
                              folder='articles/' + self.hash_folder,
                              index=pt,
                              num_processes=4)
        print('image downloaded')
        return idxs

    def generate(self, text, random, beam, max_length, repetition):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_ids = self.model.tokenizer(text, return_tensors="pt")['input_ids']
                len_input_tokens = len(input_ids[0])

                generate = self.model.generate(input_ids.cuda(),
                                                do_sample=random,
                                                num_beams=beam,
                                                temperature=1.,
                                                repetition_penalty=float(repetition),
                                                stopping_criteria=self.stopping_criteria,
                                                max_new_tokens=max_length,
                                                top_p=0.8,
                                                top_k=40,
                                                length_penalty=1.0)
        response = generate[0].tolist()
        response = response[len_input_tokens:]
        response = self.model.tokenizer.decode(response, skip_special_tokens=True)
        response = response.replace('[UNUSED_TOKEN_145]', '')
        response = response.replace('[UNUSED_TOKEN_146]', '')
        return response

    def generate_with_emb(self, emb, random, beam, max_length, repetition, im_mask=None):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                generate = self.model.generate(inputs_embeds=emb,
                                                do_sample=random,
                                                num_beams=beam,
                                                temperature=1.,
                                                repetition_penalty=float(repetition),
                                                stopping_criteria=self.stopping_criteria,
                                                max_new_tokens=max_length,
                                                top_p=0.8,
                                                top_k=40,
                                                length_penalty=1.0,
                                                im_mask=im_mask)
        response = generate[0].tolist()
        response = self.model.tokenizer.decode(response, skip_special_tokens=True)
        response = response.replace('[UNUSED_TOKEN_145]', '')
        response = response.replace('[UNUSED_TOKEN_146]', '')
        return response


    def generate_article(self, question, beam, repetition, max_length, random, seed, outline=None):
        if outline is None:
            instruction = f'请列出下面的作文题目对应的写作大纲。\n**题目：**\n{question}'
            top_p = 0.8
        else:
            instruction = f'根据下面的题目和写作大纲进行写作。\n**题目**\n{question}\n\n**大纲：**\n{outline}\n\n**正文写作：**\n'
            top_p = 1.0

        self.reset()
        set_random_seed(int(seed))
        self.hash_folder = hashlib.sha256(instruction.encode()).hexdigest()
        self.instruction = instruction

        text = self.text2instruction(instruction)
        print('random generate:{}'.format(random))
        if article_stream_output:
            input_ids = self.model.tokenizer(text, return_tensors="pt")['input_ids']
            input_embeds = self.model.model.tok_embeddings(input_ids.cuda())
            im_mask = None

            print(text)
            generate_params = dict(
                inputs_embeds=input_embeds,
                do_sample=random,
                stopping_criteria=self.stopping_criteria,
                repetition_penalty=float(repetition),
                max_new_tokens=max_length,
                top_p=top_p,
                top_k=40,
                length_penalty=1.0,
                im_mask=im_mask,
            )
            output_text = "▌"
            with self.generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    decoded_output = self.model.tokenizer.decode(output[1:])
                    if output[-1] in [self.model.tokenizer.eos_token_id, 92542]:
                        break
                    output_text = decoded_output.replace('\n', '\n\n') + "▌"
                    yield output_text
                    time.sleep(0.01)
            output_text = output_text[:-1]
            yield output_text
        else:
            output_text = self.generate(text, random, beam, max_length, repetition)
            return output_text


    def save(self, beam, repetition, text_num, random, seed):
        folder = 'save_articles/' + self.hash_folder
        if os.path.exists(folder):
            for item in os.listdir(folder):
                os.remove(os.path.join(folder, item))
        os.makedirs(folder, exist_ok=True)

        save_text = '\n'.join([self.instruction, str(beam), str(repetition), str(text_num), str(random), str(seed)]) + '\n\n'
        if len(self.texts_imgs) > 0:
            for txt, img in self.texts_imgs:
                save_text += txt + '\n'
                if img is not None:
                    save_text += f'<div align="center"> <img src={os.path.basename(img.paths[img.pts])} width = 500/> </div>'
                    path = os.path.join('articles', self.hash_folder, os.path.basename(img.paths[img.pts]))
                    if os.path.exists(path):
                        shutil.copy(path, folder)
                    else:
                        shutil.copy(img.paths[img.pts], folder)

        with open(os.path.join(folder, 'io.MD'), 'w') as f:
            f.writelines(save_text)

        archived = shutil.make_archive(folder, 'zip', folder)
        return archived

    def generate_with_callback(self, callback=None, **kwargs):
        kwargs.setdefault("stopping_criteria",
                          transformers.StoppingCriteriaList())
        kwargs["stopping_criteria"].append(Stream(callback_func=callback))
        with torch.no_grad():
            self.model.generate(**kwargs)

    def generate_with_streaming(self, **kwargs):
        return Iteratorize(self.generate_with_callback, kwargs, callback=None)

    def change_meta(self, withmeta):
        self.withmeta = withmeta


parser = argparse.ArgumentParser()
#parser.add_argument("--code_path", default='internlm/internlm-xcomposer2-7b')
parser.add_argument("--code_path", default='/mnt/hwfile/mllm/zangyuhang/share_models/7b_v4')
parser.add_argument("--private", default=False, action='store_true')
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--port", default=11111, type=int)
args = parser.parse_args()
demo_ui = Demo_UI(args.code_path, args.num_gpus)


with gr.Blocks(css=custom_css, title='浦语·灵笔 (InternLM-XComposer)') as demo:
    with gr.Row():
        with gr.Column(scale=20):
            # gr.HTML("""<h1 align="center" id="space-title" style="font-size:35px;">🤗 浦语·灵笔 (InternLM-XComposer)</h1>""")
            gr.HTML(
                """<h1 align="center"><img src="https://raw.githubusercontent.com/InternLM/InternLM-XComposer/main/assets/logo-zuowen.png", alt="InternLM-XComposer" border="0" style="margin: 0 auto; height: 120px;" /></a> </h1>"""
            )
            gr.HTML(
                """<p>Internlm-xcomposer2-作文 是基于书生·浦语2和书生·浦语·灵笔2研发的作文模型。它采用先写作文大纲，再根据大纲展开的写作方式，能够清晰地梳理文章脉络，使文章结构更严谨。欢迎大家试用并提出宝贵意见，我们将持续优化，为广大用户带来更好的作文体验。</p>"""
            )

    with gr.Row():
        with gr.Column(scale=2):
            instruction = gr.Textbox(label='作文题目',
                                     lines=5,
                                     value='''阅读下面的材料，根据要求写作。
随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？
以上材料引发了你怎样的联想和思考？请写一篇文章。
要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。''', elem_classes='edit')
        with gr.Column(scale=1):
            seed = gr.Slider(minimum=1.0, maximum=20000.0, value=8909.0, step=1.0, label='Random Seed (随机种子)')
            btn = gr.Button("Submit (提交)", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Advanced Settings (高级设置)", open=False, visible=True) as parameter_article:
                beam = gr.Slider(minimum=1.0, maximum=6.0, value=1.0, step=1.0, label='Beam Size (集束大小)')
                repetition = gr.Slider(minimum=1.0, maximum=2.0, value=1.005, step=0.001, label='Repetition_penalty (重复惩罚)')
                text_num = gr.Slider(minimum=100.0, maximum=4096.0, value=4096.0, step=1.0, label='Max output tokens (最多输出字数)')
                random = gr.Checkbox(value=True, label='Sampling (随机采样)')
                withmeta = gr.Checkbox(value=False, label='With Meta (使用meta指令)')

    province = gr.Markdown(visible=False)

    gr.Examples(
        examples=[['2024 新课标I卷', '''阅读下面的材料，根据要求写作。
随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？
以上材料引发了你怎样的联想和思考？请写一篇文章。
要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。'''],
                  ['2024 新课标II卷', '''阅读下面的材料，根据要求写作。
本试卷现代文阅读I提到，长久以来，人们只能看到月球固定朝向地球的一面，“嫦娥四号”探月任务揭开了月背的神秘面纱；随着“天问一号”飞离地球，航天人的目光又投向遥远的深空……
正如人类的太空之旅，我们每个人也都在不断抵达未知之境。
这引发了你怎样的联想与思考？请写一篇文章。
要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。'''],
                  ['2024 北京卷', '''几千年来，古老的经典常读常新，杰出的思想常用常新，中华民族的伟大精神亘古常新……很多事物，在时间的淬炼中，愈显活力和价值。请以“历久弥新”为题目，写一篇议论文。要求：论点明确，论据充分，论证合理；语言流畅，书写清晰。'''],
                  ['2024 天津卷', '''阅读下面的材料，根据要求写作。
在缤纷的世界中，无论是个人、群体还是国家，都会面对别人对我们的定义。我们要认真对待“被定义”，明辨是非，去芜存真，为自己的提升助力；也要勇于通过“自定义”来塑造自我，彰显风华，用自己的方式前进。
以上材料能引发你怎样的联想与思考？请结合你的体验和感悟，写一篇文章。
要求：①自选角度，自拟标题；②文体不限（诗歌除外），文体特征明显；③不少于800字；④不得抄袭，不得套作。'''],
                  ['2024 上海卷', '''生活中，人们常用认可度判别事物，区分高下。请写一篇文章，谈谈你对“认可度”的认识和思考。要求：（1）自拟题目；（2）不少于800字。''']],
        inputs=[province, instruction],
    )

    outline = gr.Textbox(label='文章大纲', lines=3, max_lines=8)
    #article = gr.Textbox(label='作文', lines=10, max_lines=20, elem_classes='feedback')
    #gr.Markdown(value='文章创作:', elem_classes='feedback')
    article = gr.Markdown(label='作文', elem_classes='feedback')

    btn.click(demo_ui.generate_article,
        inputs=[instruction, beam, repetition, text_num, random, seed], outputs=outline).then(
        demo_ui.generate_article, inputs=[instruction, beam, repetition, text_num, random, seed, outline], outputs=article)



if __name__ == "__main__":
    if args.private:
        demo.queue().launch(share=False, server_name="127.0.0.1", server_port=args.port, max_threads=1)
    else:
        demo.queue().launch(share=True, server_name="0.0.0.0", server_port=args.port, max_threads=1)

