import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import PIL
import re
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

import argparse
import gradio as gr

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), 'tmp')
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
from datetime import datetime
import shutil
from PIL import Image, ImageFile, ImageDraw, ImageFont
import torch
from urllib.request import urlopen
import time

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

from decord import VideoReader

from demo_asset.assets.css_html_js import custom_css
from demo_asset.gradio_patch import Chatbot as grChatbot
from demo_asset.serve_utils import Stream, Iteratorize
from demo_asset.conversation import CONV_VISION_INTERN2
from gradio_demo.utils import get_stopping_criteria, set_random_seed


meta_instruction = """You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
chat_meta = """You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.
"""

chat_stream_output = True

import random
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms


def padding_336(b, R=336):
    width, height = b.size
    tar = int(np.ceil(height / R) * R)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255])

    return b


def R560_HD18_Identity_transform(img):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= 18:
        scale += 1
    scale -= 1

    scale_low = min(np.ceil(width * 1.5 / 560), scale)
    scale_up = min(np.ceil(width * 1.5 / 560), scale)
    scale = random.randrange(scale_low, scale_up + 1)

    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(img, [new_h, new_w], )
    img = padding_336(img, 560)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


def R560_HD4_transform(img):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= 4:
        scale += 1
    scale -= 1
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(img, [new_h, new_w], )
    img = padding_336(img, 560)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


class Demo_UI:
    def __init__(self, code_path, num_gpus=1):
        self.code_path = code_path

        tokenizer = AutoTokenizer.from_pretrained(code_path, trust_remote_code=True)
        self.chat_model = AutoModelForCausalLM.from_pretrained(code_path, device_map='cuda',
                                                               trust_remote_code=True).half().eval()
        self.chat_model.tokenizer = tokenizer

        stop_words_ids = [92542]
        self.stopping_criteria = get_stopping_criteria(stop_words_ids)
        set_random_seed(1234)
        self.folder = None

    def get_context_emb(self, state, img_list):
        prompt = state.get_prompt()
        prompt_segs = prompt.split('<Img><ImageHere></Img>')

        assert len(prompt_segs) == len(
            img_list
        ) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.chat_model.tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0).input_ids.to(0)
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.chat_model.model.tok_embeddings(seg_t) for seg_t in seg_tokens]
        txt_mask = [torch.zeros(seg_e.shape[:2]) for seg_e in seg_embs]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        maxed_masks = [emb for pair in zip(txt_mask[:-1], [torch.ones(img.shape[:2]) for img in img_list]) for emb in
                       pair] + [txt_mask[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        maxed_masks = torch.cat(maxed_masks, dim=1).bool()
        return mixed_embs, maxed_masks

    def generate_with_chat_callback(self, callback=None, **kwargs):
        kwargs.setdefault("stopping_criteria",
                          transformers.StoppingCriteriaList())
        kwargs["stopping_criteria"].append(Stream(callback_func=callback))
        with torch.no_grad():
            self.chat_model.generate(**kwargs)

    def generate_with_chat_streaming(self, **kwargs):
        return Iteratorize(self.generate_with_chat_callback, kwargs, callback=None)

    def video_img_process(self, imgs):
        new_imgs = []
        for img in imgs:
            w, h = img.size
            scale = w / h
            if w > h:
                new_w = 560 * 2
                new_h = int(560 * 2 / scale)
            else:
                new_w = int(560 * 2 * scale)
                new_h = 560 * 2
            img = transforms.functional.resize(img, [new_h, new_w], )
            new_imgs.append(img)
        imgs = new_imgs
        new_w = 0
        new_h = 0
        pad = 40
        if w > h:
            for im in imgs:
                w, h = im.size
                new_w = max(new_w, w)
                new_h += h + 10 + pad
            truetype_url = 'https://huggingface.co/internlm/internlm-xcomposer2d5-7b/resolve/main/SimHei.ttf?download=true'
            ff = urlopen(truetype_url)
            font = ImageFont.truetype(ff, pad)
            new_img = Image.new('RGB', (new_w, new_h), 'white')
            draw = ImageDraw.Draw(new_img)
            curr_h = 0
            for idx, im in enumerate(imgs):
                w, h = im.size
                new_img.paste(im, (0, pad + curr_h))
                draw.text((0, curr_h), f'<IMAGE {idx}>', font=font, fill='black')
                if idx + 1 < len(imgs):
                    draw.line([(0, pad + curr_h + h + 5), (new_w, pad + curr_h + h + 5)], fill='black', width=2)
                curr_h += h + 10 + pad
            # print (new_w, new_h)
        else:
            for im in imgs:
                w, h = im.size
                new_w += w + 10
                new_h = max(new_h, h)
            new_h += pad
            font = ImageFont.truetype("SimHei.ttf", pad)
            new_img = Image.new('RGB', (new_w, new_h), 'white')
            draw = ImageDraw.Draw(new_img)
            curr_w = 0
            for idx, im in enumerate(imgs):
                w, h = im.size
                new_img.paste(im, (curr_w, pad))
                draw.text((curr_w, 0), f'<IMAGE {idx}>', font=font, fill='black')
                if idx + 1 < len(imgs):
                    draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill='black', width=2)
                curr_w += w + 10
        return new_img

    def load_video(self, vis_path, num_frm=32, start=None, end=None):
        vid = VideoReader(vis_path, num_threads=1)
        fps = vid.get_avg_fps()
        t_stride = int(2 * round(float(fps) / int(1)))
        start_idx = 0 if start is None else start
        end_idx = len(vid) if end is None else end
        all_pos = list(range(start_idx, end_idx, t_stride))
        images = [vid[i].asnumpy() for i in all_pos]
        if len(images) > num_frm:
            num_frm = min(num_frm, len(images))
            step_size = len(images) / (num_frm + 1)
            indices = [int(i * step_size) for i in range(num_frm)]
            images = [images[i] for i in indices]
        images = [Image.fromarray(arr) for arr in images]
        print(f'sample {len(images)} frames.')
        img = self.video_img_process(images)
        return img

    def chat_ask(self, state, img_list, text, image, video, task):
        if task == 'Single Image':
            grfile = gr.File(None, visible=False)
        elif task == 'Multiple Images':
            grfile = None
        else:
            grfile = gr.Video()

        print(f'Input: {text}', flush=True)

        state.skip_next = False
        if len(text) <= 0 and image is None:
            state.skip_next = True
            return (state, img_list, state.to_gradio_chatbot(), "",
                    None) + (gr.Button(),) * 4

        if task == 'Single Video' and video is not None and len(img_list) == 0:
            img_pil = self.load_video(video)
            img_pil = R560_HD18_Identity_transform(img_pil)
            state.single = True
            img_str = "<Img><ImageHere></Img>"

            img = self.chat_model.vis_processor(img_pil)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_emb = self.chat_model.encode_img(img.unsqueeze(0))

            img_list.append(image_emb)
            state.append_message(state.roles[0], [img_str, []])

        if image is not None:
            imgs_pil = []
            img_str = ""

            for j in range(len(image)):
                img_pil = Image.open(image[j]).convert('RGB')
                imgs_pil.append(img_pil)

                if "Single" in task:
                    img_pil = R560_HD18_Identity_transform(img_pil)
                    state.single = True
                    img_str = "<Img><ImageHere></Img>"
                else:
                    img_pil = R560_HD4_transform(img_pil)
                    state.single = False
                img = self.chat_model.vis_processor(img_pil)

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        image_emb = self.chat_model.encode_img(img.unsqueeze(0))

                img_list.append(image_emb)

                if not state.single:
                    img_str += f"Image{len(img_list)}: <Img><ImageHere></Img>; "

            state.append_message(state.roles[0],
                                 [img_str, imgs_pil])

        if len(state.messages) > 0 and state.messages[-1][0] == state.roles[
            0] and isinstance(state.messages[-1][1], list):
            # state.messages[-1][1] = ' '.join([state.messages[-1][1], text])
            state.messages[-1][1][0] = ''.join(
                [state.messages[-1][1][0], text])
        else:
            state.append_message(state.roles[0], text)

        state.append_message(state.roles[1], None)

        return (state, img_list, state.to_gradio_chatbot(), "",
                grfile) + (gr.Button(interactive=False),) * 4

    def chat_answer(self, state, img_list, max_output_tokens,
                    repetition_penalty, num_beams, do_sample):
        state.system = f"[UNUSED_TOKEN_146]system\n{chat_meta}[UNUSED_TOKEN_145]\n"
        if state.skip_next:
            return (state, state.to_gradio_chatbot()) + (gr.Button(),) * 4

        embs, im_mask = self.get_context_emb(state, img_list)
        if chat_stream_output:
            generate_params = dict(
                inputs_embeds=embs,
                num_beams=num_beams,
                do_sample=do_sample,
                stopping_criteria=self.stopping_criteria,
                repetition_penalty=float(repetition_penalty),
                max_length=max_output_tokens,
                bos_token_id=self.chat_model.tokenizer.bos_token_id,
                eos_token_id=self.chat_model.tokenizer.eos_token_id,
                pad_token_id=self.chat_model.tokenizer.pad_token_id,
                im_mask=im_mask,
            )
            state.messages[-1][-1] = "▌"

            with self.generate_with_chat_streaming(**generate_params) as generator:
                for i, output in enumerate(generator):
                    decoded_output = self.chat_model.tokenizer.decode(
                        output[1:])
                    if output[-1] in [
                        self.chat_model.tokenizer.eos_token_id, 92542
                    ]:
                        break
                    state.messages[-1][-1] = decoded_output + "▌"

                    yield (state,
                           state.to_gradio_chatbot()) + (gr.Button(interactive=False),) * 4
                    time.sleep(0.03)

            state.messages[-1][-1] = [state.messages[-1][-1][:-1], '']
            if self.folder and os.path.exists(self.folder):
                with open(os.path.join(self.folder, 'chat.txt'), 'a+') as fd:
                    if isinstance(state.messages[-2][1], str):
                        fd.write(state.messages[-2][0] + state.messages[-2][1])
                    else:
                        fd.write(state.messages[-2][0] + state.messages[-2][1][0])
                    fd.write(state.messages[-1][0] + ''.join(state.messages[-1][1]))

            yield (state, state.to_gradio_chatbot()) + (gr.Button(interactive=True),) * 4
            return
        else:
            outputs = self.chat_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_output_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                # temperature=float(temperature),
                do_sample=do_sample,
                repetition_penalty=float(repetition_penalty),
                bos_token_id=self.chat_model.tokenizer.bos_token_id,
                eos_token_id=self.chat_model.tokenizer.eos_token_id,
                pad_token_id=self.chat_model.tokenizer.pad_token_id,
                im_mask=im_mask,
            )

            output_token = outputs[0]
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_text = self.chat_model.tokenizer.decode(output_token, add_special_tokens=False)
            print(output_text)
            output_text = output_text.split('<TOKENS_UNUSED_1>')[
                0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_text = output_text.replace("<s>", "")
            state.messages[-1][1] = output_text

            return (state, state.to_gradio_chatbot()) + (gr.Button(interactive=True),) * 4

    def clear_answer(self, state):
        state.messages[-1][-1] = None
        return (state, state.to_gradio_chatbot())

    def chat_clear_history(self, task):
        if 'Image' in task:
            imagebox = gr.File(None, visible=True)
            videobox = gr.Video(visible=False)
        else:
            imagebox = gr.File(None, visible=False)
            videobox = gr.Video(visible=True)

        state = CONV_VISION_INTERN2.copy()
        return (state, [], state.to_gradio_chatbot(), "", imagebox, videobox) + (gr.Button(interactive=False),) * 4

    def clean_chat_ask(self, image, video, text, task, beam, chat_do_sample):
        state = CONV_VISION_INTERN2.copy()
        img_list = []

        return (state, img_list, state.to_gradio_chatbot(), text,
                image, video, beam, chat_do_sample, task)

    def task_select(self, task, videobox):
        if 'Image' in task:
            return gr.File(visible=True), gr.Video(visible=False), gr.Textbox(interactive=True), gr.Button(
                interactive=True)
        else:
            if videobox:
                return gr.File(visible=False), gr.Video(visible=True), gr.Textbox(interactive=True), gr.Button(
                    interactive=True)
            else:
                return gr.File(visible=False), gr.Video(visible=True), gr.Textbox(interactive=False), gr.Button(
                    interactive=False)

    def uploadimgs(self, task, images):
        if 'Single' in task and len(images) > 1:
            gr.Warning("Single mode do not support multiple images!!!")
            return []
        timestamp = datetime.now()
        self.folder = os.path.join('databases', timestamp.strftime("%Y%m%d"), 'chat', str(timestamp).replace(' ', '-'))
        os.makedirs(self.folder, exist_ok=True)
        for image_path in images:
            shutil.copy(image_path, self.folder)
        return images

    def uploadvideo(self):
        return gr.Textbox(interactive=True), gr.Button(interactive=True)

    def clearvideo(self):
        return gr.Textbox(interactive=False), gr.Button(interactive=False)

    def like(self, state):
        if self.folder and os.path.exists(self.folder):
            with open(os.path.join(self.folder, 'chat.txt'), 'r') as fd:
                content = fd.read()

            if content[-1] == '👎':
                content = content[:-1]
            if content[-1] != '👍':
                content = content + '👍'

            state.messages[-1][-1][1] = '👍'

            with open(os.path.join(self.folder, 'chat.txt'), 'w') as fd:
                fd.write(content)

        return state, state.to_gradio_chatbot()

    def dislike(self, state):
        if self.folder and os.path.exists(self.folder):
            with open(os.path.join(self.folder, 'chat.txt'), 'r') as fd:
                content = fd.read()

            if content[-1] == '👍':
                content = content[:-1]
            if content[-1] != '👎':
                content = content + '👎'

            state.messages[-1][-1][1] = '👎'

            with open(os.path.join(self.folder, 'chat.txt'), 'w') as fd:
                fd.write(content)

        return state, state.to_gradio_chatbot()


def load_demo():
    state = CONV_VISION_INTERN2.copy()

    return (state, [], gr.Chatbot(visible=True),
            gr.Textbox(visible=True), gr.Button(visible=True),
            gr.Row(visible=True), gr.Accordion(visible=True))


def change_language(lang):
    if lang == '中文':
        lang_btn = gr.Button(value='English')
        chat_textbox = gr.update(placeholder='输入聊天内容并回车')
        submit_btn = gr.update(value='提交')
        regenerate_btn = gr.update(value='🔄  重新生成')
        clear_btn = gr.update(value='🗑️  清空聊天框')
    elif lang == 'English':
        lang_btn = gr.Button(value='中文')
        chat_textbox = gr.update(placeholder='Enter text and press ENTER')
        submit_btn = gr.update(value='Submit')
        regenerate_btn = gr.update(value='🔄  Regenerate')
        clear_btn = gr.update(value='🗑️  Clear history')

    return [lang_btn, chat_textbox, submit_btn, regenerate_btn, clear_btn]


parser = argparse.ArgumentParser()
parser.add_argument("--code_path", default='internlm/internlm-xcomposer2d5-7b')
parser.add_argument("--private", default=False, action='store_true')
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--port", default=7860, type=int)
args = parser.parse_args()
demo_ui = Demo_UI(args.code_path, args.num_gpus)

with gr.Blocks(css=custom_css, title='浦语·灵笔 (InternLM-XComposer)') as demo:
    with gr.Row():
        with gr.Column(scale=20):
            # gr.HTML("""<h1 align="center" id="space-title" style="font-size:35px;">🤗 浦语·灵笔 (InternLM-XComposer)</h1>""")
            gr.HTML(
                """<h1 align="center"><img src="https://huggingface.co/DLight1551/JSH0626/resolve/main/teaser.png", alt="InternLM-XComposer" border="0" style="margin: 0 auto; height: 120px;" /></a> </h1>"""
            )
        with gr.Column(scale=1, min_width=100):
            lang_btn = gr.Button("中文")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("💬 Multimodal Chat (多模态对话)", elem_id="chat", id=0):
            chat_state = gr.State()
            img_list = gr.State()
            with gr.Row():
                with gr.Column(scale=3):
                    # task = gr.Dropdown(["Single Image", "Multiple Images", "Single Video"], value='Single Image', label="单图/多图/单视频模式", interactive=True)
                    task = gr.Radio(["Single Image", "Multiple Images", "Single Video"], value='Single Image',
                                    label="单图/多图/单视频模式", interactive=True)
                    imagebox = gr.File(file_count='multiple', file_types=['image'])
                    videobox = gr.Video(label='video', sources=["upload", "webcam"], format='mp4', visible=False,
                                        autoplay=True)

                    with gr.Accordion("Parameters (参数)", open=True,
                                      visible=False) as parameter_row:
                        chat_max_output_tokens = gr.Slider(
                            minimum=0,
                            maximum=4096,
                            value=1024,
                            step=64,
                            interactive=True,
                            label="Max output tokens (最多输出字数)",
                        )
                        chat_num_beams = gr.Slider(
                            minimum=1,
                            maximum=3,
                            value=2,
                            step=1,
                            interactive=True,
                            label="Beam Size (集束大小)",
                        )
                        chat_repetition_penalty = gr.Slider(
                            minimum=1,
                            maximum=2,
                            value=1.005,
                            step=0.001,
                            interactive=True,
                            label="Repetition_penalty (重复惩罚)",
                        )
                        # chat_temperature = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, interactive=True,
                        #                         label="Temperature", )
                        chat_do_sample = gr.Checkbox(interactive=True,
                                                     value=True,
                                                     label="Do_sample (采样)")

                with gr.Column(scale=6):
                    chatbot = grChatbot(elem_id="chatbot",
                                        visible=False,
                                        height=750)
                    with gr.Row():
                        with gr.Column(scale=8):
                            chat_textbox = gr.Textbox(
                                show_label=False,
                                placeholder="Enter text and press ENTER (输入聊天内容并回车)",
                                visible=True)
                        with gr.Column(scale=1, min_width=60):
                            submit_btn = gr.Button(value="Submit",
                                                   visible=False)
                    with gr.Row(visible=True) as button_row:
                        regenerate_btn = gr.Button(value="🔄  Regenerate",
                                                   interactive=False)
                        clear_btn = gr.Button(value="🗑️  Clear history",
                                              interactive=False)
                        btn_like = gr.Button(interactive=False, value='👍  Like (点赞)')
                        btn_dislike = gr.Button(interactive=False, value='👎  Dislike (点踩)')

            btn_list = [regenerate_btn, clear_btn, btn_like, btn_dislike]

            gr.Examples(
                examples=[[['demo_asset/MD1.png'], None, 'translate this image into markdown format', 'Single Image', 3, False],
                          [['demo_asset/MD2.jpg'], None, 'translate this image into markdown format', 'Single Image', 3, False],
                          [['demo_asset/4K_1.png'], None, 'analyze this image in detail', 'Single Image', 3, False],
                          [['demo_asset/4K_2.png'], None, 'analyze this chart panel by panel', 'Single Image', 3, False],
                          [['demo_asset/4K_3.png'], None, 'analyze this flowchart step by step', 'Single Image', 3, False],
                          [['demo_asset/MMDU1_0.jpg', 'demo_asset/MMDU1_1.jpg', 'demo_asset/MMDU1_2.jpg'], None, 'I want to buy a car from the three given cars, please give me some advice', 'Multiple Images', 1, False],
                          ],
                inputs=[imagebox, videobox, chat_textbox, task, chat_num_beams, chat_do_sample],
                outputs=[chat_state, img_list, chatbot, chat_textbox, imagebox, videobox, chat_num_beams,
                         chat_do_sample, task],
                fn=demo_ui.clean_chat_ask,
                run_on_click=True,
            )

            gr.HTML("""<h1 align="center"></h1>""")
            gr.HTML("""<h1 align="center"></h1>""")

            parameter_list = [
                chat_max_output_tokens, chat_repetition_penalty,
                chat_num_beams, chat_do_sample
            ]

            task.select(demo_ui.clean_chat_ask, [imagebox, videobox, chat_textbox, task, chat_num_beams, chat_do_sample],
                        [chat_state, img_list, chatbot, chat_textbox, imagebox, videobox, chat_num_beams, chat_do_sample,
                         task]).then(
                demo_ui.task_select, [task, videobox], [imagebox, videobox, chat_textbox, submit_btn])
            imagebox.upload(demo_ui.uploadimgs, [task, imagebox], [imagebox])
            videobox.upload(demo_ui.uploadvideo, None, [chat_textbox, submit_btn])
            videobox.stop_recording(demo_ui.uploadvideo, None, [chat_textbox, submit_btn])
            videobox.clear(demo_ui.clean_chat_ask, [imagebox, videobox, chat_textbox, task, chat_num_beams, chat_do_sample],
                           [chat_state, img_list, chatbot, chat_textbox, imagebox, videobox, chat_num_beams, chat_do_sample,
                            task]).then(
                demo_ui.clearvideo, None, [chat_textbox, submit_btn])

            btn_like.click(demo_ui.like, [chat_state], [chat_state, chatbot])
            btn_dislike.click(demo_ui.dislike, [chat_state], [chat_state, chatbot])

            chat_textbox.submit(
                demo_ui.chat_ask,
                [chat_state, img_list, chat_textbox, imagebox, videobox, task],
                [chat_state, img_list, chatbot, chat_textbox, imagebox] +
                btn_list).then(demo_ui.chat_answer,
                               [chat_state, img_list] + parameter_list,
                               [chat_state, chatbot] + btn_list)
            submit_btn.click(
                demo_ui.chat_ask,
                [chat_state, img_list, chat_textbox, imagebox, videobox, task],
                [chat_state, img_list, chatbot, chat_textbox, imagebox] +
                btn_list).then(demo_ui.chat_answer,
                               [chat_state, img_list] + parameter_list,
                               [chat_state, chatbot] + btn_list)

            regenerate_btn.click(demo_ui.clear_answer, chat_state,
                                 [chat_state, chatbot]).then(
                demo_ui.chat_answer,
                [chat_state, img_list] + parameter_list,
                [chat_state, chatbot] + btn_list)
            clear_btn.click(
                demo_ui.chat_clear_history, [task],
                [chat_state, img_list, chatbot, chat_textbox, imagebox, videobox] +
                btn_list)

            demo.load(load_demo, None, [
                chat_state, img_list, chatbot, chat_textbox, submit_btn,
                parameter_row
            ])

    lang_btn.click(change_language, inputs=lang_btn,
                   outputs=[lang_btn, chat_textbox, submit_btn, regenerate_btn, clear_btn])

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=args.port, max_threads=1)