import os
import re
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import argparse
import gradio as gr
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), 'tmp')
import time
from datetime import datetime
import shutil
from PIL import Image, ImageFile
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

from demo_asset.assets.css_html_js import custom_css
from demo_asset.gradio_patch import Chatbot as grChatbot
from demo_asset.serve_utils import Stream, Iteratorize
from demo_asset.conversation import CONV_VISION_INTERN2
from examples.utils import auto_configure_device_map, get_stopping_criteria, set_random_seed


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
    top_padding = int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

    return b

def R560_HD18_Identity_transform(img):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width/ height)
    scale = 1
    while scale*np.ceil(scale/ratio) <= 18:
        scale += 1
    scale -= 1

    scale_low = min(np.ceil(width * 1.5 / 560), scale)
    scale_up = min(np.ceil(width * 1.5 / 560), scale)
    scale = random.randrange(scale_low, scale_up + 1)

    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(img, [new_h, new_w],)
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
    ratio = (width/ height)
    scale = 1
    while scale*np.ceil(scale/ratio) <= 4:
        scale += 1
    scale -= 1
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(img, [new_h, new_w],)
    img = padding_336(img, 560)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


class Demo_UI:
    def __init__(self, code_path, num_gpus=1):
        self.code_path = code_path

        tokenizer = AutoTokenizer.from_pretrained(code_path, trust_remote_code=True)
        self.chat_model = AutoModelForCausalLM.from_pretrained(code_path, device_map='cuda', trust_remote_code=True).half().eval()
        self.chat_model.tokenizer = tokenizer

        stop_words_ids = [92542]
        self.stopping_criteria = get_stopping_criteria(stop_words_ids)
        set_random_seed(1234)
        self.folder = None

    def get_context_emb(self, state, img_list):
        prompt = state.get_prompt()
        print(prompt)
        prompt_segs = prompt.split('<Img><ImageHere></Img>')

        assert len(prompt_segs) == len(
            img_list
        ) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.chat_model.tokenizer(seg, return_tensors="pt",  add_special_tokens=i == 0).input_ids.to(0)
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.chat_model.model.tok_embeddings(seg_t) for seg_t in seg_tokens]
        txt_mask = [torch.zeros(seg_e.shape[:2]) for seg_e in seg_embs]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        maxed_masks = [emb for pair in zip(txt_mask[:-1], [torch.ones(img.shape[:2]) for img in img_list]) for emb in pair] + [txt_mask[-1]]
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

    def chat_ask(self, state, img_list, text, image, task):
        print(1111)
        state.skip_next = False
        if len(text) <= 0 and image is None:
            state.skip_next = True
            return (state, img_list, state.to_gradio_chatbot(), "",
                    None) + (gr.Button(), ) * 2

        if image is not None:
            image = [image]
            imgs = []
            imgs_pil = []
            for j in range(len(image)):
                img_pil = Image.open(image[j]).convert('RGB')
                imgs_pil.append(img_pil)
                if task == "Single":
                    img_pil = R560_HD18_Identity_transform(img_pil)
                    state.single = True
                else:
                    img_pil = R560_HD4_transform(img_pil)
                    state.single = False
                img = self.chat_model.vis_processor(img_pil)
                imgs.append(img)
            imgs = torch.stack(imgs, dim=0)
            print(imgs.shape)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_emb = self.chat_model.encode_img(imgs)

            image_emb = torch.cat([t.unsqueeze(0) for t in image_emb], dim=1)
            print(image_emb.shape)
            img_list.append(image_emb)

            state.append_message(state.roles[0],
                                 ["<Img><ImageHere></Img>", imgs_pil])

        if len(state.messages) > 0 and state.messages[-1][0] == state.roles[
                0] and isinstance(state.messages[-1][1], list):
            #state.messages[-1][1] = ' '.join([state.messages[-1][1], text])
            if state.single:
                state.messages[-1][1][0] = ''.join(
                    [state.messages[-1][1][0], text])
            else:
                state.messages[-1][1][0] = ''.join(
                    [f"Image{len(img_list)}: ", state.messages[-1][1][0], "; ", text])
        else:
            state.append_message(state.roles[0], text)

        print(state.messages)

        state.append_message(state.roles[1], None)

        return (state, img_list, state.to_gradio_chatbot(), "",
                None) + (gr.Button(interactive=False), ) * 4

    def chat_answer(self, state, img_list, max_output_tokens,
                    repetition_penalty, num_beams, do_sample):
        state.system = f"[UNUSED_TOKEN_146]system\n{chat_meta}[UNUSED_TOKEN_145]\n"
        if state.skip_next:
            return (state, state.to_gradio_chatbot()) + (gr.Button(), ) * 2

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
                for output in generator:
                    decoded_output = self.chat_model.tokenizer.decode(
                        output[1:])
                    if output[-1] in [
                            self.chat_model.tokenizer.eos_token_id, 92542
                    ]:
                        break
                    state.messages[-1][-1] = decoded_output + "▌"
                    yield (state,
                           state.to_gradio_chatbot()) + (gr.Button(interactive=False), ) * 4
                    time.sleep(0.03)
            state.messages[-1][-1] = [state.messages[-1][-1][:-1], '']
            if self.folder and os.path.exists(self.folder):
                with open(os.path.join(self.folder, 'chat.txt'), 'a+') as fd:
                    if isinstance(state.messages[-2][1], str):
                        fd.write(state.messages[-2][0] + state.messages[-2][1])
                    else:
                        fd.write(state.messages[-2][0] + state.messages[-2][1][0])
                    fd.write(state.messages[-1][0] + ''.join(state.messages[-1][1]))

            yield (state, state.to_gradio_chatbot()) + (gr.Button(interactive=True), ) * 4
            return
        else:
            outputs = self.chat_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_output_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                #temperature=float(temperature),
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

            return (state, state.to_gradio_chatbot()) + (gr.Button(interactive=True), ) * 4

    def clear_answer(self, state):
        state.messages[-1][-1] = None
        return (state, state.to_gradio_chatbot())

    def chat_clear_history(self):
        state = CONV_VISION_INTERN2.copy()
        return (state, [], state.to_gradio_chatbot(), "", None) + (gr.Button(interactive=False), ) * 4

    def uploadimgs(self, images):
        timestamp = datetime.now()
        self.folder = os.path.join('databases', timestamp.strftime("%Y%m%d"), 'chat', str(timestamp).replace(' ', '-'))
        os.makedirs(self.folder, exist_ok=True)
        for image_path in images:
            shutil.copy(image_path, self.folder)

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
parser.add_argument("--code_path", default='DLight1551/JSH_0527')
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
                """<h1 align="center"><img src="https://raw.githubusercontent.com/InternLM/InternLM-XComposer/InternLM-XComposer2/assets/logo_en.png", alt="InternLM-XComposer" border="0" style="margin: 0 auto; height: 120px;" /></a> </h1>"""
            )
        with gr.Column(scale=1, min_width=100):
            lang_btn = gr.Button("中文")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("💬 Multimodal Chat (多模态对话)", elem_id="chat", id=0):
            chat_state = gr.State()
            img_list = gr.State()
            with gr.Row():
                with gr.Column(scale=3):
                    #imagebox = gr.File(file_count='multiple')
                    imagebox = gr.File(file_count='single')
                    task = gr.Dropdown(["Single", "Multiple"], value='Single', label="图片数", interactive=True)

                    with gr.Accordion("Parameters (参数)", open=True,
                                      visible=False) as parameter_row:
                        chat_max_output_tokens = gr.Slider(
                            minimum=0,
                            maximum=1024,
                            value=512,
                            step=64,
                            interactive=True,
                            label="Max output tokens (最多输出字数)",
                        )
                        chat_num_beams = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=1,
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
                                visible=False)
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
            parameter_list = [
                chat_max_output_tokens, chat_repetition_penalty,
                chat_num_beams, chat_do_sample
            ]

            imagebox.upload(demo_ui.uploadimgs, imagebox, [])
            btn_like.click(demo_ui.like, [chat_state], [chat_state, chatbot])
            btn_dislike.click(demo_ui.dislike, [chat_state], [chat_state, chatbot])

            chat_textbox.submit(
                demo_ui.chat_ask,
                [chat_state, img_list, chat_textbox, imagebox, task],
                [chat_state, img_list, chatbot, chat_textbox, imagebox] +
                btn_list).then(demo_ui.chat_answer,
                               [chat_state, img_list] + parameter_list,
                               [chat_state, chatbot] + btn_list)
            submit_btn.click(
                demo_ui.chat_ask,
                [chat_state, img_list, chat_textbox, imagebox, task],
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
                demo_ui.chat_clear_history, None,
                [chat_state, img_list, chatbot, chat_textbox, imagebox] +
                btn_list)

            demo.load(load_demo, None, [
                chat_state, img_list, chatbot, chat_textbox, submit_btn,
                parameter_row
            ])

    lang_btn.click(change_language, inputs=lang_btn, outputs=[lang_btn, chat_textbox, submit_btn, regenerate_btn, clear_btn])

if __name__ == "__main__":
    if args.private:
        demo.queue().launch(share=False, server_name="127.0.0.1", server_port=args.port, max_threads=1)
    else:
        demo.queue().launch(share=True, server_name="0.0.0.0", server_port=args.port, max_threads=1)

