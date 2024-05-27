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
from transformers import StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

from demo_asset.assets.css_html_js import custom_css
from demo_asset.gradio_patch import Chatbot as grChatbot
from demo_asset.serve_utils import Stream, Iteratorize
from demo_asset.conversation import CONV_VISION_INTERN2, StoppingCriteriaSub
from demo_asset.download import download_image_thread
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


class Database(object):
    def __init__(self):
        self.title = '###'
        self.hash_title = hashlib.sha256(self.title.encode()).hexdigest()

    def addtitle(self, title, hash_folder, params):
        self.title = title
        self.hash_folder = hash_folder
        time = datetime.now()
        self.folder = os.path.join('databases', time.strftime("%Y%m%d"), 'composition', self.hash_folder)
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)

        os.makedirs(self.folder)
        with open(os.path.join(self.folder, 'index.txt'), 'w') as fd:
            fd.write(self.title + '\n')
            fd.write(self.hash_title + '\n')
            fd.write(str(time) + '\n')
            for key, val in params.items():
                fd.write(f"{key}:{val}" + '\n')
            fd.write('\n')

    def prepare_save_article(self, text_imgs, src_folder, tgt_folder):
        save_text = ''
        for txt, img in text_imgs:
            save_text += txt + '\n'
            if img is not None:
                save_text += f'<div align="center"> <img src={os.path.basename(img.paths[img.pts])} width = 500/> </div>'
                path = os.path.join(src_folder, os.path.basename(img.paths[img.pts]))
                dst_path = os.path.join(tgt_folder, os.path.basename(img.paths[img.pts]))
                if not os.path.exists(dst_path):
                    if os.path.exists(path):
                        shutil.copy(path, tgt_folder)
                    else:
                        shutil.copy(img.paths[img.pts], tgt_folder)
        return save_text

    def addarticle(self, text_imgs):
        if len(text_imgs) > 0:
            images_folder = os.path.join(self.folder, 'images')
            os.makedirs(images_folder, exist_ok=True)

        save_text = self.prepare_save_article(text_imgs, os.path.join('articles', self.hash_folder), images_folder)

        with open(os.path.join(self.folder, 'generate.MD'), 'w') as f:
            f.writelines(save_text)

    def addedit(self, edit_type, inst_edit, text_imgs):
        timestamp = datetime.now()
        with open(os.path.join(self.folder, 'index.txt'), 'a+') as f:
            f.write(str(edit_type) + '\n')
            f.write(str(inst_edit) + '\n')
            f.write(str(timestamp) + '\n\n')

        save_text = self.prepare_save_article(text_imgs, os.path.join('articles', self.hash_folder), os.path.join(self.folder, 'images'))
        with open(os.path.join(self.folder, str(timestamp).replace(' ', '-') + '.MD'), 'w') as f:
            f.writelines(save_text)


class Demo_UI:
    def __init__(self, code_path, chat_path='', num_gpus=1):
        self.code_path = code_path
        self.reset()

        tokenizer = AutoTokenizer.from_pretrained(code_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(code_path, device_map='cuda', trust_remote_code=True).half().eval()
        self.model.tokenizer = tokenizer
        self.model.vit.resize_pos()

        self.chat = False
        if len(chat_path) > 0:
            self.chat = True
            tokenizer = AutoTokenizer.from_pretrained(chat_path, trust_remote_code=True)
            self.chat_model = AutoModelForCausalLM.from_pretrained(chat_path, device_map='cuda', trust_remote_code=True).half().eval()
            self.chat_model.tokenizer = tokenizer

        self.vis_processor = ImageProcessor()

        stop_words_ids = [92397]
        self.stopping_criteria = get_stopping_criteria(stop_words_ids)
        set_random_seed(1234)
        self.r2 = re.compile(r'<Seg[0-9]*>')
        self.withmeta = False
        self.database = Database()
        self.chat_folder = None

    def reset(self):
        self.pt = 0
        self.img_pt = 0
        self.texts_imgs = []
        self.open_edit = False
        self.hash_folder = '12345'
        self.instruction = ''

    def reset_components(self):
        return (gr.Markdown(visible=True, value=''),) + (gr.Markdown(visible=False, value=''),) * (max_section - 1) + (
                gr.Button(visible=False),) * max_section + (gr.Image(visible=False),) * max_section + (gr.Accordion(visible=False),) * max_section * 2

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

    def extract_imgfeat(self, img_paths):
        if len(img_paths) == 0:
            return None
        images = []
        for j in range(len(img_paths)):
            image = self.vis_processor(img_paths[j])
            images.append(image)
        images = torch.stack(images, dim=0)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                img_embeds = self.model.encode_img(images)
        return img_embeds

    def generate_loc(self, text_sections, upimages, image_num):
        full_txt = ''.join(text_sections)
        input_text = ('<image> ' * len(upimages)).strip() + f'给定文章"{full_txt}" 根据上述文章，选择适合插入图像的{image_num}行'
        instruction = self.text2instruction(input_text) + '适合插入图像的行是'
        print(instruction)

        if len(upimages) > 0:
            img_embeds = self.extract_imgfeat(upimages)
            input_embeds, im_mask, _ = self.interleav_wrap(instruction, img_embeds)
            output_text = self.generate_with_emb(input_embeds, True, 1, 200, 1.005, im_mask=im_mask)
        else:
            output_text = self.generate(instruction, True, 1, 200, 1.005)

        inject_text = '适合插入图像的行是' + output_text
        print(inject_text)

        locs = [int(m[4:-1]) for m in self.r2.findall(inject_text)]
        print(locs)
        return inject_text, locs

    def generate_cap(self, text_sections, pos, progress):
        pasts = ''
        caps = {}
        for idx, po in progress.tqdm(enumerate(pos), desc="image captioning"):
            full_txt = ''.join(text_sections[:po + 2])
            if idx > 0:
                past = pasts[:-2] + '。'
            else:
                past = pasts

            #input_text = f' <|User|>: 给定文章"{full_txt}" {past}给出适合在<Seg{po}>后插入的图像对应的标题。' + ' \n<TOKENS_UNUSED_0> <|Bot|>: 标题是"'
            input_text = f'给定文章"{full_txt}" {past}给出适合在<Seg{po}>后插入的图像对应的标题。'
            instruction = self.text2instruction(input_text) + '标题是"'

            cap_text = self.generate(instruction, True, 1, 200, 1.005)
            cap_text = cap_text.split('"')[0].strip()
            print(cap_text)
            caps[po] = cap_text

            if idx == 0:
                pasts = f'现在<Seg{po}>后插入图像对应的标题是"{cap_text}"， '
            else:
                pasts += f'<Seg{po}>后插入图像对应的标题是"{cap_text}"， '

        print(caps)
        return caps

    def interleav_wrap(self, text, image, max_length=4096):
        device = image.device
        im_len = image.shape[1]
        image_nums = len(image)
        parts = text.split('<image>')
        wrap_embeds, wrap_im_mask = [], []
        temp_len = 0
        need_bos = True

        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = self.model.tokenizer(part,
                                                    return_tensors='pt',
                                                    padding='longest',
                                                    add_special_tokens=need_bos).to(device)
                if need_bos:
                    need_bos = False
                part_embeds = self.model.model.tok_embeddings(part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]))
                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                wrap_embeds.append(image[idx].unsqueeze(0))
                wrap_im_mask.append(torch.ones(1, image[idx].shape[0]))
                temp_len += im_len

            if temp_len > max_length:
                break

        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)
        wrap_embeds = wrap_embeds[:, :max_length].to(device)
        wrap_im_mask = wrap_im_mask[:, :max_length].to(device).bool()
        return wrap_embeds, wrap_im_mask, temp_len

    def model_select_image(self, output_text, locs, images_paths, progress):
        print('model_select_image')
        pre_text = ''
        pre_img = []
        pre_text_list = []
        ans2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        selected = {k: 0 for k in locs}
        for i, text in enumerate(output_text):
            pre_text += text + '\n'
            if i in locs:
                images = copy.deepcopy(pre_img)
                for j in range(len(images_paths[i])):
                    image = self.vis_processor(images_paths[i][j])
                    images.append(image)
                images = torch.stack(images, dim=0)

                pre_text_list.append(pre_text)
                pre_text = ''

                images = images.cuda()
                text = '根据给定上下文和候选图像，选择合适的配图：' + '<image>'.join(pre_text_list) + '候选图像包括: ' + '\n'.join([chr(ord('A') + j) + '.<image>' for j in range(len(images_paths[i]))])
                input_text = self.text2instruction(text) + '最合适的图是'
                print(input_text)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        img_embeds = self.model.encode_img(images)
                        input_embeds, im_mask, len_input_tokens = self.interleav_wrap(input_text, img_embeds)

                with torch.no_grad():
                    outputs = self.model.generate(
                                            inputs_embeds=input_embeds,
                                            do_sample=True,
                                            temperature=1.,
                                            max_new_tokens=10,
                                            repetition_penalty=1.005,
                                            top_p=0.8,
                                            top_k=40,
                                            length_penalty=1.0,
                                            im_mask=im_mask
                                            )
                response = outputs[0][2:].tolist()   #<s>: C
                #print(response)
                out_text = self.model.tokenizer.decode(response, add_special_tokens=True)
                print(out_text)

                try:
                    answer = out_text.lstrip()[0]
                    pre_img.append(images[len(pre_img) + ans2idx[answer]].cpu())
                except:
                    print('Select fail, use first image')
                    answer = 'A'
                    pre_img.append(images[len(pre_img) + ans2idx[answer]].cpu())
                selected[i] = ans2idx[answer]
        return selected

    def model_select_imagebase(self, output_text, locs, imagebase, progress):
        print('model_select_imagebase')
        pre_text = ''
        pre_img = []
        pre_text_list = []
        selected = []

        images = []
        for j in range(len(imagebase)):
            image = self.vis_processor(imagebase[j])
            images.append(image)
        images = torch.stack(images, dim=0).cuda()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                img_embeds = self.model.encode_img(images)

        for i, text in enumerate(output_text):
            pre_text += text + '\n'
            if i in locs:
                pre_text_list.append(pre_text)
                pre_text = ''
                print(img_embeds.shape)
                cand_embeds = torch.stack([item for j, item in enumerate(img_embeds) if j not in selected], dim=0)
                ans2idx = {}
                count = 0
                for j in range(len(img_embeds)):
                    if j not in selected:
                        ans2idx[chr(ord('A') + count)] = j
                        count += 1

                if cand_embeds.shape[0] > 1:
                    text = '根据给定上下文和候选图像，选择合适的配图：' + '<image>'.join(pre_text_list) + '候选图像包括: ' + '\n'.join([chr(ord('A') + j) + '.<image>' for j in range(len(cand_embeds))])
                    input_text = self.text2instruction(text) + '最合适的图是'
                    print(input_text)

                    all_img = cand_embeds if len(pre_img) == 0 else torch.cat(pre_img + [cand_embeds], dim=0)
                    input_embeds, im_mask, len_input_tokens = self.interleav_wrap(input_text, all_img)

                    with torch.no_grad():
                        outputs = self.model.generate(
                                                inputs_embeds=input_embeds,
                                                do_sample=True,
                                                temperature=1.,
                                                max_new_tokens=10,
                                                repetition_penalty=1.005,
                                                top_p=0.8,
                                                top_k=40,
                                                length_penalty=1.0,
                                                im_mask=im_mask
                                                )
                    response = outputs[0][2:].tolist()   #<s>: C
                    #print(response)
                    out_text = self.model.tokenizer.decode(response, add_special_tokens=True)
                    print(out_text)

                    try:
                        answer = out_text.lstrip()[0]
                    except:
                        print('Select fail, use first image')
                        answer = 'A'
                else:
                    answer = 'A'

                pre_img.append(img_embeds[ans2idx[answer]].unsqueeze(0))
                selected.append(ans2idx[answer])
        selected = {loc: j for loc, j in zip(locs, selected)}
        print(selected)
        return selected

    def show_article(self, show_cap=False):
        md_shows = []
        imgs_show = []
        edit_bts = []
        for i in range(len(self.texts_imgs)):
            text, img = self.texts_imgs[i]
            md_shows.append(gr.Markdown(visible=True, value=text))
            edit_bts.append(gr.Button(visible=True, interactive=True, ))
            imgs_show.append(gr.Image(visible=False) if img is None else gr.Image(visible=True, value=img.paths[img.pts]))

        print(f'show {len(md_shows)} text sections')
        for _ in range(max_section - len(self.texts_imgs)):
            md_shows.append(gr.Markdown(visible=False, value=''))
            edit_bts.append(gr.Button(visible=False))
            imgs_show.append(gr.Image(visible=False))

        return md_shows + edit_bts + imgs_show

    def generate_article(self, instruction, upimages, beam, repetition, max_length, random, seed):
        self.reset()
        set_random_seed(int(seed))
        self.hash_folder = hashlib.sha256(instruction.encode()).hexdigest()
        self.instruction = instruction
        if upimages is None:
            upimages = []
        else:
            upimages = [t.image.path for t in upimages.root]
        img_instruction = '<image> ' * len(upimages)
        instruction = img_instruction.strip() + instruction
        text = self.text2instruction(instruction)
        print('random generate:{}'.format(random))
        if article_stream_output:
            if len(upimages) == 0:
                input_ids = self.model.tokenizer(text, return_tensors="pt")['input_ids']
                input_embeds = self.model.model.tok_embeddings(input_ids.cuda())
                im_mask = None
            else:
                images = []
                for j in range(len(upimages)):
                    image = self.vis_processor(upimages[j])
                    images.append(image)
                images = torch.stack(images, dim=0)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        img_embeds = self.model.encode_img(images)

                text = self.text2instruction(instruction)

                input_embeds, im_mask, len_input_tokens = self.interleav_wrap(text, img_embeds)

            print(text)
            generate_params = dict(
                inputs_embeds=input_embeds,
                do_sample=random,
                stopping_criteria=self.stopping_criteria,
                repetition_penalty=float(repetition),
                max_new_tokens=max_length,
                top_p=0.8,
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
                    yield (output_text,) + (gr.Markdown(visible=False),) * (max_section - 1) + (
                            gr.Button(visible=False),) * max_section + (gr.Image(visible=False),) * max_section
                    time.sleep(0.01)
            output_text = output_text[:-1]
            yield (output_text,) + (gr.Markdown(visible=False),) * (max_section - 1) + (
                            gr.Button(visible=False),) * max_section + (gr.Image(visible=False),) * max_section
        else:
            output_text = self.generate(text, random, beam, max_length, repetition)

        output_text = re.sub(r'(\n\s*)+', '\n', output_text.strip())
        print(output_text)

        output_text = output_text.split('\n')[:max_section]

        self.texts_imgs = [[t, None] for t in output_text]
        self.database.addtitle(text, self.hash_folder,
                               params={'beam': beam, 'repetition': repetition, 'max_length': max_length,
                                       'random': random, 'seed': seed})

        if article_stream_output:
            yield self.show_article()
        else:
            return self.show_article()

    def insert_images(self, upimages, llmO, img_num, seed, progress=gr.Progress()):
        set_random_seed(int(seed))
        if not llmO:
            output_text = [t[0] for t in self.texts_imgs]
            idx_text_sections = [f'<Seg{i}>' + ' ' + it + '\n' for i, it in enumerate(output_text)]

            if img_num == 'Automatic (自动)':
                img_num = ''

            if upimages is None:
                upimages = []
            else:
                upimages = [t.image.path for t in upimages.root]

            inject_text, locs = self.generate_loc(idx_text_sections, upimages, img_num)
            if len(upimages) == 0:
                caps = self.generate_cap(idx_text_sections, locs, progress)

                self.ex_idxs = []
                images_paths = {}
                for loc, cap in progress.tqdm(caps.items(), desc="download image"):
                    idxs = self.get_images_xlab(cap, self.img_pt, self.ex_idxs)
                    paths = [os.path.join('articles', self.hash_folder, f'temp_{self.img_pt}_{i}.png') for i in range(4)]
                    images_paths[loc] = [it for it in paths if os.path.exists(it)]
                    if len(images_paths[loc]) == 0:
                        gr.Warning('Image download fail !!!')
                        del images_paths[loc]
                    self.img_pt += 1
                    self.ex_idxs.extend(idxs)

                locs = [k for k in caps.keys() if k in images_paths]

                if True:
                    selected = self.model_select_image(output_text, locs, images_paths, progress)
                else:
                    selected = {k: 0 for k in locs}

                self.texts_imgs = [
                    [t, ImageGroup(caps[i], images_paths[i], selected[i])] if i in selected else [t, None]
                    for i, t in enumerate(output_text)]
            else:
                selected = self.model_select_imagebase(output_text, locs, upimages, progress)
                self.texts_imgs = [[t, ImageGroup('', [upimages[selected[i]]])] if i in selected else [t, None] for i, t in enumerate(output_text)]

        self.database.addarticle(self.texts_imgs)
        return self.show_article()

    def show_edit(self, text, pt):
        if self.open_edit and pt != self.pt:
            gr.Warning('Please close the editing panel before open another editing panel !!!')
            return gr.Accordion(visible=False), text, ''
        else:
            self.pt = pt
            self.open_edit = True
            return gr.Accordion(visible=True), text, ''

    def show_gallery(self, img_pt):
        if self.open_edit and img_pt != self.pt:
            gr.Warning('Please close the editing panel before open another editing panel !!!')
            return gr.Accordion(visible=False), '', gr.Gallery()
        elif len(self.texts_imgs[img_pt][1].paths) == 1:
            gr.Warning('This imag can not be edited !!!')
            return gr.Accordion(visible=False), '', gr.Gallery()
        else:
            self.pt = img_pt
            self.open_edit = True
            gallery = gr.Gallery(value=self.texts_imgs[img_pt][1].paths)
            return gr.Accordion(visible=True), self.texts_imgs[img_pt][1].cap, gallery

    def hide_edit(self, flag=False):
        self.open_edit = flag
        return gr.Accordion(visible=False), None, gr.Textbox(value='', interactive=False), ''

    def hide_gallery(self):
        self.open_edit = False
        self.database.addedit('changeimage', '', self.texts_imgs)
        return gr.Accordion(visible=False), '', gr.Gallery(value=None)

    def delete_gallery(self, pt):
        self.texts_imgs[pt][1] = None
        return [gr.Image(visible=False, value=None)] + list(self.hide_gallery())

    def edit_types_change(self, edit_type):
        if edit_type in ['缩写', 'abbreviate']:
            return gr.Textbox(interactive=False)
        elif edit_type in ['扩写', '改写', '前插入一段', '后插入一段', 'expand', 'rewrite', 'insert a paragraph before', 'insert a paragraph after']:
            return gr.Textbox(interactive=True)

    def insert_image(self):
        return list(self.hide_edit(flag=True)) + [gr.Accordion(visible=True), '', gr.Gallery(visible=True, value=None)]

    def done_edit(self, edit_type, inst_edit, new_text):
        if new_text == '':
            self.texts_imgs = self.texts_imgs[:self.pt] + self.texts_imgs[self.pt+1:]
        else:
            sub_text = re.sub(r'\n+', ' ', new_text)
            if edit_type in ['扩写', '缩写', '改写', 'expand', 'rewrite', 'abbreviate']:
                self.texts_imgs[self.pt][0] = sub_text
            elif edit_type in ['前插入一段', 'insert a paragraph before']:
                self.texts_imgs = self.texts_imgs[:self.pt] + [[sub_text, None]] + self.texts_imgs[self.pt:]
            elif edit_type in ['后插入一段', 'insert a paragraph after']:
                self.texts_imgs = self.texts_imgs[:self.pt+1] + [[sub_text, None]] + self.texts_imgs[self.pt+1:]
            else:
                print(new_text)
                assert 0 == 1

        self.database.addedit(edit_type, inst_edit, self.texts_imgs)
        return list(self.hide_edit()) + list(self.show_article())

    def paragraph_edit(self, edit_type, text, instruction, pts):
        if edit_type in ['扩写', 'expand']:
            inst_text = f'扩写以下段落：{text}\n基于以下素材：{instruction}'
        elif edit_type in ['改写', 'rewrite']:
            inst_text = f'改写以下段落：{text}\n基于以下素材：{instruction}'
        elif edit_type in ['缩写', 'abbreviate']:
            inst_text = '缩写以下段落：' + text
        elif edit_type in ['前插入一段', 'insert a paragraph before']:
            pre_text = '' if pts == 0 else self.texts_imgs[pts-1][0]
            inst_text = f'在以下两段中插入一段。\n第一段：{pre_text}\n第二段：{text}\n插入段的大纲：{instruction}'
        elif edit_type in ['后插入一段', 'insert a paragraph after']:
            post_text = '' if pts + 1 >= len(self.texts_imgs) else self.texts_imgs[pts + 1][0]
            inst_text = f'在以下两段中插入一段。\n第一段：{text}\n第二段：{post_text}\n插入段的大纲：{instruction}'
        elif edit_type is None:
            if article_stream_output:
                yield text, gr.Button(interactive=True), gr.Button(interactive=True)
            else:
                return text, gr.Button(interactive=True), gr.Button(interactive=True)

        if instruction == '' and edit_type in ['前插入一段', '后插入一段', 'insert a paragraph before', 'insert a paragraph after']:
            gr.Warning('Please input the instruction !!!')
            if article_stream_output:
                yield '', gr.Button(interactive=True), gr.Button(interactive=True)
            else:
                return '', gr.Button(interactive=True), gr.Button(interactive=True)
        else:
            print(inst_text)
            instruction = self.text2instruction(inst_text)
            if article_stream_output:
                input_ids = self.model.tokenizer(instruction, return_tensors="pt")['input_ids']
                len_input_tokens = len(input_ids[0])
                input_embeds = self.model.model.tok_embeddings(input_ids.cuda())
                generate_params = dict(
                    inputs_embeds=input_embeds,
                    do_sample=True,
                    stopping_criteria=self.stopping_criteria,
                    repetition_penalty=1.005,
                    max_length=500 - len_input_tokens,
                    top_p=0.8,
                    top_k=40,
                    length_penalty=1.0
                )
                output_text = "▌"
                with self.generate_with_streaming(**generate_params) as generator:
                    for output in generator:
                        decoded_output = self.model.tokenizer.decode(output[1:])
                        if output[-1] in [self.model.tokenizer.eos_token_id, 92542]:
                            break
                        output_text = decoded_output.replace('\n', '\n\n') + "▌"
                        yield output_text, gr.Button(interactive=False), gr.Button(interactive=False)
                        time.sleep(0.1)
                output_text = output_text[:-1]
                print(output_text)
                yield output_text, gr.Button(interactive=True), gr.Button(interactive=True)
            else:
                output_text = self.generate(text, True, 1, 500, 1.005)
                return output_text, gr.Button(interactive=True), gr.Button(interactive=True)

    def search_image(self, text, pt):
        if text == '':
            return gr.Gallery()

        idxs = self.get_images_xlab(text, self.img_pt, self.ex_idxs)
        images_paths = [os.path.join('articles', self.hash_folder, f'temp_{self.img_pt}_{i}.png') for i in
                             range(4)]
        self.img_pt += 1
        self.ex_idxs.extend(idxs)

        self.texts_imgs[pt][1] = ImageGroup(text, images_paths)

        ga_show = gr.Gallery(visible=True, value=images_paths)
        return ga_show, gr.Image(visible=True, value=images_paths[0])

    def replace_image(self, pt, evt: gr.SelectData):
        self.texts_imgs[pt][1].pts = evt.index
        img = self.texts_imgs[pt][1]
        return gr.Image(visible=True, value=img.paths[img.pts])

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

    def change_meta(self, withmeta):
        self.withmeta = withmeta

    def upload_images(self, files):
        if len(files) > 10:
            gr.Warning('No more than 10 images !!!')
            files = files[:10]
        return gr.Gallery(value=files), gr.Dropdown(value=str(len(files)))

    def clear_images(self):
        return gr.Gallery(value=None)

    def limit_imagenum(self, img_num, upshows):
        if upshows is None:
            return gr.Dropdown()
        maxnum = len(upshows.root)
        if img_num == 'Automatic (自动)' or int(img_num) > maxnum:
            img_num = str(maxnum)
        return gr.Dropdown(value=img_num)

    def chat_ask(self, state, img_list, text, image):
        print(1111)
        state.skip_next = False
        if len(text) <= 0 and image is None:
            state.skip_next = True
            return (state, img_list, state.to_gradio_chatbot(), "",
                    None) + (gr.Button(), ) * 2

        if image is not None:
            imgs = []
            imgs_pil = []
            for j in range(len(image)):
                img_pil = Image.open(image[j]).convert('RGB')
                imgs_pil.append(img_pil)
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
            state.messages[-1][1][0] = ''.join(
                [state.messages[-1][1][0], text])
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
            state.messages[-1][-1] = state.messages[-1][-1][:-1]
            if self.chat_folder and os.path.exists(self.chat_folder):
                with open(os.path.join(self.chat_folder, 'chat.txt'), 'a+') as fd:
                    if isinstance(state.messages[-2][1], str):
                        fd.write(state.messages[-2][0] + state.messages[-2][1])
                    else:
                        fd.write(state.messages[-2][0] + state.messages[-2][1][0])
                    fd.write(state.messages[-1][0] + state.messages[-1][1])

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

    def enable_like(self):
        return [gr.Button(visible=True)] * 2

    def like(self):
        with open(os.path.join(self.database.folder, 'like.txt'), 'w') as fd:
            fd.write('like')
        return [gr.Button(visible=False)] * 2

    def dislike(self):
        with open(os.path.join(self.database.folder, 'like.txt'), 'w') as fd:
            fd.write('dislike')
        return [gr.Button(visible=False)] * 2

    def uploadimgs(self, images):
        timestamp = datetime.now()
        self.chat_folder = os.path.join('databases', timestamp.strftime("%Y%m%d"), 'chat', str(timestamp).replace(' ', '-'))
        os.makedirs(self.chat_folder, exist_ok=True)
        for image_path in images:
            shutil.copy(image_path, self.chat_folder)

    def chat_like(self):
        if self.chat_folder and os.path.exists(self.chat_folder):
            with open(os.path.join(self.chat_folder, 'chat.txt'), 'a+') as fd:
                fd.write('#like#')
        return [gr.Button(interactive=False)] * 2

    def chat_dislike(self):
        if self.chat_folder and os.path.exists(self.chat_folder):
            with open(os.path.join(self.chat_folder, 'chat.txt'), 'a+') as fd:
                fd.write('#dislike#')
        return [gr.Button(interactive=False)] * 2


def load_demo():
    state = CONV_VISION_INTERN2.copy()

    return (state, [], gr.Chatbot(visible=True),
            gr.Textbox(visible=True), gr.Button(visible=True),
            gr.Row(visible=True), gr.Accordion(visible=True))


def change_language(lang):
    edit_types, inst_edits, insertIMGs, edit_dones, edit_cancels = [], [], [], [], []
    cap_searchs, gallery_dels, gallery_dones = [], [], []
    if lang == '中文':
        lang_btn = gr.Button(value='English')
        for _ in range(max_section):
            edit_types.append(gr.Radio(["改写", "扩写", "缩写", "前插入一段", "后插入一段"]))
            inst_edits.append(gr.Textbox(label='插入段落时，请输入插入段的大纲：'))
            insertIMGs.append(gr.Button(value='在下方插入图片'))
            edit_dones.append(gr.Button(value='确定'))
            edit_cancels.append(gr.Button(value='取消'))
            cap_searchs.append(gr.Button(value='搜图'))
            gallery_dels.append(gr.Button(value='删除'))
            gallery_dones.append(gr.Button(value='确定'))

        chat_textbox = gr.update(placeholder='输入聊天内容并回车')
        submit_btn = gr.update(value='提交')
        regenerate_btn = gr.update(value='🔄  重新生成')
        clear_btn = gr.update(value='🗑️  清空聊天框')
    elif lang == 'English':
        lang_btn = gr.Button(value='中文')
        for _ in range(max_section):
            edit_types.append(gr.Radio(["rewrite", "expand", "abbreviate", "insert a paragraph before", "insert a paragraph after"]))
            inst_edits.append(gr.Textbox(label='Instrcution for editing:'))
            insertIMGs.append(gr.Button(value='Insert image below'))
            edit_dones.append(gr.Button(value='Done'))
            edit_cancels.append(gr.Button(value='Cancel'))
            cap_searchs.append(gr.Button(value='Search'))
            gallery_dels.append(gr.Button(value='Delete'))
            gallery_dones.append(gr.Button(value='Done'))

        chat_textbox = gr.update(placeholder='Enter text and press ENTER')
        submit_btn = gr.update(value='Submit')
        regenerate_btn = gr.update(value='🔄  Regenerate')
        clear_btn = gr.update(value='🗑️  Clear history')

    return [lang_btn] + edit_types + inst_edits + insertIMGs + edit_dones + edit_cancels + cap_searchs + gallery_dels + gallery_dones + [chat_textbox, submit_btn, regenerate_btn, clear_btn]


parser = argparse.ArgumentParser()
parser.add_argument("--code_path", default='internlm/internlm-xcomposer2-7b')
parser.add_argument("--chat_path", default='internlm/internlm-xcomposer2-vl-7b')
parser.add_argument("--private", default=False, action='store_true')
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--port", default=11111, type=int)
args = parser.parse_args()
demo_ui = Demo_UI(args.code_path, args.chat_path, args.num_gpus)


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
        with gr.TabItem("📝 Write Interleaved-text-image Article (创作图文并茂文章)"):
            with gr.Row():
                with gr.Column(scale=2):
                    instruction = gr.Textbox(label='Write an illustrated article based on the given instruction: (根据素材或指令创作图文并茂的文章)',
                                             lines=5,
                                             value='''根据以下标题：“中国水墨画：流动的诗意与东方美学”，创作长文章，字数不少于800字。请结合以下文本素材：
“水墨画是由水和墨调配成不同深浅的墨色所画出的画，是绘画的一种形式，更多时候，水墨画被视为中国传统绘画，也就是国画的代表。也称国画，中国画。墨水画是中国传统画之一。墨水是国画的起源，以笔墨运用的技法基础画成墨水画。线条中锋笔，侧锋笔，顺锋和逆锋，点染，擦，破墨，拨墨的技法。墨于水的变化分为五色。画成作品，题款，盖章。就是完整的墨水画作品。
基本的水墨画，仅有水与墨，黑与白色，但进阶的水墨画，也有工笔花鸟画，色彩缤纷。后者有时也称为彩墨画。在中国画中，以中国画特有的材料之一，墨为主要原料加以清水的多少引为浓墨、淡墨、干墨、湿墨、焦墨等，画出不同浓淡（黑、白、灰）层次。别有一番韵味称为“墨韵”。而形成水墨为主的一种绘画形式。”''')
                with gr.Column(scale=1):
                    img_num = gr.Dropdown(
                        ["Automatic (自动)", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
                        value='6', label="Image Number (插图数量)", info="Select the number of the inserted images",
                        interactive=True)
                    seed = gr.Slider(minimum=1.0, maximum=20000.0, value=8909.0, step=1.0, label='Random Seed (随机种子)')
                    btn = gr.Button("Submit (提交)", scale=1)

            with gr.Accordion("Click to add image material (点击添加图片素材）, optional（可选）", open=False, visible=True):
                with gr.Row():
                    uploads = gr.File(file_count='multiple', scale=1)
                    upshows = gr.Gallery(columns=4, scale=2)

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Advanced Settings (高级设置)", open=False, visible=True) as parameter_article:
                        beam = gr.Slider(minimum=1.0, maximum=6.0, value=1.0, step=1.0, label='Beam Size (集束大小)')
                        repetition = gr.Slider(minimum=1.0, maximum=2.0, value=1.005, step=0.001, label='Repetition_penalty (重复惩罚)')
                        text_num = gr.Slider(minimum=100.0, maximum=4096.0, value=4096.0, step=1.0, label='Max output tokens (最多输出字数)')
                        llmO = gr.Checkbox(value=False, label='LLM Only (纯文本写作)')
                        random = gr.Checkbox(value=True, label='Sampling (随机采样)')
                        withmeta = gr.Checkbox(value=False, label='With Meta (使用meta指令)')

            with gr.Row():
                btn_like = gr.Button(interactive=True, visible=False, value='👍  Like This Article (点赞这篇文章)')
                btn_dislike = gr.Button(interactive=True, visible=False, value='👎  Dislike This Article (点踩这篇文章)')

            articles, edit_bts = [], []
            text_editers, edit_types, edit_subs, insertIMGs, edit_dones, edit_cancels = [], [], [], [], [], []
            before_edits, inst_edits, after_edits = [], [], []
            img_editers, cap_boxs, cap_searchs, gallerys, gallery_dels, gallery_dones = [], [], [], [], [], []
            image_shows = []
            with gr.Column():
                for i in range(max_section):
                    visible = True if i == 0 else False
                    with gr.Row():
                        with gr.Column(scale=3):
                            articles.append(gr.Markdown(visible=visible, elem_classes='feedback'))

                        edit_bts.append(gr.Button(interactive=True, visible=False, value='\U0001F58C', elem_classes='sm_btn'))
                        with gr.Column(scale=2):
                            with gr.Accordion('Text Editing (文本编辑)', open=True, visible=False) as text_editer:
                                gr.HTML('<p style="color:gray;">Befor Editing (编辑前):</p>', elem_classes='edit')
                                gr.HTML('<p style="color:gray;">===========</p>', elem_classes='edit')
                                before_edits.append(gr.HTML('', elem_classes='edit'))
                                gr.HTML('<p style="color:gray;">===========</p>', elem_classes='edit')

                                edit_types.append(gr.Radio(["rewrite", "expand", "abbreviate", "insert a paragraph before", "insert a paragraph after"], label="Paragraph modification (段落修改)",
                                                     info="选择后点击右侧按钮，模型自动修改", elem_classes='editsmall'))
                                with gr.Row():
                                    inst_edits.append(gr.Textbox(label='Instrcution for editing:', interactive=False, elem_classes='editsmall'))
                                    edit_subs.append(gr.Button(elem_classes='smax_btn'))

                                gr.HTML('<p style="color:gray;">After Editing (编辑后):</p>', elem_classes='edit')
                                after_edits.append(gr.Textbox(show_label=False, elem_classes='edit'))

                                with gr.Row():
                                    insertIMGs.append(gr.Button(value='Insert image below'))
                                    edit_dones.append(gr.Button(value='Done'))
                                    edit_cancels.append(gr.Button(value='Cancel'))

                    with gr.Row():
                        with gr.Column(scale=3):
                            image_shows.append(gr.Image(visible=False, width=600, elem_classes='feedback'))

                        with gr.Column(scale=2):
                            with gr.Accordion('Images Editing (图片编辑)', open=True, visible=False) as img_editer:
                                with gr.Row():
                                    cap_boxs.append(gr.Textbox(label='image caption (图片标题)', interactive=True, scale=6))
                                    cap_searchs.append(gr.Button(value="Search", scale=1))
                                with gr.Row():
                                    gallerys.append(gr.Gallery(columns=2, height='auto'))

                                with gr.Row():
                                    gallery_dels.append(gr.Button(value="Delete"))
                                    gallery_dones.append(gr.Button(value="Done"))

                    text_editers.append(text_editer)
                    img_editers.append(img_editer)

            save_btn = gr.Button("Save article (保存文章)")
            save_file = gr.File(label="Save article (保存文章)")
            save_btn.click(demo_ui.save, inputs=[beam, repetition, text_num, random, seed], outputs=save_file)

            uploads.upload(demo_ui.upload_images, inputs=uploads, outputs=[upshows, img_num])
            uploads.clear(demo_ui.clear_images, inputs=[], outputs=upshows)
            img_num.select(demo_ui.limit_imagenum, inputs=[img_num, upshows], outputs=img_num)

            withmeta.change(demo_ui.change_meta, inputs=withmeta, outputs=[])

            for i in range(max_section):
                edit_bts[i].click(demo_ui.show_edit, inputs=[articles[i], gr.Number(value=i, visible=False)], outputs=[text_editers[i], before_edits[i], after_edits[i]])
                edit_types[i].select(demo_ui.edit_types_change, inputs=[edit_types[i]], outputs=[inst_edits[i]])
                edit_subs[i].click(demo_ui.paragraph_edit, inputs=[edit_types[i], before_edits[i], inst_edits[i], gr.Number(value=i, visible=False)], outputs=[after_edits[i], insertIMGs[i], edit_dones[i]])
                insertIMGs[i].click(demo_ui.insert_image, inputs=[], outputs=[text_editers[i], edit_types[i], inst_edits[i], after_edits[i], img_editers[i], cap_boxs[i], gallerys[i]])
                edit_dones[i].click(demo_ui.done_edit, inputs=[edit_types[i], inst_edits[i], after_edits[i]], outputs=[text_editers[i], edit_types[i], inst_edits[i], after_edits[i]] + articles + edit_bts + image_shows)
                edit_cancels[i].click(demo_ui.hide_edit, inputs=[], outputs=[text_editers[i], edit_types[i], inst_edits[i], after_edits[i]])

                image_shows[i].select(demo_ui.show_gallery, inputs=[gr.Number(value=i, visible=False)], outputs=[img_editers[i], cap_boxs[i], gallerys[i]])
                cap_searchs[i].click(demo_ui.search_image, inputs=[cap_boxs[i], gr.Number(value=i, visible=False)], outputs=[gallerys[i], image_shows[i]])
                gallerys[i].select(demo_ui.replace_image, inputs=[gr.Number(value=i, visible=False)], outputs=[image_shows[i]])
                gallery_dels[i].click(demo_ui.delete_gallery, inputs=[gr.Number(value=i, visible=False)], outputs=[image_shows[i], img_editers[i], cap_boxs[i], gallerys[i]])
                gallery_dones[i].click(demo_ui.hide_gallery, inputs=[], outputs=[img_editers[i], cap_boxs[i], gallerys[i]])

            btn_like.click(demo_ui.like, inputs=[], outputs=[btn_like, btn_dislike])
            btn_dislike.click(demo_ui.dislike, inputs=[], outputs=[btn_like, btn_dislike])

            btn.click(demo_ui.reset_components, inputs=[],
                      outputs=articles + edit_bts + image_shows + text_editers + img_editers).then(
                    demo_ui.generate_article,
                    inputs=[instruction, upshows, beam, repetition, text_num, random, seed],
                    outputs=articles + edit_bts + image_shows).then(
                    demo_ui.insert_images, inputs=[upshows, llmO, img_num, seed], outputs=articles + edit_bts + image_shows).then(
                    demo_ui.enable_like, inputs=[], outputs=[btn_like, btn_dislike])

        with gr.TabItem("💬 Multimodal Chat (多模态对话)", elem_id="chat", id=0):
            chat_state = gr.State()
            img_list = gr.State()
            with gr.Row():
                with gr.Column(scale=3):
                    imagebox = gr.File(file_count='multiple')

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
                        chat_btn_like = gr.Button(interactive=False, value='👍  Like (点赞)')
                        chat_btn_dislike = gr.Button(interactive=False, value='👎  Dislike (点踩)')

            btn_list = [regenerate_btn, clear_btn, chat_btn_like, chat_btn_dislike]
            parameter_list = [
                chat_max_output_tokens, chat_repetition_penalty,
                chat_num_beams, chat_do_sample
            ]

            imagebox.upload(demo_ui.uploadimgs, imagebox, [])
            chat_btn_like.click(demo_ui.chat_like, [], [chat_btn_like, chat_btn_dislike])
            chat_btn_dislike.click(demo_ui.chat_dislike, [], [chat_btn_like, chat_btn_dislike])

            chat_textbox.submit(
                demo_ui.chat_ask,
                [chat_state, img_list, chat_textbox, imagebox],
                [chat_state, img_list, chatbot, chat_textbox, imagebox] +
                btn_list).then(demo_ui.chat_answer,
                               [chat_state, img_list] + parameter_list,
                               [chat_state, chatbot] + btn_list)
            submit_btn.click(
                demo_ui.chat_ask,
                [chat_state, img_list, chat_textbox, imagebox],
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

    lang_btn.click(change_language, inputs=lang_btn, outputs=[lang_btn] + edit_types + inst_edits + insertIMGs + edit_dones + edit_cancels + cap_searchs + gallery_dels + gallery_dones + [chat_textbox, submit_btn, regenerate_btn, clear_btn])

if __name__ == "__main__":
    demo.launch()

