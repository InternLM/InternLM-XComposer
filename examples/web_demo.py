import os
import re
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import gradio as gr
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), 'tmp')
import copy
import time
import shutil
import requests
from PIL import Image, ImageFile
import torch
import transformers
from transformers import StoppingCriteriaList, AutoTokenizer, AutoModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

from demo_asset.assets.css_html_js import custom_css
from demo_asset.gradio_patch import Chatbot as grChatbot
from demo_asset.serve_utils import Stream, Iteratorize
from demo_asset.conversation import CONV_VISION_7132_v2, StoppingCriteriaSub
from demo_asset.download import download_image_thread

max_section = 60
no_change_btn = gr.Button.update()
disable_btn = gr.Button.update(interactive=False)
enable_btn = gr.Button.update(interactive=True)
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


class Demo_UI:
    def __init__(self):
        #self.llm_model = AutoModel.from_pretrained(
        #    'internlm/internlm-xcomposer-7b', trust_remote_code=True)
        #tokenizer = AutoTokenizer.from_pretrained(
        #    'internlm/internlm-xcomposer-7b', trust_remote_code=True)
        self.llm_model = AutoModel.from_pretrained('/mnt/petrelfs/share_data/dongxiaoyi/share_models/release_chat', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('/mnt/petrelfs/share_data/dongxiaoyi/share_models/release_chat', trust_remote_code=True)
        self.llm_model.internlm_tokenizer = tokenizer
        self.llm_model.tokenizer = tokenizer
        self.llm_model.eval().to('cuda')
        self.device = 'cuda'
        print(f" load model done: ", type(self.llm_model))

        self.eoh = self.llm_model.internlm_tokenizer.decode(
            torch.Tensor([103027]), skip_special_tokens=True)
        self.eoa = self.llm_model.internlm_tokenizer.decode(
            torch.Tensor([103028]), skip_special_tokens=True)

        self.vis_processor = self.llm_model.vis_processor
        self.device = 'cuda'

        stop_words_ids = [
            torch.tensor([943]).to(self.device),
            torch.tensor([2917, 44930]).to(self.device),
            torch.tensor([45623]).to(self.device),  ### new setting
            torch.tensor([46323]).to(self.device),  ### new setting
            torch.tensor([103027]).to(self.device),  ### new setting
            torch.tensor([103028]).to(self.device),  ### new setting
        ]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])
        self.r2 = re.compile(r'<Seg[0-9]*>')

    def reset(self):
        self.output_text = ''
        self.caps = {}
        self.show_caps = False
        self.show_ids = {}

    def get_images_xlab(self, caption, loc, exclude):
        urls, idxs = get_urls(caption.strip()[:53], exclude)
        print(urls[0])
        print('download image with url')
        download_image_thread(urls,
                              folder='articles/' + self.title,
                              index=self.show_ids[loc] * 1000 + loc,
                              num_processes=4)
        print('image downloaded')
        return idxs

    def generate(self, text, random, beam, max_length, repetition):
        input_tokens = self.llm_model.internlm_tokenizer(
            text, return_tensors="pt",
            add_special_tokens=True).to(self.llm_model.device)
        img_embeds = self.llm_model.internlm_model.model.embed_tokens(
            input_tokens.input_ids)
        with torch.no_grad():
            with self.llm_model.maybe_autocast():
                outputs = self.llm_model.internlm_model.generate(
                    inputs_embeds=img_embeds,
                    stopping_criteria=self.stopping_criteria,
                    do_sample=random,
                    num_beams=beam,
                    max_length=max_length,
                    repetition_penalty=float(repetition),
                )
        output_text = self.llm_model.internlm_tokenizer.decode(
            outputs[0][1:], add_special_tokens=False)
        output_text = output_text.split('<TOKENS_UNUSED_1>')[0]
        return output_text

    def generate_text(self, title, beam, repetition, text_num, random):
        text = ' <|User|>:根据给定标题写一个图文并茂，不重复的文章：{}\n'.format(
            title) + self.eoh + ' <|Bot|>:'
        print('random generate:{}'.format(random))
        output_text = self.generate(text, random, beam, text_num, repetition)
        return output_text

    def generate_loc(self, text_sections, image_num, progress):
        full_txt = ''.join(text_sections)
        input_text = f' <|User|>:给定文章"{full_txt}" 根据上述文章，选择适合插入图像的{image_num}行' + ' \n<TOKENS_UNUSED_0> <|Bot|>:适合插入图像的行是'

        for _ in progress.tqdm([1], desc="image spotting"):
            output_text = self.generate(input_text,
                                        random=False,
                                        beam=5,
                                        max_length=300,
                                        repetition=1.)
        inject_text = '适合插入图像的行是' + output_text
        print(inject_text)

        locs = []
        for m in self.r2.findall(inject_text):
            locs.append(int(m[4:-1]))
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

            input_text = f' <|User|>: 给定文章"{full_txt}" {past}给出适合在<Seg{po}>后插入的图像对应的标题。' + ' \n<TOKENS_UNUSED_0> <|Bot|>: 标题是"'

            cap_text = self.generate(input_text,
                                     random=False,
                                     beam=1,
                                     max_length=100,
                                     repetition=5.)
            cap_text = cap_text.split('"')[0].strip()
            print(cap_text)
            caps[po] = cap_text

            if idx == 0:
                pasts = f'现在<Seg{po}>后插入图像对应的标题是"{cap_text}"， '
            else:
                pasts += f'<Seg{po}>后插入图像对应的标题是"{cap_text}"， '

        print(caps)
        return caps

    def generate_loc_cap(self, text_sections, image_num, progress):
        inject_text, locs = self.generate_loc(text_sections, image_num,
                                              progress)
        caps = self.generate_cap(text_sections, locs, progress)
        return caps

    def model_select_image(self, out_text, caps, root, progress):
        print('model_select_image')
        pre_img = []
        pre_text_list = []
        ans2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        selected = dict()
        for i, text in progress.tqdm(enumerate(out_text.split('\n')), desc="image selection"):
            if i not in caps:
                continue
            images = copy.deepcopy(pre_img)
            for j in range(4):
                img_path = os.path.join(
                    root, f'temp_{self.show_ids[i] * 1000 + i}_{j}.png')
                images.append(img_path)
            instruct = ' <|User|>:根据给定上下文和候选图像，选择合适的配图：\n\n候选图像包括:'
            prompt_embeds = [self.llm_model.encode_text(instruct)]
            options = ['A', '\nB', '\nC', '\nD']
            for option, img in zip(options, images):
                option_embed = self.llm_model.encode_text(option)
                img_embed = self.llm_model.encode_img(img)
                prompt_embeds.extend([option_embed, img_embed])
            prompt_embeds.append(
                self.llm_model.encode_text('\n\n<TOKENS_UNUSED_0> <|Bot|>:最合适的图是'))
            prompt_embeds = torch.cat(prompt_embeds, dim=1)
            out_embeds = self.llm_model.internlm_model.generate(
                inputs_embeds=prompt_embeds,
                do_sample=False,
                num_beams=5,
                max_length=10,
                repetition_penalty=1.,
            )
            out_option = self.llm_model.decode_text(out_embeds)
            answer = out_option[1] if out_text[0] == ' ' else out_option[0]
            selected[i] = ans2idx[answer]
        return selected

    def show_md(self, text_sections, title, caps, selected, show_cap=False):
        md_shows = []
        ga_shows = []
        btn_shows = []
        cap_textboxs, cap_searchs = [], []
        editers = []
        for i in range(len(text_sections)):
            if i in caps:
                if show_cap:
                    md = text_sections[
                        i] + '\n' + '<div align="center"> <img src="file/articles/{}/temp_{}_{}.png" width = 500/> {} </div>'.format(
                            title, self.show_ids[i] * 1000 + i, selected[i],
                            caps[i])
                else:
                    md = text_sections[
                        i] + '\n' + '<div align="center"> <img src="file=articles/{}/temp_{}_{}.png" width = 500/> </div>'.format(
                            title, self.show_ids[i] * 1000 + i, selected[i])
                img_list = [('articles/{}/temp_{}_{}.png'.format(
                    title, self.show_ids[i] * 1000 + i,
                    j), 'articles/{}/temp_{}_{}.png'.format(
                        title, self.show_ids[i] * 1000 + i, j))
                            for j in range(4)]

                ga_show = gr.Gallery.update(visible=True, value=img_list)
                ga_shows.append(ga_show)

                btn_show = gr.Button.update(visible=True,
                                            value='\U0001f5d1\uFE0F')

                cap_textboxs.append(
                    gr.Textbox.update(visible=True, value=caps[i]))
                cap_searchs.append(gr.Button.update(visible=True))
            else:
                md = text_sections[i]
                ga_show = gr.Gallery.update(visible=False, value=[])
                ga_shows.append(ga_show)

                btn_show = gr.Button.update(visible=True, value='\u2795')
                cap_textboxs.append(gr.Textbox.update(visible=False))
                cap_searchs.append(gr.Button.update(visible=False))

            md_show = gr.Markdown.update(visible=True, value=md)
            md_shows.append(md_show)
            btn_shows.append(btn_show)
            editers.append(gr.update(visible=True))
            print(i, md)

        md_hides = []
        ga_hides = []
        btn_hides = []
        for i in range(max_section - len(text_sections)):
            md_hide = gr.Markdown.update(visible=False, value='')
            md_hides.append(md_hide)

            btn_hide = gr.Button.update(visible=False)
            btn_hides.append(btn_hide)
            editers.append(gr.update(visible=False))

        for i in range(max_section - len(ga_shows)):
            ga_hide = gr.Gallery.update(visible=False, value=[])
            ga_hides.append(ga_hide)
            cap_textboxs.append(gr.Textbox.update(visible=False))
            cap_searchs.append(gr.Button.update(visible=False))

        return md_shows + md_hides + ga_shows + ga_hides + btn_shows + btn_hides + cap_textboxs + cap_searchs + editers, md_shows

    def generate_article(self,
                         title,
                         beam,
                         repetition,
                         text_num,
                         msi,
                         random,
                         progress=gr.Progress()):
        self.reset()
        self.title = title
        if article_stream_output:
            text = ' <|User|>:根据给定标题写一个图文并茂，不重复的文章：{}\n'.format(
                title) + self.eoh + ' <|Bot|>:'
            input_tokens = self.llm_model.internlm_tokenizer(
                text, return_tensors="pt",
                add_special_tokens=True).to(self.llm_model.device)
            img_embeds = self.llm_model.internlm_model.model.embed_tokens(
                input_tokens.input_ids)
            generate_params = dict(
                inputs_embeds=img_embeds,
                num_beams=beam,
                do_sample=random,
                stopping_criteria=self.stopping_criteria,
                repetition_penalty=float(repetition),
                max_length=text_num,
                bos_token_id=self.llm_model.internlm_tokenizer.bos_token_id,
                eos_token_id=self.llm_model.internlm_tokenizer.eos_token_id,
                pad_token_id=self.llm_model.internlm_tokenizer.pad_token_id,
            )
            output_text = "▌"
            with self.generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    decoded_output = self.llm_model.internlm_tokenizer.decode(
                        output[1:])
                    if output[-1] in [
                            self.llm_model.internlm_tokenizer.eos_token_id
                    ]:
                        break
                    output_text = decoded_output.replace('\n', '\n\n') + "▌"
                    yield (output_text,) + (gr.Markdown.update(visible=False),) * (max_section - 1) + (gr.Gallery.update(visible=False),) * max_section + \
                          (gr.Button.update(visible=False),) * max_section + (gr.Textbox.update(visible=False),) * max_section + (gr.Button.update(visible=False),) * max_section + \
                          (gr.update(visible=False),) * max_section + (disable_btn,) * 2
                    time.sleep(0.03)
            output_text = output_text[:-1]
            yield (output_text,) + (gr.Markdown.update(visible=False),) * (max_section - 1) + (gr.Gallery.update(visible=False),) * max_section + \
                  (gr.Button.update(visible=False),) * max_section + (gr.Textbox.update(visible=False),) * max_section + (gr.Button.update(visible=False),) * max_section +\
                  (gr.update(visible=False),) * max_section + (disable_btn,) * 2
        else:
            output_text = self.generate_text(title, beam, repetition, text_num,
                                             random)

        print(output_text)
        output_text = re.sub(r'(\n[ \t]*)+', '\n', output_text)
        if output_text[-1] == '\n':
            output_text = output_text[:-1]
        print(output_text)
        output_text = '\n'.join(output_text.split('\n')[:max_section])

        text_sections = output_text.split('\n')
        idx_text_sections = [
            f'<Seg{i}>' + ' ' + it + '\n' for i, it in enumerate(text_sections)
        ]
        caps = self.generate_loc_cap(idx_text_sections, '', progress)
        #caps = {0: '成都的三日游路线图，包括春熙路、太古里、IFS国金中心、大慈寺、宽窄巷子、奎星楼街、九眼桥（酒吧一条街）、武侯祠、锦里、杜甫草堂、浣花溪公园、青羊宫、金沙遗址博物馆、文殊院、人民公园、熊猫基地、望江楼公园、东郊记忆、建设路小吃街、电子科大清水河校区、三圣乡万福花卉市场、龙湖滨江天街购物广场和返程。', 2: '春熙路的繁华景象，各种时尚潮流的品牌店和美食餐厅鳞次栉比。', 4: 'IFS国金中心的豪华购物中心，拥有众多国际知名品牌的旗舰店和专卖店，同时还有电影院、健身房 配套设施。', 6: '春熙路上的著名景点——太古里，是一个集购物、餐饮、娱乐于一体的高端时尚街区，也是成都著名的网红打卡地之一。', 8: '大慈寺的外观，是一座历史悠久的佛教寺庙，始建于唐朝，有着深厚的文化底蕴和历史价值。'}
        #self.show_ids = {k:0 for k in caps.keys()}
        self.show_ids = {k: 1 for k in caps.keys()}

        print(caps)
        self.ex_idxs = []
        for loc, cap in progress.tqdm(caps.items(), desc="download image"):
            #self.show_ids[loc] += 1
            idxs = self.get_images_xlab(cap, loc, self.ex_idxs)
            self.ex_idxs.extend(idxs)

        if msi:
            self.selected = self.model_select_image(output_text, caps,
                                                    'articles/' + title,
                                                    progress)
        else:
            self.selected = {k: 0 for k in caps.keys()}
        components, md_shows = self.show_md(text_sections, title, caps,
                                            self.selected)
        self.show_caps = False

        self.output_text = output_text
        self.caps = caps
        if article_stream_output:
            yield components + [enable_btn] * 2
        else:
            return components + [enable_btn] * 2

    def adjust_img(self, img_num, progress=gr.Progress):
        text_sections = self.output_text.split('\n')
        idx_text_sections = [
            f'<Seg{i}>' + ' ' + it + '\n' for i, it in enumerate(text_sections)
        ]
        img_num = min(img_num, len(text_sections))
        caps = self.generate_loc_cap(idx_text_sections, int(img_num), progress)
        #caps = {1:'318川藏线沿途的风景照片', 4:'泸定桥的全景照片', 6:'折多山垭口的全景照片', 8:'稻城亚丁机场的全景照片', 10:'姊妹湖的全景照片'}

        print(caps)
        sidxs = []
        for loc, cap in caps.items():
            if loc in self.show_ids:
                self.show_ids[loc] += 1
            else:
                self.show_ids[loc] = 1
            idxs = self.get_images_xlab(cap, loc, sidxs)
            sidxs.extend(idxs)
        self.sidxs = sidxs

        self.selected = {k: 0 for k in caps.keys()}
        components, md_shows = self.show_md(text_sections, self.title, caps,
                                            self.selected)

        self.caps = caps
        return components

    def add_delete_image(self, text, status, index):
        index = int(index)
        if status == '\U0001f5d1\uFE0F':
            if index in self.caps:
                self.caps.pop(index)
                self.selected.pop(index)
            md_show = gr.Markdown.update(value=text.split('\n')[0])
            gallery = gr.Gallery.update(visible=False, value=[])
            btn_show = gr.Button.update(value='\u2795')
            cap_textbox = gr.Textbox.update(visible=False)
            cap_search = gr.Button.update(visible=False)
        else:
            md_show = gr.Markdown.update()
            gallery = gr.Gallery.update(visible=True, value=[])
            btn_show = gr.Button.update(value='\U0001f5d1\uFE0F')
            cap_textbox = gr.Textbox.update(visible=True)
            cap_search = gr.Button.update(visible=True)

        return md_show, gallery, btn_show, cap_textbox, cap_search

    def search_image(self, text, index):
        index = int(index)
        if text == '':
            return gr.Gallery.update()

        if index in self.show_ids:
            self.show_ids[index] += 1
        else:
            self.show_ids[index] = 1
        self.caps[index] = text
        idxs = self.get_images_xlab(text, index, self.ex_idxs)
        self.ex_idxs.extend(idxs)

        img_list = [('articles/{}/temp_{}_{}.png'.format(
            self.title, self.show_ids[index] * 1000 + index,
            j), 'articles/{}/temp_{}_{}.png'.format(
                self.title, self.show_ids[index] * 1000 + index, j))
                    for j in range(4)]
        ga_show = gr.Gallery.update(visible=True, value=img_list)
        return ga_show

    def replace_image(self, article, index, evt: gr.SelectData):
        index = int(index)
        self.selected[index] = evt.index
        if '<div align="center">' in article:
            return re.sub(r'file=.*.png', 'file={}'.format(evt.value), article)
        else:
            return article + '\n' + '<div align="center"> <img src="file={}" width = 500/> </div>'.format(
                evt.value)

    def add_delete_caption(self):
        self.show_caps = False if self.show_caps else True
        text_sections = self.output_text.split('\n')
        components, _ = self.show_md(text_sections,
                                     self.title,
                                     self.caps,
                                     selected=self.selected,
                                     show_cap=self.show_caps)
        return components

    def save(self):
        folder = 'save_articles/' + self.title
        if os.path.exists(folder):
            for item in os.listdir(folder):
                os.remove(os.path.join(folder, item))
        os.makedirs(folder, exist_ok=True)

        save_text = ''
        count = 0
        if len(self.output_text) > 0:
            text_sections = self.output_text.split('\n')
            for i in range(len(text_sections)):
                if i in self.caps:
                    if self.show_caps:
                        md = text_sections[
                            i] + '\n' + '<div align="center"> <img src="temp_{}_{}.png" width = 500/> {} </div>'.format(
                                self.show_ids[i] * 1000 + i, self.selected[i],
                                self.caps[i])
                    else:
                        md = text_sections[
                            i] + '\n' + '<div align="center"> <img src="temp_{}_{}.png" width = 500/> </div>'.format(
                                self.show_ids[i] * 1000 + i, self.selected[i])
                    count += 1
                else:
                    md = text_sections[i]

                save_text += md + '\n\n'
            save_text = save_text[:-2]

        with open(os.path.join(folder, 'io.MD'), 'w') as f:
            f.writelines(save_text)

        for k in self.caps.keys():
            shutil.copy(
                os.path.join(
                    'articles', self.title,
                    f'temp_{self.show_ids[k] * 1000 + k}_{self.selected[k]}.png'
                ), folder)
        archived = shutil.make_archive(folder, 'zip', folder)
        return archived

    def get_context_emb(self, state, img_list):
        prompt = state.get_prompt()
        print(prompt)
        prompt_segs = prompt.split('<Img><ImageHere></Img>')

        assert len(prompt_segs) == len(
            img_list
        ) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llm_model.internlm_tokenizer(seg,
                                              return_tensors="pt",
                                              add_special_tokens=i == 0).to(
                                                  self.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [
            self.llm_model.internlm_model.model.embed_tokens(seg_t)
            for seg_t in seg_tokens
        ]
        mixed_embs = [
            emb for pair in zip(seg_embs[:-1], img_list) for emb in pair
        ] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def chat_ask(self, state, img_list, text, image):
        print(1111)
        state.skip_next = False
        if len(text) <= 0 and image is None:
            state.skip_next = True
            return (state, img_list, state.to_gradio_chatbot(), "",
                    None) + (no_change_btn, ) * 2

        if image is not None:
            image_pt = self.vis_processor(image).unsqueeze(0).to(0)
            image_emb = self.llm_model.encode_img(image_pt)
            img_list.append(image_emb)

            state.append_message(state.roles[0],
                                 ["<Img><ImageHere></Img>", image])

        if len(state.messages) > 0 and state.messages[-1][0] == state.roles[
                0] and isinstance(state.messages[-1][1], list):
            #state.messages[-1][1] = ' '.join([state.messages[-1][1], text])
            state.messages[-1][1][0] = ' '.join(
                [state.messages[-1][1][0], text])
        else:
            state.append_message(state.roles[0], text)

        print(state.messages)

        state.append_message(state.roles[1], None)

        return (state, img_list, state.to_gradio_chatbot(), "",
                None) + (disable_btn, ) * 2

    def generate_with_callback(self, callback=None, **kwargs):
        kwargs.setdefault("stopping_criteria",
                          transformers.StoppingCriteriaList())
        kwargs["stopping_criteria"].append(Stream(callback_func=callback))
        with torch.no_grad():
            with self.llm_model.maybe_autocast():
                self.llm_model.internlm_model.generate(**kwargs)

    def generate_with_streaming(self, **kwargs):
        return Iteratorize(self.generate_with_callback, kwargs, callback=None)

    def chat_answer(self, state, img_list, max_output_tokens,
                    repetition_penalty, num_beams, do_sample):
        # text = '图片中是一幅油画，描绘了红军长征的场景。画面中，一群红军战士正在穿过一片草地，他们身后的旗帜在风中飘扬。'
        # for i in range(len(text)):
        #     state.messages[-1][-1] = text[:i+1] + "▌"
        #     yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2
        # state.messages[-1][-1] = text[:i + 1]
        # yield (state, state.to_gradio_chatbot()) + (enable_btn, ) * 2
        # return

        if state.skip_next:
            return (state, state.to_gradio_chatbot()) + (no_change_btn, ) * 2

        embs = self.get_context_emb(state, img_list)
        if chat_stream_output:
            generate_params = dict(
                inputs_embeds=embs,
                num_beams=num_beams,
                do_sample=do_sample,
                stopping_criteria=self.stopping_criteria,
                repetition_penalty=float(repetition_penalty),
                max_length=max_output_tokens,
                bos_token_id=self.llm_model.internlm_tokenizer.bos_token_id,
                eos_token_id=self.llm_model.internlm_tokenizer.eos_token_id,
                pad_token_id=self.llm_model.internlm_tokenizer.pad_token_id,
            )
            state.messages[-1][-1] = "▌"
            with self.generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    decoded_output = self.llm_model.internlm_tokenizer.decode(
                        output[1:])
                    if output[-1] in [
                            self.llm_model.internlm_tokenizer.eos_token_id
                    ]:
                        break
                    state.messages[-1][-1] = decoded_output + "▌"
                    yield (state,
                           state.to_gradio_chatbot()) + (disable_btn, ) * 2
                    time.sleep(0.03)
            state.messages[-1][-1] = state.messages[-1][-1][:-1]
            yield (state, state.to_gradio_chatbot()) + (enable_btn, ) * 2
            return
        else:
            outputs = self.llm_model.internlm_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_output_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                #temperature=float(temperature),
                do_sample=do_sample,
                repetition_penalty=float(repetition_penalty),
                bos_token_id=self.llm_model.internlm_tokenizer.bos_token_id,
                eos_token_id=self.llm_model.internlm_tokenizer.eos_token_id,
                pad_token_id=self.llm_model.internlm_tokenizer.pad_token_id,
            )

            output_token = outputs[0]
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_text = self.llm_model.internlm_tokenizer.decode(
                output_token, add_special_tokens=False)
            print(output_text)
            output_text = output_text.split('<TOKENS_UNUSED_1>')[
                0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_text = output_text.replace("<s>", "")
            state.messages[-1][1] = output_text

            return (state, state.to_gradio_chatbot()) + (enable_btn, ) * 2

    def clear_answer(self, state):
        state.messages[-1][-1] = None
        return (state, state.to_gradio_chatbot())

    def chat_clear_history(self):
        state = CONV_VISION_7132_v2.copy()
        return (state, [], state.to_gradio_chatbot(), "",
                None) + (disable_btn, ) * 2


def load_demo():
    state = CONV_VISION_7132_v2.copy()

    return (state, [], gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True), gr.Button.update(visible=True),
            gr.Row.update(visible=True), gr.Accordion.update(visible=True))


def change_language(lang):
    if lang == '中文':
        lang_btn = gr.update(value='English')
        title = gr.update(label='根据给定标题写一个图文并茂的文章：')
        btn = gr.update(value='生成')
        parameter_article = gr.update(label='高级设置')

        beam = gr.update(label='集束大小')
        repetition = gr.update(label='重复惩罚')
        text_num = gr.update(label='最多输出字数')
        msi = gr.update(label='模型选图')
        random = gr.update(label='采样')
        img_num = gr.update(label='生成文章后，可选择全文配图数量')
        adjust_btn = gr.update(value='固定数量配图')
        cap_searchs, editers = [], []
        for _ in range(max_section):
            cap_searchs.append(gr.update(value='搜索'))
            editers.append(gr.update(label='编辑'))

        save_btn = gr.update(value='文章下载')
        save_file = gr.update(label='文章下载')

        parameter_chat = gr.update(label='参数')
        chat_text_num = gr.update(label='最多输出字数')
        chat_beam = gr.update(label='集束大小')
        chat_repetition = gr.update(label='重复惩罚')
        chat_random = gr.update(label='采样')

        chat_textbox = gr.update(placeholder='输入聊天内容并回车')
        submit_btn = gr.update(value='提交')
        regenerate_btn = gr.update(value='🔄  重新生成')
        clear_btn = gr.update(value='🗑️  清空聊天框')
    elif lang == 'English':
        lang_btn = gr.update(value='中文')
        title = gr.update(
            label='Write an illustrated article based on the given title:')
        btn = gr.update(value='Submit')
        parameter_article = gr.update(label='Advanced Settings')

        beam = gr.update(label='Beam Size')
        repetition = gr.update(label='Repetition_penalty')
        text_num = gr.update(label='Max output tokens')
        msi = gr.update(label='Model selects images')
        random = gr.update(label='Do_sample')
        img_num = gr.update(
            label=
            'Select the number of the inserted image after article generation.'
        )
        adjust_btn = gr.update(value='Insert a fixed number of images')
        cap_searchs, editers = [], []
        for _ in range(max_section):
            cap_searchs.append(gr.update(value='Search'))
            editers.append(gr.update(label='edit'))

        save_btn = gr.update(value='Save article')
        save_file = gr.update(label='Save article')

        parameter_chat = gr.update(label='Parameters')
        chat_text_num = gr.update(label='Max output tokens')
        chat_beam = gr.update(label='Beam Size')
        chat_repetition = gr.update(label='Repetition_penalty')
        chat_random = gr.update(label='Do_sample')

        chat_textbox = gr.update(placeholder='Enter text and press ENTER')
        submit_btn = gr.update(value='Submit')
        regenerate_btn = gr.update(value='🔄  Regenerate')
        clear_btn = gr.update(value='🗑️  Clear history')

    return [lang_btn, title, btn, parameter_article, beam, repetition, text_num, msi, random, img_num, adjust_btn] +\
           cap_searchs + editers + [save_btn, save_file] +[parameter_chat, chat_text_num, chat_beam, chat_repetition, chat_random] + \
           [chat_textbox, submit_btn, regenerate_btn, clear_btn]


demo_ui = Demo_UI()

with gr.Blocks(css=custom_css, title='浦语·灵笔 (InternLM-XComposer)') as demo:
    with gr.Row():
        with gr.Column(scale=20):
            #gr.HTML("""<h1 align="center" id="space-title" style="font-size:35px;">🤗 浦语·灵笔 (InternLM-XComposer)</h1>""")
            gr.HTML(
                """<h1 align="center"><img src="https://raw.githubusercontent.com/panzhang0212/interleaved_io/main/logo.png", alt="InternLM-XComposer" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>"""
            )
        with gr.Column(scale=1, min_width=100):
            lang_btn = gr.Button("中文")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("📝 创作图文并茂文章 (Write Interleaved-text-image Article)"):
            with gr.Row():
                title = gr.Textbox(
                    label=
                    'Write an illustrated article based on the given title:',
                    scale=2)
                btn = gr.Button("Submit", scale=1)

            with gr.Row():
                img_num = gr.Slider(
                    minimum=1.0,
                    maximum=30.0,
                    value=5.0,
                    step=1.0,
                    scale=2,
                    label=
                    'Select the number of the inserted image after article generation.'
                )
                adjust_btn = gr.Button('Insert a fixed number of images',
                                       interactive=False,
                                       scale=1)

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("Advanced Settings",
                                      open=False,
                                      visible=True) as parameter_article:
                        beam = gr.Slider(minimum=1.0,
                                         maximum=6.0,
                                         value=5.0,
                                         step=1.0,
                                         label='Beam Size')
                        repetition = gr.Slider(minimum=0.0,
                                               maximum=10.0,
                                               value=5.0,
                                               step=0.1,
                                               label='Repetition_penalty')
                        text_num = gr.Slider(minimum=100.0,
                                             maximum=2000.0,
                                             value=1000.0,
                                             step=1.0,
                                             label='Max output tokens')
                        msi = gr.Checkbox(value=True, label='Model selects images')
                        random = gr.Checkbox(label='Do_sample')

                with gr.Column(scale=1):
                    gr.Examples(
                        examples=[["我的海南自驾游"],
                                  ["My Hainan self-driving tour"]],
                        inputs=[title],
                    )

            articles = []
            gallerys = []
            add_delete_btns = []
            cap_textboxs = []
            cap_searchs = []
            editers = []
            with gr.Column():
                for i in range(max_section):
                    with gr.Row():
                        visible = True if i == 0 else False
                        with gr.Column(scale=2):
                            article = gr.Markdown(visible=visible,
                                                  elem_classes='feedback')
                            articles.append(article)

                        with gr.Column(scale=1):
                            with gr.Accordion('edit',
                                              open=False,
                                              visible=False) as editer:
                                with gr.Row():
                                    cap_textbox = gr.Textbox(show_label=False,
                                                             interactive=True,
                                                             scale=6,
                                                             visible=False)
                                    cap_search = gr.Button(value="Search",
                                                           visible=False,
                                                           scale=1)
                                with gr.Row():
                                    gallery = gr.Gallery(visible=False,
                                                         columns=2,
                                                         height='auto')

                                add_delete_btn = gr.Button(visible=False)

                            gallery.select(demo_ui.replace_image, [
                                articles[i],
                                gr.Number(value=i, visible=False)
                            ], articles[i])
                            gallerys.append(gallery)
                            add_delete_btns.append(add_delete_btn)

                            cap_textboxs.append(cap_textbox)
                            cap_searchs.append(cap_search)
                            editers.append(editer)

                save_btn = gr.Button("Save article")
                save_file = gr.File(label="Save article")

                for i in range(max_section):
                    add_delete_btns[i].click(demo_ui.add_delete_image,
                                             inputs=[
                                                 articles[i],
                                                 add_delete_btns[i],
                                                 gr.Number(value=i,
                                                           visible=False)
                                             ],
                                             outputs=[
                                                 articles[i], gallerys[i],
                                                 add_delete_btns[i],
                                                 cap_textboxs[i],
                                                 cap_searchs[i]
                                             ])
                    cap_searchs[i].click(demo_ui.search_image,
                                         inputs=[
                                             cap_textboxs[i],
                                             gr.Number(value=i, visible=False)
                                         ],
                                         outputs=gallerys[i])

                btn.click(
                    demo_ui.generate_article,
                    inputs=[title, beam, repetition, text_num, msi, random],
                    outputs=articles + gallerys + add_delete_btns +
                    cap_textboxs + cap_searchs + editers + [btn, adjust_btn])
                # cap_btn.click(demo_ui.add_delete_caption, inputs=None, outputs=articles)
                save_btn.click(demo_ui.save, inputs=None, outputs=save_file)
                adjust_btn.click(demo_ui.adjust_img,
                                 inputs=img_num,
                                 outputs=articles + gallerys +
                                 add_delete_btns + cap_textboxs + cap_searchs +
                                 editers)

        with gr.TabItem("💬 多模态对话 (Multimodal Chat)", elem_id="chat", id=0):
            chat_state = gr.State()
            img_list = gr.State()
            with gr.Row():
                with gr.Column(scale=3):
                    imagebox = gr.Image(type="pil")

                    with gr.Accordion("Parameters", open=True,
                                      visible=False) as parameter_row:
                        chat_max_output_tokens = gr.Slider(
                            minimum=0,
                            maximum=1024,
                            value=512,
                            step=64,
                            interactive=True,
                            label="Max output tokens",
                        )
                        chat_num_beams = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            interactive=True,
                            label="Beam Size",
                        )
                        chat_repetition_penalty = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=0.1,
                            interactive=True,
                            label="Repetition_penalty",
                        )
                        # chat_temperature = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, interactive=True,
                        #                         label="Temperature", )
                        chat_do_sample = gr.Checkbox(interactive=True,
                                                     value=True,
                                                     label="Do_sample")
                with gr.Column(scale=6):
                    chatbot = grChatbot(elem_id="chatbot",
                                        visible=False,
                                        height=750)
                    with gr.Row():
                        with gr.Column(scale=8):
                            chat_textbox = gr.Textbox(
                                show_label=False,
                                placeholder="Enter text and press ENTER",
                                visible=False).style(container=False)
                        with gr.Column(scale=1, min_width=60):
                            submit_btn = gr.Button(value="Submit",
                                                   visible=False)
                    with gr.Row(visible=True) as button_row:
                        regenerate_btn = gr.Button(value="🔄  Regenerate",
                                                   interactive=False)
                        clear_btn = gr.Button(value="🗑️  Clear history",
                                              interactive=False)

            btn_list = [regenerate_btn, clear_btn]
            parameter_list = [
                chat_max_output_tokens, chat_repetition_penalty,
                chat_num_beams, chat_do_sample
            ]

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

    lang_btn.click(change_language, inputs=lang_btn, outputs=[lang_btn, title, btn, parameter_article] +\
                                [beam, repetition, text_num, msi, random, img_num, adjust_btn] + cap_searchs + editers +\
                                [save_btn, save_file] + [parameter_row, chat_max_output_tokens, chat_num_beams, chat_repetition_penalty, chat_do_sample] +\
                                [chat_textbox, submit_btn, regenerate_btn, clear_btn])
    demo.queue(concurrency_count=8, status_update_rate=10, api_open=False)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=11111)
