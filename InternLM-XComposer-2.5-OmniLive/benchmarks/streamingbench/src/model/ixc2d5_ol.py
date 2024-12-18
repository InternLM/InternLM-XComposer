import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from decord import VideoReader
from transformers import AutoTokenizer, AutoModelForCausalLM

import torchvision.transforms as transforms
from model.modelclass import Model


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def padding_336(b, pad=336):
    width, height = b.size
    tar = int(np.ceil(height / pad) * pad)
    top_padding = 0  # int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255])

    return b


def Identity_transform(img, hd_num=25):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    new_h = int(scale * 560)
    new_w = int(new_h * ratio)
    #print(new_h, new_w)

    img = transforms.functional.resize(img, [new_h, new_w], )
    img = img.transpose(Image.TRANSPOSE)
    img = padding_336(img, 560)
    width, height = img.size
    if not trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


def model_gen(model, text, images, need_bos=True, hd_num=36, max_new_token=2, beam=3, do_sample=False):
    pt1 = 0
    embeds = []
    im_mask = []
    if images is None:
        images = []
        images_loc = []
    else:
        images = [images]
        images_loc = [len('[UNUSED_TOKEN_146]user\n')]
    for i, pts in enumerate(images_loc + [len(text)]):
        subtext = text[pt1:pts]
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            need_bos = False
        if i < len(images):
            try:
                image = Image.open(images[i]).convert('RGB')
            except:
                image = images[i].convert('RGB')

            image = Identity_transform(image, hd_num=hd_num)
            image = model.vis_processor(image).unsqueeze(0).cuda()
            image_embeds = model.encode_img(image)
            #print(image_embeds.shape)
            embeds.append(image_embeds)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                             temperature=1.0, max_new_tokens=max_new_token, num_beams=beam,
                             do_sample=False, repetition_penalty=1.00)

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
    return output_text


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
        font = ImageFont.truetype("benchmarks/SimHei.ttf", pad)
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
        font = ImageFont.truetype("benchmarks/SimHei.ttf", pad)
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


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def load_video(vis_path, num_frm=16, max_clip=4):
    block_size = 1
    vr = VideoReader(vis_path)
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    total_time = total_frame_num / fps

    num_clip = total_time / num_frm
    num_clip = int(block_size * np.round(num_clip / block_size)) if num_clip > block_size else int(np.round(num_clip))
    num_clip = max(num_clip, 5)
    num_clip = min(num_clip, max_clip)
    total_num_frm = num_frm * num_clip
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)

    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


class IXC2d5_OL(Model):
    def __init__(self, ixc_model_path, max_frame=64):
        """
        Initialize the model
        """
        super().__init__()
        disable_torch_init()

        self.max_frame = max_frame

        ixc_tokenizer = AutoTokenizer.from_pretrained(ixc_model_path, trust_remote_code=True)
        self.ixc_model = AutoModelForCausalLM.from_pretrained(ixc_model_path, device_map="cuda",
                                                         trust_remote_code=True).eval().cuda().to(torch.bfloat16)
        self.ixc_model.tokenizer = ixc_tokenizer

    def Run(self, file, inp, ques, timestamp):
        """
        Given the video file and input prompt, run the model and return the response
        file: Video file path
        inp: Input prompt
        timestampe: The time when question is asked
        """
        try:
            frames = load_video(file, num_frm=16, max_clip=32)
            sele_frames = frames

            if len(sele_frames) > self.max_frame:
                step = (len(sele_frames) - 1) / (self.max_frame - 1)
                sele_frames = [sele_frames[int(i * step)] for i in range(self.max_frame)]

            img = img_process(sele_frames)
            query = inp
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    response = model_gen(self.ixc_model, query, img, hd_num=36, do_sample=False, beam=1)
            return response[0]
        except:
            print("inference error!!!, chose A.")
            return "A"

    @staticmethod
    def name():
        """
        Return the name of the model
        """
        return "IXC2d5_OL"
