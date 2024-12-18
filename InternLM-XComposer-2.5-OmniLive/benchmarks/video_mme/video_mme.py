import os
import math
import json
import time
import argparse
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from decord import VideoReader
from tqdm import tqdm
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

import torchvision.transforms as transforms


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


def split_list(lst, n, balance_w):
    """Split a list into n (roughly) equal-sized chunks"""
    avg_chunk_size = math.ceil(len(lst) / n)  # integer division
    chunk = []
    start = 0
    for i in range(n):
        end = int(start + avg_chunk_size * balance_w[i]) if i < n-1 else len(lst)
        chunk.append(lst[start:end])
        start = end
    return chunk


def get_chunk(lst, n, k, balance_w):
    assert n == len(balance_w)
    chunks = split_list(lst, n, balance_w)
    print([len(it) for it in chunks])
    return chunks[k]


def eval_dataset(args):
    disable_torch_init()
    ixc_tokenizer = AutoTokenizer.from_pretrained(args.ixc_model_path, trust_remote_code=True)
    ixc_model = AutoModelForCausalLM.from_pretrained(args.ixc_model_path, device_map="cuda", trust_remote_code=True).eval().cuda().to(torch.bfloat16)
    ixc_model.tokenizer = ixc_tokenizer

    balance_w = [1] * args.num_chunks
    if args.task == 'all':
        tasks = ['short', 'medium', 'long']
        if args.num_chunks == 8:
            balance_w = [1.6, 1.6, 1.4, 1, 0.9, 0.5, 0.5, 0.5]  # balance for gpu time
    else:
        tasks = [args.task]

    data = pd.read_parquet("benchmarks/video_mme/test-00000-of-00001.parquet", engine='pyarrow')
    samples = []
    for i in range(len(data)):
        task = data.loc[i, 'duration']
        if task in tasks:
            sample = {}
            sample['task'] = task
            sample['path'] = data.loc[i, 'videoID']
            sample['question'] = data.loc[i, 'question']
            sample['options'] = data.loc[i, 'options']
            sample['answer'] = data.loc[i, 'answer']
            samples.append(sample)
    print(len(samples))

    out = {}
    for task in tasks:
        out[task] = []

    samples = get_chunk(samples, args.num_chunks, args.chunk_idx, balance_w)
    start_time = time.time()
    for sample in tqdm(samples):
        task = sample['task']
        path = sample['path']
        video_path = f'{args.video_folder}/{path}.mp4'

        try:
            frames = load_video(video_path, num_frm=16, max_clip=32)
            sele_frames = frames

            question = 'Here are some frames of a video. ' + sample['question']
            ans = sample['answer']
            options = sample['options']
            options_prompt = ''
            for item in options:
                idx = item[0]
                cnt = item[3:]
                options_prompt += f'{idx}. {cnt}\n'

            if len(sele_frames) > args.max_frame:
                step = (len(sele_frames) - 1) / (args.max_frame - 1)
                sele_frames = [sele_frames[int(i*step)] for i in range(args.max_frame)]

            img = img_process(sele_frames)

            mid_prompt = 'Question: ' + question + '\nOptions: ' + options_prompt
            query = f'[UNUSED_TOKEN_146]user\n{mid_prompt}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    response = model_gen(ixc_model, query, img, hd_num=36, do_sample=False, beam=1)

            out[task].append([path, response[0] == ans])
            #print(np.mean(out[task]), response, ans)
        except:
            continue

    os.makedirs(args.save_folder, exist_ok=True)
    json.dump(out, open(os.path.join(args.save_folder, f'{args.chunk_idx}_of_{args.num_chunks}.json'), 'w'))
    print(f"Rank {args.chunk_idx} use {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ixc-model-path", type=str, default="internlm-xcomposer2d5-ol-7b/base")
    parser.add_argument("--video-folder", type=str)
    parser.add_argument("--save-folder", type=str, default="outputs/video_mme")
    parser.add_argument("--max-frame", type=int, default=64)
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()
    print(args)
    eval_dataset(args)