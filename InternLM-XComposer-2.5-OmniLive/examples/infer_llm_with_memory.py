import sys
sys.path.append('internlm-xcomposer2d5-ol-7b')
sys.path.append('internlm-xcomposer2d5-ol-7b/memory')

import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from decord import VideoReader
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModelForCausalLM

from base.ixc_utils import Video_transform
from base.modeling_internlm_xcomposer2 import get_stopping_criteria
from memory.constants import DEFAULT_IMAGE_PATCH_TOKEN
from memory.grounding_qwen import GroundQwenForCausalLM
from memory.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from memory.mm_utils import tokenizer_image_token


def model_gen_withmem(model, text, images, glb, lol, need_bos=True, hd_num=36, max_new_token=2, beam=3, do_sample=False):
    temp_emb = []

    _, c = glb.shape
    glb = model.video_mem_proj(glb.view(1, -1, c))
    glb_text = model.encode_text('This is video overview memory:', add_special_tokens=False)
    temp_emb.append(glb_text)
    temp_emb.append(glb)

    if len(lol) > 0:
        _, c = lol.shape
        lol = model.video_mem_proj(lol.view(1, -1, c))
        lol_text = model.encode_text('This is question related video memory:', add_special_tokens=False)
        temp_emb.append(lol_text)
        temp_emb.append(lol)

    image = Video_transform(images, hd_num=hd_num)
    image_embeds = model.vis_processor(image).unsqueeze(0).cuda()
    image_embeds = model.encode_img(image_embeds)
    temp_emb.append(image_embeds)
    temp_emb = torch.cat(temp_emb, dim=1)
    images = temp_emb

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
            image_embeds = images[i]
            embeds.append(image_embeds)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    stop_words_ids = [92542]
    stopping_criteria = get_stopping_criteria(stop_words_ids)

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                             temperature=1.0, max_new_tokens=max_new_token, num_beams=beam,
                             do_sample=False, repetition_penalty=1.00, stopping_criteria=stopping_criteria)

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
    output_text = output_text.split('<|im_end|>')[0].strip()
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
        font = ImageFont.truetype("internlm-xcomposer2d5-ol-7b/base/SimHei.ttf", pad)
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
        font = ImageFont.truetype("internlm-xcomposer2d5-ol-7b/base/SimHei.ttf", pad)
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


def preprocess_time(time, num_clip, tokenizer):
    time = time.reshape(2, num_clip)
    seq = []

    block_size = 1
    for i in range(num_clip):
        start, end = time[:, i]
        start = int(np.round(start))
        end = int(np.round(end))
        if (i + 1) % block_size == 0:
            history_end = end
        sentence = 'This contains a clip sampled in %d to %d seconds' % (start, end) + DEFAULT_IMAGE_TOKEN
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq


def preprocess_question(questions, tokenizer):
    seq = []
    for q in questions:
        sentence = tokenizer_image_token(q, tokenizer, return_tensors='pt')
        seq.append(sentence)

    return seq


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_seq_time(vr, frame_idx, num_clip):
    frm_per_clip = len(frame_idx) // num_clip
    key_frame = [[frame_idx[i * frm_per_clip], frame_idx[i * frm_per_clip + frm_per_clip - 1]] for i in range(num_clip)]
    time = vr.get_frame_timestamp(key_frame)
    return np.hstack([time[:, 0, 0], time[:, 1, 1]])


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

    time_idx = get_seq_time(vr, frame_idx, num_clip)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs, time_idx, num_clip


def inference(args):
    ixc_tokenizer = AutoTokenizer.from_pretrained(f'{args.ixc_model_path}/merge_lora', trust_remote_code=True)
    ixc_model = AutoModelForCausalLM.from_pretrained(f'{args.ixc_model_path}/merge_lora', device_map="cuda:0", trust_remote_code=True).eval().cuda().to(torch.bfloat16)
    ixc_model.tokenizer = ixc_tokenizer

    kwargs = {"device_map": 'cuda:0'}
    kwargs['torch_dtype'] = torch.float16
    vs_tokenizer = AutoTokenizer.from_pretrained(f'{args.ixc_model_path}/memory', use_fast=False)
    vs_model = GroundQwenForCausalLM.from_pretrained(f'{args.ixc_model_path}/memory', low_cpu_mem_usage=True, **kwargs)
    mm_use_im_start_end = getattr(vs_model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(vs_model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        vs_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        vs_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    vs_model.resize_token_embeddings(len(vs_tokenizer))

    vision_tower = vs_model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(0, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    video_path = 'examples/videos/needle_32.mp4'
    question_ = 'What does the hand coming out of the computer do?'
    candidates = ['Delivers a product', "Shakes the woman's hand", "Takes the woman's credit card", 'Points at something on the screen']
    frames, time_idx, num_clips = load_video(video_path, num_frm=16, max_clip=32)
    video = image_processor.preprocess(frames, return_tensors='pt')['pixel_values']
    video = video.view(num_clips, 16, *video.shape[1:])
    seqs = preprocess_time(time_idx, num_clips, vs_tokenizer)
    seqs = torch.nn.utils.rnn.pad_sequence(
        seqs,
        batch_first=True,
        padding_value=vs_tokenizer.pad_token_id)
    compress_mask = seqs.ne(vs_tokenizer.pad_token_id)

    question = preprocess_question([question_], vs_tokenizer)
    question = torch.nn.utils.rnn.pad_sequence(
        question,
        batch_first=True,
        padding_value=vs_tokenizer.pad_token_id)
    qs_mask = question.ne(vs_tokenizer.pad_token_id)

    with torch.no_grad():
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                similarity, glb, lol = vs_model.forward_grounding(
                    input_ids=seqs.to(device='cuda', non_blocking=True),
                    attention_mask=compress_mask.to(device='cuda', non_blocking=True),
                    images=video.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    qs_ids=question.to(device='cuda', non_blocking=True),
                    qs_mask=qs_mask.to(device='cuda', non_blocking=True))

    lol = lol.view(-1, 64, 1536)
    sele_frames = []
    sele_lol = []
    for i in range(len(frames) // 16):
        if similarity[0][i] > args.vs_thresh:
            sele_frames.extend(frames[i * 16:(i + 1) * 16])
            sele_lol.append(lol[i])
    if len(sele_frames) == 0:
        print('grounding fail!!!')
        sele_frames = frames

    if len(sele_lol) > 0:
        sele_lol = torch.cat(sele_lol, dim=0)

    question = 'Here are some frames of a video. ' + question_
    options = candidates
    options_prompt = ''
    for idx, item in enumerate(options):
        idx = chr(65 + idx)
        options_prompt += f'{idx}. {item}\n'

    if len(sele_frames) > args.max_frame:
        step = (len(sele_frames) - 1) / (args.max_frame - 1)
        sele_frames = [sele_frames[int(i * step)] for i in range(args.max_frame)]
    img = img_process(sele_frames)

    mid_prompt = 'Question: ' + question
    query = f'[UNUSED_TOKEN_146]user\n{mid_prompt}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            response = model_gen_withmem(ixc_model, query, img, glb, sele_lol, hd_num=36, do_sample=False, beam=1, max_new_token=1024)
        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ixc-model-path", type=str, default="internlm-xcomposer2d5-ol-7b")
    parser.add_argument("--max-frame", type=int, default=32)
    parser.add_argument("--vs-thresh", type=float, default=0.2)
    args = parser.parse_args()
    print(args)
    inference(args)