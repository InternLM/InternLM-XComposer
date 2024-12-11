import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from grounding_qwen import GroundQwenForCausalLM
from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from conversation import conv_templates, SeparatorStyle
from utils import disable_torch_init
from mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import pickle
from decord import VideoReader
import numpy as np

from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer
from petrel_client.client import Client
client = Client('~/petreloss.conf')


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
    key_frame = [[frame_idx[i*frm_per_clip], frame_idx[i*frm_per_clip+frm_per_clip-1]] for i in range(num_clip)]
    time = vr.get_frame_timestamp(key_frame)
    return np.hstack([time[:, 0, 0], time[:, 1, 1]])


def calculate_diff(scene_sep, start_frame):
    diff = [scene_sep[0]-start_frame]
    for i in range(len(scene_sep)-1):
        diff.append(scene_sep[i+1]-scene_sep[i])
    return diff


def load_video(vis_path, scene_sep, num_frm=16, max_clip=4):
    block_size = 1
    vr = VideoReader(vis_path)
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    total_time = total_frame_num / fps

    if len(scene_sep) == 0:
        num_clip = total_time / num_frm
        num_clip = int(block_size*np.round(num_clip/block_size)) if num_clip > block_size else int(np.round(num_clip))
        num_clip = max(num_clip, 5)
        num_clip = min(num_clip, max_clip)
        total_num_frm = num_frm * num_clip
        start_frame = 0
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    else:
        num_clip = max(len(scene_sep), 5)
        num_clip = min(num_clip, max_clip)
        total_num_frm = num_frm * num_clip
        start_frame = 0
        frame_idx = []
        if len(scene_sep) < 5:
            diff = calculate_diff(scene_sep, start_frame)
            new_sep = max(diff) / (5-len(scene_sep)+1)
            max_idx = np.argmax(diff)
            if max_idx == 0:
                scene_sep = [int(start_frame+new_sep*i) for i in range(1, 5-len(scene_sep)+1)] + scene_sep
            else:
                scene_sep = scene_sep[:max_idx]+[int(scene_sep[max_idx-1]+i*new_sep) for i in range(1, 5-len(scene_sep)+1)] + scene_sep[max_idx:]
        elif len(scene_sep) > max_clip:
            diff = calculate_diff(scene_sep, start_frame)
            min_idx = np.argsort(diff[:-1])[:len(scene_sep)-max_clip] ##minimum diff to remove
            for i in np.sort(min_idx)[::-1]:
                del scene_sep[i]

        start_ = start_frame
        for end_frame in scene_sep:
            idx_list = np.linspace(start_, end_frame, num=num_frm, endpoint=False)
            frame_idx.extend([int(id) for id in idx_list])
            start_ = end_frame

        time_idx = get_seq_time(vr, frame_idx, num_clip)
        img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs, time_idx, num_clip


def preprocess_time(time, num_clip, tokenizer):
    time = time.reshape(2, num_clip)
    seq = []

    block_size = 1
    for i in range(num_clip):
        start, end = time[:, i]
        start = int(np.round(start))
        end = int(np.round(end))
        if (i+1) % block_size == 0:
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



def eval_dataset(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    device = 'cuda'
    kwargs = {"device_map": 'auto'}
    kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = GroundQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    video_path = '/mnt/hwfile/mllm/qianrui/test_video/air.mp4'
    question = ['How many pilots are shown in the video?', 'What airlines are shown in the video?', 'How is the decoration of the airport?']
    scene_sep = json.load(open('/mnt/hwfile/mllm/qianrui/test_video/air.json', 'r'))['scene_sep']
    frames, time_idx, num_clips = load_video(video_path, scene_sep, num_frm=16, max_clip=32)
    video = image_processor.preprocess(frames, return_tensors='pt')['pixel_values']
    video = video.view(num_clips, 16, *video.shape[1:])
    seqs = preprocess_time(time_idx, num_clips, tokenizer)
    seqs = torch.nn.utils.rnn.pad_sequence(
        seqs, 
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    compress_mask = seqs.ne(tokenizer.pad_token_id)
    question = preprocess_question(question, tokenizer)
    question = torch.nn.utils.rnn.pad_sequence(
        question, 
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    qs_mask = question.ne(tokenizer.pad_token_id)

    with torch.inference_mode():
        similarity = model.forward_grounding(
            input_ids=seqs.to(device='cuda', non_blocking=True),
            attention_mask=compress_mask.to(device='cuda', non_blocking=True),
            images=video.to(dtype=torch.float16, device='cuda', non_blocking=True),
            qs_ids=question.to(device='cuda', non_blocking=True),
            qs_mask=qs_mask.to(device='cuda', non_blocking=True))
    
    print(similarity.shape, similarity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    args = parser.parse_args()

    eval_dataset(args)
