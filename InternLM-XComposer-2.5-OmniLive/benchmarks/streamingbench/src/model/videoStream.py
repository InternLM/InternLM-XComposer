import os
import transformers
import torch
import sys
sys.path.append('/mnt/petrelfs/dingshuangrui/LLaVA')
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_ANS_TOKEN, DEFAULT_TODO_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
# from data.utils import ffmpeg_once
import pdb
from PIL import Image
import math
import pickle
from decord import VideoReader
import numpy as np

from transformers import StoppingCriteria, StoppingCriteriaList
from model.modelclass import Model

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False



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


def load_video(vis_path, scene_sep, num_frm=16, max_clip=4, sample_frame=None):
    block_size = 1
    vr = VideoReader(vis_path, num_threads=1)
    total_frame_num = len(vr) if sample_frame is None else (sample_frame[0][1]-sample_frame[0][0])
    fps = vr.get_avg_fps()
    total_time = total_frame_num / fps

    if len(scene_sep) == 0:
        num_clip = total_time / num_frm
        num_clip = int(block_size*np.round(num_clip/block_size)) if num_clip > block_size else int(np.round(num_clip))
        num_clip = max(num_clip, 1) ### default 5
        num_clip = min(num_clip, max_clip)
        total_num_frm = num_frm * num_clip
        start_frame = 0 if sample_frame is None else sample_frame[0][0]
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    else:
        end_frame = total_frame_num if sample_frame is None else sample_frame[0][1]
        new_scene_sep = []
        for ele in scene_sep:
            sep = int(fps*(ele+1))
            sep = min(sep, end_frame-1)
            new_scene_sep.append(sep)
        new_scene_sep += [end_frame-1]
        scene_sep = new_scene_sep
        if len(scene_sep) > max_clip:
            diff = calculate_diff(scene_sep, start_frame=0)
            min_idx = np.argsort(diff[:-1])[:len(scene_sep)-max_clip] ##minimum diff to remove
            for i in np.sort(min_idx)[::-1]:
                del scene_sep[i]        
        start_ = 0
        for end_frame in scene_sep:
            idx_list = np.linspace(start_, end_frame, num=num_frm, endpoint=False)
            frame_idx.extend([int(id) for id in idx_list])
            start_ = end_frame

    time_idx = get_seq_time(vr, frame_idx, num_clip)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    # a, H, W, _ = img_array.shape
    # h, w = 224, 224
    # if img_array.shape[-3] != h or img_array.shape[-2] != w:
    #     img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
    #     img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
    #     img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    a, H, W, _ = img_array.shape
    if H != W:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(min(H, W), min(H, W)))
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
        # if i // block_size == 0:
        #     sentence = 'This contains a clip sampled in %d to %d seconds' % (start, end) + DEFAULT_IMAGE_TOKEN
        # else:
        #     sentence = 'This contains a history of %d to %d seconds, and a clip sampled in %d to %d seconds' % (0, history_end, start, end) + DEFAULT_IMAGE_TOKEN
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq


def preprocess_question(questions, tokenizer):
    seq = []
    for q in questions:
        sentence = tokenizer_image_token(q+DEFAULT_TODO_TOKEN, tokenizer, return_tensors='pt')
        seq.append(sentence)
    
    return seq


def process_data(video_id, scene_sep, question, model_config, tokenizer, processor, processor_large, time_tokenizer):
    num_frames = 16
    num_clips = 100
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = video_id
    presigned_url = client.generate_presigned_url(image, client_method ='get_object', expires_in=3600) if 's3://' in image else image
    frames, time_idx, num_clips = load_video(presigned_url, scene_sep, num_frames, num_clips)
    video = processor.preprocess(frames, return_tensors='pt')['pixel_values']
    video = video.view(num_clips, num_frames, *video.shape[1:])
    video_large = processor_large.preprocess(frames, return_tensors='pt')['pixel_values']
    video_large = video_large.view(num_clips, num_frames, *video_large.shape[1:])[:, :1].contiguous()
    seqs = preprocess_time(time_idx, num_clips, time_tokenizer)
    seqs = torch.nn.utils.rnn.pad_sequence(
        seqs, 
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id)
    compress_mask = seqs.ne(time_tokenizer.pad_token_id)
    question = preprocess_question([question], time_tokenizer)
    question = torch.nn.utils.rnn.pad_sequence(
        question, 
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id)
    qs_mask = question.ne(time_tokenizer.pad_token_id)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    return input_ids, video, video_large, seqs, compress_mask, question, qs_mask


class videoStream(Model):
    def __init__(self):
        """
        Initialize the model
        """
        super().__init__()
        model_path = os.path.expanduser('/mnt/hwfile/mllm/qianrui/checkpoints/long-qwen-16f-64r-4p-16-causal-qwen-streaming-224-sft/checkpoint-7000')
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)

        self.image_processor, self.time_tokenizer = image_processor
        self.image_processor_large = self.image_processor
        if self.time_tokenizer.pad_token is None:
            self.time_tokenizer.pad_token = '<pad>'


        stop_words_ids = [
            # torch.tensor(tokenizer('</s>').input_ids).cuda(),
            # torch.tensor(tokenizer('<|end_of_text|>').input_ids).cuda(),
            # torch.tensor(tokenizer('<|eot_id|>').input_ids).cuda(),
            torch.tensor(self.tokenizer('<|im_end|>').input_ids).cuda(),
        ]

        self.stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)])



    def Run(self, file, inp, timestamp):
        """
        Given the video file and input prompt, run the model and return the response
        file: Video file path
        inp: Input prompt
        timestampe: The time when question is asked
        """
        # timestamp = float(file.split('/')[-1].rsplit('.', 1)[0].split('_')[-1])
        # pdb.set_trace()
        try:
            input_ids, image_tensor, image_tensor_large, seqs, compress_mask, qs, qs_mask = process_data(file, 
                                    [], 
                                    inp, 
                                    self.model.config, 
                                    self.tokenizer, 
                                    self.image_processor, 
                                    self.image_processor_large, 
                                    self.time_tokenizer,
                                    )
            input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)
            # print(image_tensor.shape)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    images_large=image_tensor_large.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    seqs=seqs.to(device='cuda', non_blocking=True),
                    compress_mask=compress_mask.to(device='cuda', non_blocking=True),
                    qs=qs.to(device='cuda', non_blocking=True),
                    qs_mask=qs_mask.to(device='cuda', non_blocking=True),
                    ans_token=self.time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt").input_ids.to(device='cuda', non_blocking=True),
                    todo_token=self.time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt").input_ids.to(device='cuda', non_blocking=True),
                    q_id=None,
                    insert_position=0,
                    ans_position=[],
                    do_sample=False,
                    max_new_tokens=1024,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=self.stopping_criteria,
                    use_cache=True)
        except:
            return "A"
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        return outputs

    @staticmethod
    def name():
        """
        Return the name of the model
        """
        return "videoStream"
