import os
import sys
import time
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.utils import disable_logging
from dataclasses import dataclass
from itertools import count
from decord import VideoReader
import numpy as np
import torch
from PIL import Image
import warnings
from transformers import AutoConfig, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from queue import Queue

from lmdeploy.pytorch.engine import Engine
from lmdeploy import PytorchEngineConfig
sys.path.append('/app/Long-Grounding-qwen')
from grounding_qwen import GroundQwenForCausalLM

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


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


def preprocess_time(time, num_clip, tokenizer):
    time = time.reshape(2, num_clip)
    seq = []
    block_size = 1
    for i in range(num_clip):
        start, end = time[:, i]
        start = int(np.round(start))
        end = int(np.round(end))
        sentence = 'This contains a clip sampled in %d to %d seconds' % (start, end) + DEFAULT_IMAGE_TOKEN
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq


def get_time_prompt(start, end, tokenizer):
    start = int(np.round(start))
    end = int(np.round(end))
    sentence = 'This contains a clip sampled in %d to %d seconds' % (start, end) + DEFAULT_IMAGE_TOKEN
    sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
    return sentence


def preprocess_question(questions, tokenizer):
    seq = []
    for q in questions:
        sentence = tokenizer_image_token(q, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


try:
    sys.path.insert(0, os.path.dirname(__file__))
    from configs_vs import setup4lmdeploy
except Exception:
    raise

logger = get_logger('demo')

@dataclass
class Session:
    """session."""
    _ids = count(0)

    def __init__(self, hash: str = None):
        self._id: int = next(self._ids)
        self.hash = hash
        self.vr = None
        self.num_frm = None
        self.frame_idx = None
        self.num_clip = None
        self.time_idx = None
        self.full_memory = []
        self.full_time = []
        self.full_img = []
        self.global_memory = []
        self.progress_que = Queue()
        self.current_clip = 0
        self.query_clip = 0
        self.current_frame = 0
        self.previous_segment = 0
        self.select_memory = None
        self.select_global = None
        self.select_img = None

    def init_video(self, filepath: str, num_frm: int, max_clip: int):
        logger.info(
            f'session={self.hash}, id={self._id}, init_video={filepath}')
        vr = VideoReader(filepath)
        total_frame_num = len(vr)
        fps = vr.get_avg_fps()
        total_time = total_frame_num #/ fps
        num_clip = total_time / num_frm
        num_clip = max(num_clip, 4)
        num_clip = min(num_clip, max_clip)
        total_num_frm = num_frm * num_clip
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
        time_idx = get_seq_time(vr, frame_idx, num_clip)
        self.vr = vr
        self.num_frm = num_frm
        self.frame_idx = frame_idx
        self.num_clip = num_clip
        self.time_idx = time_idx
        self.total_num_frm = total_num_frm

    def get_next_clip(self):
        logger.info(f'session={self.hash}, id={self._id}, '
                    f'get_next_clip={self.current_clip}')
        assert self.current_clip < self.num_clip
        left = self.current_clip * self.num_frm
        right = (self.current_clip + 1) * self.num_frm
        current_idx = self.frame_idx[left:right]
        img_array = self.vr.get_batch(current_idx).asnumpy()
        h, w = 224, 224
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        clip_imgs = [Image.fromarray(x) for x in img_array]
        self.current_clip += 1
        return clip_imgs
    
    def get_next_frame(self):
        logger.info(f'session={self.hash}, id={self._id}, '
                    f'get_next_frame={self.current_frame}')
        assert self.current_frame < self.total_num_frm
        # determine whether separate a new scene, currently uniform separate scene
        need_new_seg = True if (self.current_frame-self.previous_segment) >= 16 else False
        if need_new_seg:
            self.previous_segment = self.current_frame
        left = self.previous_segment
        right = self.current_frame + 1
        current_idx = self.frame_idx[left:right]
        img = self.vr.get_batch(current_idx).asnumpy()
        h, w = 224, 224
        img_array = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        clip_imgs = [Image.fromarray(x) for x in img_array]
        start = self.vr.get_frame_timestamp(self.previous_segment)[0]
        end = self.vr.get_frame_timestamp(self.current_frame)[1]
        self.current_frame += 1
        if need_new_seg or len(self.full_img) == 0:
            self.full_img.append(img)
        else:
            self.full_img[-1] = img
        return clip_imgs, start, end, need_new_seg


class Model:
    """streaming encoder."""

    def __init__(self, model_path: str):
        # pass
        self._load_processor(model_path)
        self._load_llava_qwen(model_path)
        torch.cuda.empty_cache()

    def _load_processor(self, model_path: str):
        logger.info('loading processor')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.tokenizer = tokenizer

        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cfg = AutoConfig.from_pretrained(model_path,
                                             trust_remote_code=True)
            model = GroundQwenForCausalLM._from_config(cfg, attn_implementation=cfg._attn_implementation)
            del model.model.layers
            del model.model.embed_tokens
            del model.model.norm
            # del model.lm_head

        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         checkpoint=model_path,
                                         device_map={'': 0},
                                         dtype=torch.float16)
        model = model.eval()
        self.compressor = model
        self.image_processor = model.get_vision_tower(
        ).image_processor

    def _load_llava_qwen(self, model_path: str):
        setup4lmdeploy()
        pt_model = Engine(model_path,
                          engine_config=PytorchEngineConfig(
                              cache_max_entry_count=0.1, thread_safe=True))
        self.pt_model = pt_model

    def forward_video_encoding(self, sess: Session, images: torch.Tensor,
                               seqs: torch.Tensor, need_new_seg: bool = True):
        image_features = images.to(device='cuda',
                                   dtype=torch.float16,
                                   non_blocking=True)
        with torch.inference_mode():
            image_features = self.compressor.extract_images(image_features)
            image_features = self.compressor.mix_spatial_tokens(image_features)
            image_features = image_features.view(
                1, images.shape[0] * image_features.shape[1],
                image_features.shape[2])
            image_features = self.compressor.project_features(image_features)
            image_features = image_features.squeeze()

        model = self.compressor.model
        pool_num = model.mm_projector.pool_num
        resolution = model.mm_projector.resolution + pool_num
        num_slot = (image_features.shape[0] - 1) // resolution * pool_num

        input_ids = seqs.cpu().numpy().tolist(
        )[:-1]  # remove last indicator of -200

        n_feat = image_features.shape[0]
        input_embeddings = [image_features]
        input_embedding_ranges = [[len(input_ids), len(input_ids) + n_feat]]
        input_ids += [0] * n_feat

        current_states = self.pt_model.decode(
            [input_ids],
            input_embeddings=[input_embeddings],
            input_embedding_ranges=[input_embedding_ranges],
        )
        history_mem = current_states[:, -num_slot -
                                     1:-1, :].detach().clone().squeeze(0)
        history_time = current_states[:, -1, :].detach().clone()

        if need_new_seg or len(sess.full_memory) == 0:
            sess.full_memory.append(history_mem) # k c
            sess.full_time.append(history_time) # 1 c
        else:
            sess.full_memory[-1] = history_mem
            sess.full_time[-1] = history_time

        global_embeddings = [torch.cat([torch.cat(sess.full_memory, dim=0), torch.cat(sess.full_time, dim=0)], dim=0)]
        global_ids = [0] * global_embeddings[0].shape[0]
        global_embedding_ranges = [[0, len(global_ids)]]
        current_states = self.pt_model.decode(
            [global_ids],
            input_embeddings=[global_embeddings],
            input_embedding_ranges=[global_embedding_ranges],
        )
        global_memory = current_states[:, -len(sess.full_time):, :].detach().clone().squeeze(0)
        if need_new_seg or len(sess.global_memory) == 0:
            sess.global_memory.append(global_memory) # t c
        else:
            sess.global_memory[-1] = global_memory

    def process(self, sess: Session):
        logger.info(f'session={sess.hash}, id={sess._id}, process start')
        image_processor = self.image_processor

        # clip-wise streaming
        # seqs = preprocess_time(sess.time_idx, sess.num_clip,
        #                        self.tokenizer)
        # for i in range(sess.num_clip):
        #     clips = sess.get_next_clip()
        #     video = image_processor.preprocess(
        #         clips, return_tensors='pt')['pixel_values']
        #     seq = seqs[i]
        #     self.forward_video_encoding(sess=sess, images=video, seqs=seq)
        #     sess.progress_que.put(1)
        
        # frame-wise streaming
        for i in range(sess.total_num_frm):
            start_ = time.time()
            clips, start, end, need_new_seg = sess.get_next_frame()
            video = image_processor.preprocess(
                clips, return_tensors='pt')['pixel_values']
            seq = get_time_prompt(start, end, self.tokenizer)
            self.forward_video_encoding(sess=sess, images=video, seqs=seq, need_new_seg=need_new_seg)
            sess.progress_que.put(1)
            self.select_memory(sess, 'How many pilots are shown in the video?')
            print('Time consuming for %i-th frame:' % i, time.time()-start_)

        logger.info(f'session={sess.hash}, id={sess._id}, process finished')

    def _ensure_feature(self, sess: Session):
        while len(sess.full_time) == 0:
            time.sleep(0.1)

    def forward_question(self, sess: Session, query: str):
        logger.info(f'session={sess.hash}, id={sess._id}, forward_question')
        question = preprocess_question([query], self.tokenizer)[0]

        self._ensure_feature(sess)
        sess.query_clip = len(sess.full_time)
        logger.info(f'session={sess.hash}, id={sess._id}, '
                    f'query_clip={sess.query_clip}')
        memory = sess.global_memory[sess.query_clip - 1]
        input_ids = question.cpu().numpy().tolist()
        n_feat = memory.shape[0]
        input_embeddings = [memory]
        input_embedding_ranges = [[0, n_feat]]
        input_ids = [0] * n_feat + input_ids  # memory at the beginning

        current_states = self.pt_model.decode(
            [input_ids],
            input_embeddings=[input_embeddings],
            input_embedding_ranges=[input_embedding_ranges],
        )
        qs_token = current_states[:, -1, :].detach().clone()
        return qs_token

    def select_memory(self, sess: Session, query: str):
        logger.info(f'session={sess.hash}, id={sess._id}, select_memory')
        qs_token = self.forward_question(sess, query)

        # select relevant memory for each question
        full_time = torch.cat(sess.full_time[:sess.query_clip], dim=0)
        full_time = torch.nn.functional.normalize(full_time.cuda(), dim=1, p=2)
        qs_token = torch.nn.functional.normalize(qs_token.cuda(), dim=1, p=2)
        similarity = torch.einsum('qc,nc->qn', qs_token, full_time)
        select_index = torch.where(similarity>0.3)[1].tolist()
        select_memory = []
        select_img = []
        for idx in select_index:
            select_memory.append(sess.full_memory[idx])
            select_img.append(sess.full_img[idx])
        sess.select_memory = select_memory # selected local memory
        sess.select_img = select_img # selected img
        sess.select_global = sess.global_memory[sess.query_clip-1] # final global memory
        print(similarity, sess.global_memory[-1].shape, sess.full_img[-1].shape, sess.full_memory[-1].shape)
        # torch.save(sess.global_memory[sess.query_clip - 1], 'temp_glb.pth')
        # torch.save(select_memory, 'temp_lol_mem.pth')
        # torch.save(select_img, 'temp_lol_img.pth')


model = Model('/app/Long-Grounding-qwen')

sess = Session()
sess.init_video('examples/air.mp4', num_frm=16, max_clip=32)

# streaming process video frame by frame
model.process(sess)
