import os
import sys
import time
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.utils import disable_logging
from dataclasses import dataclass
from itertools import count
from decord import VideoReader
import numpy as np
import copy
import torch
from PIL import Image
import warnings
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoProcessor
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from queue import Queue

from lmdeploy.pytorch.engine import Engine
from lmdeploy import PytorchEngineConfig
sys.path.append(f'{os.getenv("ROOT_DIR")}/memory')
from grounding_qwen import GroundQwenForCausalLM

import threading

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
        self.current_clip = 0
        self.query_clip = 0
        self.current_frame = 0
        self.previous_segment = 0
        self.select_memory = None
        self.select_global = None
        self.select_img = None
        self.todo_task = None
        self.todo_finish = False
        # self.full_memory = torch.load('full_memory.pth')
        # self.full_time = torch.load('full_time.pth')
        # self.full_img = torch.load('full_img.pth')
        # self.global_memory = torch.load('global_memory.pth')

    def init_video(self, frame_list: Queue, num_frm: int, max_clip: int):
        self.vr = frame_list
        self.num_frm = num_frm
        self.num_clip = max_clip

    def backup(self):
        self.full_memory_backup = copy.deepcopy(self.full_memory)
        self.full_time_backup = copy.deepcopy(self.full_time)
        self.full_img_backup = copy.deepcopy(self.full_img)
        self.global_memory_backup = copy.deepcopy(self.global_memory)
        # torch.save(self.full_memory_backup, 'full_memory.pth')
        # torch.save(self.full_time_backup, 'full_time.pth')
        # torch.save(self.full_img_backup, 'full_img.pth')
        # torch.save(self.global_memory_backup, 'global_memory.pth')

    #def get_next_frame(self, stop_event: threading.Event):
    def get_next_frame(self, sig_model, sig_processor):
        img_list = []
        pre_image_embeds = None
        with threading.Lock():
            #while not stop_event.is_set():
            while True:
                if not self.vr.empty():
                    #print(f"frame_list: {len(self.vr.queue)}")
                    frame = self.vr.get()
                    while not self.vr.empty():
                        frame = self.vr.get()
                    img = frame[np.newaxis, :, :, ::-1]  # bgr2rgb

                    img_pil = Image.fromarray(img[0])
                    inputs = sig_processor(images=[img_pil], return_tensors="pt")
                    with torch.no_grad():
                        output = sig_model.vision_model(inputs['pixel_values'][:1].to(torch.cuda.device_count() - 1)).pooler_output
                    image_embeds = output / output.norm(p=2, dim=-1, keepdim=True)
                    if (self.current_frame-self.previous_segment >= 3) and (self.current_frame-self.previous_segment >= 16 or \
                            (pre_image_embeds is not None and (image_embeds * pre_image_embeds).sum().item() <= 0.9)):
                        need_new_seg = True
                    else:
                        need_new_seg = False

                    pre_image_embeds = image_embeds

                    #need_new_seg = True if (self.current_frame-self.previous_segment) >= 16 else False
                    if need_new_seg:
                        self.previous_segment = self.current_frame
                        print(f'==previous clip: {len(img_list)}')
                        img_list = [img]
                    else:
                        img_list.append(img)

                    img_list = [it for it in img_list if it.shape[1] == img_list[-1].shape[1]]
                    if len(img_list) == 0:
                        yield None, None, None, None
                        continue
                    img = np.concatenate(img_list, axis=0)
                    #h, w = 224, 224
                    h, w = 336, 336
                    img_array = torch.from_numpy(img).permute(0, 3, 1, 2).float()
                    img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
                    img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
                    clip_imgs = [Image.fromarray(x) for x in img_array]
                    start = self.previous_segment
                    end = self.current_frame + 1
                    self.current_frame += 1
                    if need_new_seg or len(self.full_img) == 0:
                        self.full_img.append(img)
                    else:
                        self.full_img[-1] = img
                    yield clip_imgs, start, end, need_new_seg
                else:
                    yield None, None, None, None

                time.sleep(0.2)


class Model:
    """streaming encoder."""

    def __init__(self, model_path: str):
        # pass
        self._load_siglip()
        self._load_processor(model_path)
        self._load_llava_qwen(model_path)
        torch.cuda.empty_cache()

    def _load_siglip(self):
        self.sig_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(torch.cuda.device_count() - 1)
        self.sig_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

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

    def forward_video_encoding(self, images: torch.Tensor,
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

        if need_new_seg or len(self.sess.full_memory) == 0:
            self.sess.full_memory.append(history_mem) # k c
            self.sess.full_time.append(history_time) # 1 c
        else:
            self.sess.full_memory[-1] = history_mem
            self.sess.full_time[-1] = history_time

        global_embeddings = [torch.cat([torch.cat(self.sess.full_memory, dim=0), torch.cat(self.sess.full_time, dim=0)], dim=0)]
        global_ids = [0] * global_embeddings[0].shape[0]
        global_embedding_ranges = [[0, len(global_ids)]]
        current_states = self.pt_model.decode(
            [global_ids],
            input_embeddings=[global_embeddings],
            input_embedding_ranges=[global_embedding_ranges],
        )
        global_memory = current_states[:, -len(self.sess.full_time):, :].detach().clone().squeeze(0)
        if need_new_seg or len(self.sess.global_memory) == 0:
            self.sess.global_memory.append(global_memory) # t c
        else:
            self.sess.global_memory[-1] = global_memory

    #def process(self, stop_event: threading.Event, stream_open: threading.Event):
    def process(self, vs_dict):
        image_processor = self.image_processor
        
        # frame-wise streaming
        # for clips, start, end, need_new_seg in self.sess.get_next_frame(stop_event):
            # if stop_event.is_set():
            #     break
        for clips, start, end, need_new_seg in self.sess.get_next_frame(self.sig_model, self.sig_processor):
            if vs_dict['break']:
                print('break!!!!!!!!!!')
                vs_dict['stream_open'] = False
                break

            if clips is None:
                continue

            if vs_dict['backup']:
                vs_dict['backup'] = False
                self.sess.backup()

            #if self.sess.todo_task is not None:
            if vs_dict['query']:
                text, mem_folder = vs_dict['query'], f"{os.getenv('ROOT_DIR')}/RealTime/tmp/mem"
                self.select_memory(text, mem_folder)
                vs_dict['query'] = ''
                vs_dict['query_finish'] = True

            start_ = time.time()
            video = image_processor.preprocess(
                clips, return_tensors='pt')['pixel_values']
            seq = get_time_prompt(start, end, self.tokenizer)
            #print(f"{threading.current_thread()}:1111111111")
            self.forward_video_encoding(images=video, seqs=seq, need_new_seg=need_new_seg)
            if not vs_dict['stream_open']:
                vs_dict['stream_open'] = True
            # if not stream_open.is_set():
            #     stream_open.set()
            #print(f"handle {start} to {end} seconds")

        logger.info(f'session={self.sess.hash}, id={self.sess._id}, process finished')

    def init_sess(self, frame_list,):
        self.sess = Session()
        self.sess.init_video(frame_list, num_frm=16, max_clip=32)

    def _ensure_feature(self, sess: Session):
        while len(sess.full_time) == 0:
            time.sleep(0.1)

    def forward_question(self, query: str):
        question = preprocess_question([query], self.tokenizer)[0]

        self._ensure_feature(self.sess)
        print(len(self.sess.full_time_backup), len(self.sess.global_memory_backup), len(self.sess.full_img_backup))
        self.sess.query_clip = len(self.sess.full_time_backup)
        memory = self.sess.global_memory_backup[self.sess.query_clip - 1]
        input_ids = question.cpu().numpy().tolist()
        n_feat = memory.shape[0]
        input_embeddings = [memory]
        input_embedding_ranges = [[0, n_feat]]
        input_ids = [0] * n_feat + input_ids  # memory at the beginning
        print(f"mem token: {len(input_ids)}")

        current_states = self.pt_model.decode(
            [input_ids],
            input_embeddings=[input_embeddings],
            input_embedding_ranges=[input_embedding_ranges],
        )
        qs_token = current_states[:, -1, :].detach().clone()
        return qs_token

    def select_memory(self, query: str, folder: str):
        logger.info('select_memory')
        qs_token = self.forward_question(query)

        # select relevant memory for each question
        full_time = torch.cat(self.sess.full_time_backup[:self.sess.query_clip], dim=0)
        full_time = torch.nn.functional.normalize(full_time.cuda(), dim=1, p=2)
        qs_token = torch.nn.functional.normalize(qs_token.cuda(), dim=1, p=2)
        similarity = torch.einsum('qc,nc->qn', qs_token, full_time)
        select_index = torch.where(similarity>0.3)[1].tolist()
        select_index = select_index[:1]
        select_memory = []
        select_img = []
        for idx in select_index:
            select_memory.append(self.sess.full_memory_backup[idx])
            select_img.extend([Image.fromarray(item) for item in self.sess.full_img_backup[idx]])

        if len(select_img) == 0:
            msg = 'ground fail!!!'
            select_img.extend([Image.fromarray(item) for item in self.sess.full_img_backup[-1][-3:]])
            # num = len(select_img)
            # if num < 3 and len(self.sess.full_img_backup) > 1:
            #     select_img.extend([Image.fromarray(item) for item in self.sess.full_img_backup[-2][num-3:]])
        else:
            msg = 'ground success!!!'
        print(msg)

        self.sess.select_memory = select_memory # selected local memory
        self.sess.select_img = select_img # selected img
        self.sess.select_global = self.sess.global_memory_backup[self.sess.query_clip-1] # final global memory
        print(similarity, self.sess.global_memory_backup[-1].shape, self.sess.full_img_backup[-1].shape, self.sess.full_memory_backup[-1].shape)

        os.makedirs(os.path.join(folder, 'check', query[:30]), exist_ok=True)
        torch.save(self.sess.global_memory_backup[self.sess.query_clip - 1], os.path.join(folder, 'temp_glb.pth'))
        torch.save(self.sess.global_memory_backup[self.sess.query_clip - 1], os.path.join(folder, 'check', query[:30], 'temp_glb.pth'))
        if len(select_memory) == 0:
            torch.save(self.sess.full_memory_backup, os.path.join(folder, 'temp_lol_mem.pth'))
            torch.save(self.sess.full_memory_backup, os.path.join(folder, 'check', query[:30], 'temp_lol_mem.pth'))
        else:
            torch.save(select_memory, os.path.join(folder, 'temp_lol_mem.pth'))
            torch.save(select_memory, os.path.join(folder, 'check', query[:30], 'temp_lol_mem.pth'))
        torch.save(select_img, os.path.join(folder, 'temp_lol_img.pth'))
        torch.save(select_img, os.path.join(folder, 'check', query[:30], 'temp_lol_img.pth'))
        with open(os.path.join(folder, 'prompt.txt'), 'w') as fd:
            fd.write(msg + '\n')
            fd.write(query)
        self.sess.todo_finish = True
