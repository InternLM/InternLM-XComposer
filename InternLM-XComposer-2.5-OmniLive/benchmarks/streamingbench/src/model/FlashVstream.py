import requests
from decord import VideoReader, cpu
import torch
from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from flash_vstream.conversation import conv_templates, SeparatorStyle
from flash_vstream.model.builder import load_pretrained_model
from flash_vstream.utils import disable_torch_init
from flash_vstream.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from torch.multiprocessing import Process, Queue, Manager
from transformers import TextStreamer

from model.modelclass import Model
class FlashVstream(Model):
    def __init__(self):
        FlashVstream_Init()

    def Run(self, file, inp):
        return FlashVstream_Run(file, inp)
    
    def name(self):
        return "Flash-VStream"

def FlashVstream_Init():
    global tokenizer, model, image_processor, context_len
    model_path = "IVGSZ/Flash-VStream-7b"
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device="cuda", device_map="auto")
    print("Model initialized.")

def load_video(video_path):
    vr = VideoReader(video_path)
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def FlashVstream_Run(file, inp):
    
    video = load_video(file)
    video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
    video = [video]

    qs = inp
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video,
            do_sample=True,
            temperature=0.002,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
        
    input_token_len = input_ids.shape[1]
        
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    print(outputs)
    return outputs
