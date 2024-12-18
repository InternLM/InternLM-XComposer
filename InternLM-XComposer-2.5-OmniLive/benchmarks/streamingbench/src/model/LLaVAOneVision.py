from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import os
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = None, None, None, None

from model.modelclass import Model
class LLaVAOneVision(Model):
    def __init__(self):
        LLaVAOneVision_Init()

    def Run(self, file, inp):
        return LLaVAOneVision_Run(file, inp)
    
    def name(self):
        return "LLaVA-OneVision"

def LLaVAOneVision_Init():
    # Load the OneVision model
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    global tokenizer, model, image_processor, max_length
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

    model.eval()

# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx)
    spare_frames = spare_frames.asnumpy()
    return spare_frames  # (frames, height, width, channels)

def LLaVAOneVision_Run(file, inp):
    image_tensors = []
    image_sizes = []
    if file.endswith('.mp4'):
        video_frames = load_video(file, 32)
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)
        image_sizes = [frame.size for frame in video_frames]
        modality = "video"
    elif file.endswith('.jpg'):
        image = Image.open(file)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        image_sizes = [image.size]
        modality = "image"
    else:
        images = []
        for img in os.listdir(file):
            img = os.path.join(file, img)
            image = np.asarray(Image.open(img))
            images.append(image)
        frames = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)
        image_sizes = [frame.size for frame in images]
        modality = "video"

    # Prepare conversation input
    conv_template = "qwen_1_5"
    question = f"{DEFAULT_IMAGE_TOKEN}\n{inp}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=[modality],
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    response = text_outputs[0]
    print(response)
    return response