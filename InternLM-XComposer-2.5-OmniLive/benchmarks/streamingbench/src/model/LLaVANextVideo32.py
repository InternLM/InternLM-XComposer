import torch
import os
import numpy as np
from PIL import Image
import copy

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from decord import VideoReader, cpu
from transformers import AutoConfig

device = "cuda"
device_map = "auto"

from model.modelclass import Model
class LLaVANextVideo32(Model):
    def __init__(self):
        LLaVANextVideo32_Init()

    def Run(self, file, inp):
        return LLaVANextVideo32_Run(file, inp)
    
    def name(self):
        return "LLaVA-Next-Video-32B"

cfg_pretrained, tokenizer, model, image_processor = None, None, None, None

def LLaVANextVideo32_Init():
    # Initialize the model
    model_path = "lmms-lab/LLaVA-NeXT-Video-32B-Qwen"
    model_name = get_model_name_from_path(model_path)
    # Set model configuration parameters if they exist
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_mode"] = "average"
    overwrite_config["mm_spatial_pool_stride"] = 2
    overwrite_config["mm_newline_position"] = "grid"
    overwrite_config["mm_pooling_position"] = "after"

    global cfg_pretrained, tokenizer, model, image_processor
    cfg_pretrained = AutoConfig.from_pretrained(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, overwrite_config=overwrite_config)
    model.eval()

    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643

def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    if len(frame_idx) > 32:
        sample_fps = 32
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def LLaVANextVideo32_Run(file, inp):
    image_tensors = []
    image_sizes = []
    if file.endswith('.mp4'):
        video_frames = load_video(file)
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