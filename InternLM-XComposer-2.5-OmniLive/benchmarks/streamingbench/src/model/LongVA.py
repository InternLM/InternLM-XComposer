from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed

torch.manual_seed(0)

tokenizer, model, image_processor, max_frames_num, gen_kwargs = None, None, None, None, None

from model.modelclass import Model
class LongVA(Model):
    def __init__(self):
        LongVA_Init()

    def Run(self, file, inp):
        return LongVA_Run(file, inp)
    
    def name(self):
        return "LongVA"

def LongVA_Init():
    global tokenizer, model, image_processor, max_frames_num, gen_kwargs

    model_path = "lmms-lab/LongVA-7B"

    max_frames_num = 128 # you can change this to several thousands so long you GPU memory can handle it :)
    gen_kwargs = {"do_sample": False, "temperature": 0.0, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
    # you can also set the device map to auto to accomodate more frames
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto")

def LongVA_Run(file, inp):
    video_path = file
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{}<|im_end|>\n<|im_start|>assistant\n".format(inp)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)

    return outputs