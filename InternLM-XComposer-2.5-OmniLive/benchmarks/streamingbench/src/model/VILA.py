import torch
from llava.mm_utils import opencv_extract_frames
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from model.modelclass import Model

class VILA(Model):
    def __init__(self):
        VILA_Init()

    def Run(self, file, inp):
        return VILA_Run(file, inp)

    def name(self):
        return "VILA"

tokenizer, model, image_processor, context_len = None, None, None, None

def VILA_Init():
    model_path = "Efficient-Large-Model/Llama-3-VILA1.5-8B"
    model_name = get_model_name_from_path(model_path)
    global tokenizer, model, image_processor, context_len
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)

def VILA_Run(file, inp):
    num_video_frames = 16
    images, num_frames = opencv_extract_frames(file, num_video_frames)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if model.config.mm_use_im_start_end:
        qs = (image_token_se + "\n") * len(images) + inp
    else:
        qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + inp

    conv_mode = "llama_3"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs
