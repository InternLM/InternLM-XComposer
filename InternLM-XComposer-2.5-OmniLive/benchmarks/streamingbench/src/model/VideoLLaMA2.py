import torch
from videollama2.conversation import conv_templates
from videollama2.utils import disable_torch_init
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from videollama2.model.builder import load_pretrained_model

model, processor, context_len, tokenizer, conv_mode = None, None, None, None, None

from model.modelclass import Model
class VideoLLaMA2(Model):
    def __init__(self):
        VideoLLaMA2_Init()

    def Run(self, file, inp):
        return VideoLLaMA2_Run(file, inp)
    
    def name(self):
        return "VideoLLaMA2"

def VideoLLaMA2_Init():
    global model, processor, context_len, tokenizer, conv_mode
    disable_torch_init()
    # 1. Initialize the model.
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
    # Base model inference (only need to replace model_path)
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-Base'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    conv_mode = 'llama_2'

def VideoLLaMA2_Run(file, inp):
    # Video Inference
    paths = [file]
    questions = [inp]
    # Reply:
    # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.
    modal_list = ['video']

    # 2. Visual preprocess (load & transform image or video).
    if modal_list[0] == 'video':
        tensor = process_video(paths[0], processor, model.config.image_aspect_ratio).to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    else:
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
        modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
    tensor = [tensor]

    # 3. text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs[0])
    return outputs[0]