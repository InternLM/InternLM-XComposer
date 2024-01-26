import torch
from transformers import AutoModel, AutoTokenizer

import auto_gptq
from auto_gptq.modeling import BaseGPTQForCausalLM

auto_gptq.modeling._base.SUPPORTED_MODELS = ["InternLMXComposer"]

torch.set_grad_enabled(False)


class InternLMXComposerQForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "internlm_model.model.layers"
    outside_layer_modules = [
        "query_tokens",
        "flag_image_start",
        "flag_image_end",
        "visual_encoder",
        "Qformer",
        "internlm_model.model.embed_tokens",
        "internlm_model.model.norm",
        "internlm_proj",
        "internlm_model.lm_head",
    ]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj"],
        ["mlp.up_proj"],
        ["mlp.down_proj"],
    ]


# init model and tokenizer
model = InternLMXComposerQForCausalLM.from_quantized(
    "internlm/internlm-xcomposer-7b-4bit", trust_remote_code=True, device="cuda:0"
)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    "internlm/internlm-xcomposer-7b-4bit", trust_remote_code=True
)
model.model.tokenizer = tokenizer

# example image
image = "examples/images/aiyinsitan.jpg"

# Multi-Turn Text-Image Dialogue
# 1st turn
text = 'Describe this image in detial.'
image = "examples/images/aiyinsitan.jpg"
response, history = model.chat(text, image)
print(f"User: {text}")
print(f"Bot: {response}") 
# The image features a black and white portrait of Albert Einstein, the famous physicist and mathematician. 
# Einstein is seated in the center of the frame, looking directly at the camera with a serious expression on his face. 
# He is dressed in a suit, which adds a touch of professionalism to his appearance. 

# 2nd turn
text = "What is the style of the image."
response, history = model.chat(text=text, image=None, history=history)
print(f"User: {text}")
print(f"Bot: {response}")
# The style of the image is a black and white portrait.

