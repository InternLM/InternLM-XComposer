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

# Single-Turn Pure-Text Dialogue
text = "请介绍下爱因斯坦的生平"
response = model.generate(text=text)
print(f"User: {text}")
print(f"Bot: {response}")

# Single-Turn Text-Image Dialogue
text = "请问这张图片里面的人是谁？并介绍下他。"
image = "examples/images/aiyinsitan.jpg"
response = model.generate(text=text, image=image)
print(f"User: {text}")
print(f"Bot: {response}")

# Multi-Turn Text-Image Dialogue
# 1st turn
text = "图片里面的是谁？"
response, history = model.chat(text=text, image=image, history=None)
print(f"User: {text}")
print(f"Bot: {response}")

# 2nd turn
text = "他有哪些成就?"
response, history = model.chat(text=text, image=None, history=history)
print(f"User: {text}")
print(f"Bot: {response}")

# 3rd turn
text = "他是最伟大的物理学家吗?"
response, history = model.chat(text=text, image=None, history=history)
print(f"User: {text}")
print(f"Bot: {response}")
