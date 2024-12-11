import torch
from swift.llm import (
    get_model_tokenizer, get_template, ModelType,
    get_default_template_type, inference
)
from swift.utils import seed_everything

model_type = ModelType.qwen2_audio_7b_instruct
model_id_or_path = 'internlm-xcomposer2d5-ol-7b/audio'
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_id_or_path=model_id_or_path,
                                       model_kwargs={'device_map': 'cuda:0'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

# Chinese ASR
query = '<audio>Detect the language and recognize the speech.'
response, _ = inference(model, template, query, audios='examples/audios/chinese.mp3')
print(f'query: {query}')
print(f'response: {response}')

# English ASR
query = '<audio>Detect the language and recognize the speech.'
response, _ = inference(model, template, query, audios='examples/audios/english.mp3')
print(f'query: {query}')
print(f'response: {response}')

# Audio Cls
query = '<audio>Classify the audio.'
response, _ = inference(model, template, query, audios='examples/audios/cls_cat.wav')
print(f'query: {query}')
print(f'response: {response}')
