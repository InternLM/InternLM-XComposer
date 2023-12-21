import transformers
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--model', help='model name or path')
parser.add_argument('--tar_path', help='local path to save merged model')
args = parser.parse_args()

model_name_or_path = args.model
tar_path = args.tar_path

### save config
config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
del config.lora_cfg
config.internlm_lora = None
config.save_pretrained(tar_path)

### save tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer.save_pretrained(tar_path)

### merge & save model
model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,  ## internlm/internlm-xcomposer-7b
        device_map="cpu",
        trust_remote_code=True,
    )

model_ckpt = model.state_dict()
scale = 1
keys = list(model_ckpt)
for name in keys:
    if 'lora_A' in name:
        lora_a = name
        lora_b = name.replace('lora_A', 'lora_B')
        linear = name.replace('lora_A.', '')
        lora_a_weight = model_ckpt.pop(lora_a).float()
        lora_b_weight = model_ckpt.pop(lora_b).float()
        lora_weight = lora_b_weight @ lora_a_weight
        ori_weight = model_ckpt.pop(linear)
        new_weight = (ori_weight.float() + scale*lora_weight).to(ori_weight.device)
        model_ckpt[linear] = new_weight

new_model = transformers.AutoModel.from_pretrained(model_name_or_path, config=config, device_map='cpu', trust_remote_code=True)
new_model.load_state_dict(model_ckpt)
new_model.save_pretrained(tar_path, max_shard_size="5GB")
