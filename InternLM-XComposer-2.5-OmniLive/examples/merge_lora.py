import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = 'internlm-xcomposer2d5-ol-7b'
base_dir = f'{model_dir}/base'
adapter_dir = f'{model_dir}/adapter'
out_dir = f'{model_dir}/merge_lora'
peft_config = PeftConfig.from_pretrained(adapter_dir)
model = AutoModelForCausalLM.from_pretrained(base_dir,
                                             return_dict=True,
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)

# Load the PEFT model
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
