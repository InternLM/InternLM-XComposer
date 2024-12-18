import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.modelclass import Model
class Kangaroo(Model):
    def __init__(self):
        Kangaroo_Init()

    def Run(self, file, inp):
        return Kangaroo_Run(file, inp)
    
    def name(self):
        return "Kangaroo"
    
tokenizer, model, terminators = None, None, None

def Kangaroo_Init():
    global tokenizer, model, terminators
    tokenizer = AutoTokenizer.from_pretrained("KangarooGroup/kangaroo")
    model = AutoModelForCausalLM.from_pretrained(
        "KangarooGroup/kangaroo",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cuda")
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

def Kangaroo_Run(file, inp):
    out, history = model.chat(video_path=file,
                            query=inp,
                            tokenizer=tokenizer,
                            max_new_tokens=512,
                            eos_token_id=terminators,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,)
    print('Assitant: \n', out)
    return out