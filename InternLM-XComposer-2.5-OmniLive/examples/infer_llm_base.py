import sys
sys.path.insert(0, '../')

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from example_code.utils import auto_configure_device_map

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--dtype", default='fp16', type=str)
args = parser.parse_args()

model_path = 'internlm-xcomposer2d5-ol-7b/base'
# init model and tokenizer
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()

if args.dtype == 'fp16':
    model.half().cuda()
elif args.dtype == 'fp32':
    model.cuda()

if args.num_gpus > 1:
    from accelerate import dispatch_model
    device_map = auto_configure_device_map(args.num_gpus)
    model = dispatch_model(model, device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.tokenizer = tokenizer

question = 'Analyze the given image in a detail manner'
image = ['../examples/dubai.png']

with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, _ = model.chat(tokenizer, question, image, do_sample=False, num_beams=3, use_meta=True)
print(response)

# Expected output
"""
The image presents an infographic titled "Amazing Facts About Dubai," highlighting various aspects of the city. It begins with a depiction of Dubai's skyline, featuring iconic buildings like the Burj Al Arab and the Burj Khalifa. The infographic mentions that Palm Jumeirah is the largest artificial island in the world and is visible from space. It also states that in 1968, there were only 1.5 million cars in Dubai, whereas today there are more than 1.5 million cars. Dubai has the world's largest Gold Chain, with about 7 of the 10 tallest hotels in the world located there. The Gold Chain is 4.2 km long. The crime rate in Dubai is 0%, and the income tax rate is also 0%. Dubai Mall is the largest shopping mall in the world with 1200 stores. Dubai has no standard address system, and the Burj Khalifa is so tall that its residents on top floors need to wait longer to break fast during Ramadan. Dubai is building a climate-controlled City, and the Royal Suite at the Burj Al Arab costs $24,000 per night. The net worth of the four listed billionaires is roughly equivalent to the GDP of Honduras. The infographic concludes with a note that you need a license to drink alcohol even at home. The sources of the information are cited at the bottom, and the infographic was designed and compiled by www.fmextensions.ae.
"""
