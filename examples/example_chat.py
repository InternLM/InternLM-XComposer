import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from examples.utils import auto_configure_device_map

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--dtype", default='fp16', type=str)
args = parser.parse_args()

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).eval()
if args.dtype == 'fp16':
    model.half().cuda()
elif args.dtype == 'fp32':
    model.cuda()

if args.num_gpus > 1:
    from accelerate import dispatch_model
    device_map = auto_configure_device_map(args.num_gpus)
    model = dispatch_model(model, device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)

text = '<ImageHere>Please describe this image in detail.'
image = 'examples/image1.webp'
with torch.cuda.amp.autocast():
    response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False)
print(response)
# The image features a quote by Oscar Wilde, "Live life with no excuses, travel with no regret,"
# set against a backdrop of a breathtaking sunset. The sky is painted in hues of pink and orange,
# creating a serene atmosphere. Two silhouetted figures stand on a cliff, overlooking the horizon.
# They appear to be hiking or exploring, embodying the essence of the quote.
# The overall scene conveys a sense of adventure and freedom, encouraging viewers to embrace life without hesitation or regrets.
