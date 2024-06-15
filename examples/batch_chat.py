import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import argparse
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from examples.utils import auto_configure_device_map

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--dtype", default='fp16', type=str)
args = parser.parse_args()

meta_instruction = """You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.
"""


img_paths = ['examples/image1.webp',
             'examples/image1.webp']
questions = ['Please describe this image in detail.',
             'What is the text in this images? Please describe it in detail.']

assert len(img_paths) == len(questions)

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

images = []
for img_path in img_paths:
    image = Image.open(img_path).convert('RGB')
    image = model.vis_processor(image).unsqueeze(0).half().cuda()
    images.append(image)
images = torch.cat(images, dim=0)

with torch.cuda.amp.autocast():
    with torch.no_grad():
        images = model.encode_img(images)

        inputs_list = []
        masks_list = []
        max_len = 0
        for image, question in zip(images, questions):
            inputs, im_mask = model.interleav_wrap_chat(tokenizer, "<ImageHere>" + question, image.unsqueeze(0), [], meta_instruction)
            inputs_list.append(inputs)
            masks_list.append(im_mask)
            max_len = max(max_len, im_mask.shape[1])

        pad_embed = model.model.tok_embeddings(torch.tensor(tokenizer.pad_token_id).cuda()).unsqueeze(0).unsqueeze(0)
        batch_inputs, batch_masks, batch_atten_masks = [], [], []
        for inputs, im_mask in zip(inputs_list, masks_list):
            if im_mask.shape[1] < max_len:
                pad_length = max_len - im_mask.shape[1]
                pad_embeds = pad_embed.repeat(1, pad_length, 1) 
                pad_masks = torch.tensor([0]*(max_len - im_mask.shape[1])).unsqueeze(0).cuda()
                inputs = torch.cat([pad_embeds, inputs['inputs_embeds']], dim=1)
                atten_masks = torch.cat([pad_masks, torch.ones_like(im_mask)], dim=1)
                im_mask = torch.cat([pad_masks, im_mask], dim=1)
            else:
                inputs = inputs['inputs_embeds']
                atten_masks = torch.ones_like(im_mask)

            batch_inputs.append(inputs)
            batch_masks.append(im_mask)
            batch_atten_masks.append(atten_masks)

        batch_inputs = {'inputs_embeds': torch.cat(batch_inputs, dim=0)}
        batch_masks = torch.cat(batch_masks, dim=0).bool()
        batch_atten_masks = torch.cat(batch_atten_masks, dim=0).bool()

        print(batch_inputs['inputs_embeds'].shape, batch_masks.shape)
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
        ]

        outputs = model.generate(
                    **batch_inputs,
                    streamer=None,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=1.0,
                    top_p=0.8,
                    eos_token_id=eos_token_id,
                    repetition_penalty=1.005,
                    im_mask=batch_masks,
                    attention_mask=batch_atten_masks,
                )

for i in range(outputs.shape[0]):
    output = outputs[i].cpu().tolist()
    response = tokenizer.decode(output, skip_special_tokens=True)
    print(response.split('[UNUSED_TOKEN_145]')[0])
    print('=======================')
# The image features a quote by Oscar Wilde, "Live life with no excuses, travel with no regret,"
# set against a backdrop of a breathtaking sunset. The sky is painted in hues of pink and orange,
# creating a serene atmosphere. Two silhouetted figures stand on a cliff, overlooking the horizon.
# They appear to be hiking or exploring, embodying the essence of the quote.
# The overall scene conveys a sense of adventure and freedom, encouraging viewers to embrace life without hesitation or regrets.
