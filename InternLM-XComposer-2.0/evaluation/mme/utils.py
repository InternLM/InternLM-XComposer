import torch
import torchvision
from PIL import Image

def __resize_img__( b):
        
        width, height = b.size
        tar = max(width, height)
        top_padding = int((tar - height)/2)
        bottom_padding = tar - height - top_padding
        left_padding = int((tar - width)/2)
        right_padding = tar - width - left_padding
        b = torchvision.transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding])
        return b

def model_gen( model, text, images, need_bos=True):
    text = text.split('Please answer')[0].strip()
    text = f'{text} Answer this question briefly'
    text = f'[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
    pt1 = 0
    embeds = []
    im_mask = []
    images = [images]
    images_loc = [0]
    for i, pts in enumerate(images_loc + [len(text)]):
        subtext = text[pt1:pts]
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            embeds.append(text_embeds)
            need_bos = False
        if i < len(images):
            image = Image.open(images[i]).convert('RGB')
            image = __resize_img__(image)
            image = model.vis_processor(image).unsqueeze(0).cuda()
            image_embeds = model.encode_img(image)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
            embeds.append(image_embeds)
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                        temperature=1.0, max_new_tokens=5, num_beams=5,
                        do_sample=False, repetition_penalty=1.0)
    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0]
    return output_text

