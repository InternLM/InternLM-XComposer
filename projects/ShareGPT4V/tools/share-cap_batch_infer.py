import argparse
import json

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", type=str,
                        default="Lin-Chen/ShareCaptioner")
    parser.add_argument("--images-file", type=str, default="images_to_describe.json",
                        help="a list, each element is a string for image path")
    parser.add_argument("--save-path", type=str, default="captions.json")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="cuda", trust_remote_code=True).eval()
    model.tokenizer = tokenizer

    model.cuda()
    model.half()

    imgs = json.load(open(args.images_file, 'r'))
    part_len = len(imgs)

    seg1 = '<|User|>:'
    seg2 = f'Analyze the image in a comprehensive and detailed manner.{model.eoh}\n<|Bot|>:'
    seg_emb1 = model.encode_text(seg1, add_special_tokens=True)
    seg_emb2 = model.encode_text(seg2, add_special_tokens=False)

    captions = []

    chunk_size = len(imgs)//args.batch_size

    if len(imgs) % args.batch_size != 0:
        chunk_size += 1

    for i in range(chunk_size):
        print(f'{i}/{chunk_size}')
        subs = []
        for j in range(args.batch_size):
            if i*args.batch_size+j < len(imgs):
                img_path = imgs[i*args.batch_size+j]
                image = Image.open(img_path).convert("RGB")
                subs.append(model.vis_processor(image).unsqueeze(0))
        if len(subs) == 0:
            break
        subs = torch.cat(subs, dim=0).cuda()
        tmp_bs = subs.shape[0]
        tmp_seg_emb1 = seg_emb1.repeat(tmp_bs, 1, 1)
        tmp_seg_emb2 = seg_emb2.repeat(tmp_bs, 1, 1)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                subs = model.encode_img(subs)
                input_emb = torch.cat(
                    [tmp_seg_emb1, subs, tmp_seg_emb2], dim=1)
                out_embeds = model.internlm_model.generate(inputs_embeds=input_emb,
                                                           max_length=500,
                                                           num_beams=3,
                                                           min_length=1,
                                                           do_sample=True,
                                                           repetition_penalty=1.5,
                                                           length_penalty=1.0,
                                                           temperature=1.,
                                                           eos_token_id=model.tokenizer.eos_token_id,
                                                           num_return_sequences=1,
                                                           )
        for j, out in enumerate(out_embeds):
            out[out == -1] = 2
            response = model.decode_text([out])
            captions.append({imgs[i*args.batch_size+j]: response})

    with open(args.save_path, 'w') as f:
        json.dump(captions, f, indent=4)
    print('Done')
