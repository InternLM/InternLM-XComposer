import argparse
import json
import math
import os

import numpy as np
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from dualfocus.constants import IMAGE_TOKEN_INDEX
from dualfocus.conversation import conv_templates, SeparatorStyle
from dualfocus.misc import expand_box, enlarge_box, denorm_bbox, find_boxes, process_outputs
from dualfocus.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from dualfocus.model.builder import load_pretrained_model
from dualfocus.train.train import preprocess, DataCollatorForSupervisedDataset
from dualfocus.utils import disable_torch_init, all_logging_disabled

all_options = ['A', 'B', 'C', 'D']

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def generate(model, prompt, image_tensor, tokenizer, device):
    input_ids = tokenizer_image_token(prompt,
                                      tokenizer,
                                      IMAGE_TOKEN_INDEX,
                                      return_tensors='pt')[None, ...]
    stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else \
    conv_templates[args.conv_mode].sep2
    input_ids = input_ids.to(device=device, non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16,
                                   device=device,
                                   non_blocking=True),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=128,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids !=
                           output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids'
        )
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:],
                                     skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.use_slurm:
        device = f'cuda:{args.chunk_idx % 8}'
    else:
        device = f'cuda'
    with all_logging_disabled():
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map=device, device=device)

    questions = [
        json.loads(q)
        for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    collator = DataCollatorForSupervisedDataset(tokenizer)

    for line in tqdm(questions, total=len(questions)):
        idx = line["question_id"]
        question = line['text'].split('\n')[0]
        options = line['text'].split('\n')[1:-1]
        question_options_hint = line['text']

        image_file = os.path.join(args.image_folder, line['image'])
        img_pil = Image.open(image_file).convert('RGB')
        img_h, img_w = img_pil.height, img_pil.width
        image_tensor = process_images([img_pil], image_processor, model.config)

        # Normal Inference
        base_prompt_raw = f'<image>\n{question_options_hint}'
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], base_prompt_raw)
        conv.append_message(conv.roles[1], None)
        base_prompt = conv.get_prompt()
        base_answer = generate(model, base_prompt, image_tensor, tokenizer, device=device)
        base_answer = process_outputs(base_answer, options)
        outputs = base_answer
        base_ans_valid = False
        if base_answer.upper() in all_options:
            base_ans_valid = True

        # DualFocus Inference
        round1_question = f'<image>\nPlease provide the bounding box coordinate of the region this question asks: {question}'
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], round1_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_file = os.path.join(args.image_folder, line['image'])
        img_pil = Image.open(image_file).convert('RGB')
        image_tensor = process_images([img_pil], image_processor, model.config)
        round1_outputs = generate(model, prompt, image_tensor, tokenizer, device)

        try:
            boxes = find_boxes(round1_outputs)
            sub_area = denorm_bbox(boxes, img_h, img_w)[0]
            sub_area = expand_box(sub_area, img_h, img_w)[0]
            sub_area = enlarge_box(sub_area, img_h, img_w)[0]
            sub_area = sub_area.astype(np.int32)
            x1, y1, x2, y2 = sub_area
            img_np = np.asarray(img_pil)
            sub_img_np = img_np[y1:y2, x1:x2]
            sub_img_pil = Image.fromarray(sub_img_np).convert('RGB')
            cot = True
        except Exception as e:
            cot = False
            print(e, round1_outputs)

        if not cot:
            outputs = base_answer
        else:
            sub_image_tensor = process_images([sub_img_pil], image_processor, model.config)
            image_tensor = torch.cat([image_tensor, sub_image_tensor])
            round2_question = f'<image>\nPlease combine the information from these two images and answer this question: {question_options_hint}'
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], round1_question)
            conv.append_message(conv.roles[1], round1_outputs)
            conv.append_message(conv.roles[0], round2_question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            cot_answer = generate(model, prompt, image_tensor, tokenizer, device)
            cot_answer = cot_answer.strip('\n').strip('.')
            cot_answer = process_outputs(cot_answer, options)
            if cot_answer.upper() in all_options:
                cot_ans_valid = True
            else:
                cot_ans_valid = False

        do_ppl = False
        if not cot:
            outputs = base_answer
        else:
            if base_ans_valid and (not cot_ans_valid):
                outputs = base_answer
            if (not base_ans_valid) and cot_ans_valid:
                outputs = cot_answer
            if (not base_ans_valid) and (not cot_ans_valid):
                outputs = base_answer
            if base_ans_valid and cot_ans_valid:
                do_ppl = True

        # PPL Selection
        if do_ppl:
            ppl_question = f'<image>\nPlease combine the information from these two images and answer this question: {question}\nAnswer the question using a single word or phrase'
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], round1_question)
            conv.append_message(conv.roles[1], round1_outputs)
            conv.append_message(conv.roles[0], ppl_question)
            # conv.append_message(conv.roles[1], None)
            ppl_prompt = conv.get_prompt()
            ppl_prompt = ppl_prompt[ppl_prompt.index('<image>'):]

            new_options = [options[ord(base_answer) - ord('A')][3:], options[ord(cot_answer) - ord('A')][3:]]
            convs = []
            for option in new_options:
                opt_conv = []
                opt_conv.append({'from': 'human', 'value': ppl_prompt})
                opt_conv.append({'from': 'gpt', 'value': option})
                tmp = preprocess([opt_conv], tokenizer, has_image=True)
                tmp['input_ids'] = tmp['input_ids'][0]
                tmp['labels'] = tmp['labels'][0]
                convs.append(tmp)

            input_dict = collator(convs)
            input_dict['input_ids'] = input_dict['input_ids'].squeeze().cuda()
            input_dict['labels'] = input_dict['labels'].squeeze().cuda()
            input_dict['attention_mask'] = input_dict['attention_mask'].squeeze().cuda()
            with torch.inference_mode():
                loss = model.forward(**input_dict, images=image_tensor.repeat(len(convs), 1, 1, 1).cuda().half(),
                                     reduction='none')['loss']
            out_option_idx = loss.argmin()
            ppl_answer = [base_answer, cot_answer][out_option_idx]
            outputs = ppl_answer

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps({
                "question_id": idx,
                "prompt": line['text'],
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {}
            }) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file",
                        type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    if 'lora' not in args.model_path:
        args.model_base = None

    if args.chunk_idx == -1:
        proc_id = int(os.environ['SLURM_PROCID'])
        args.chunk_idx = proc_id
        prefix = '/'.join(args.answers_file.split('/')[:-1])
        args.answers_file = os.path.join_path(prefix, f'{args.num_chunks}_{proc_id}.jsonl')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(proc_id % 8)
        torch.cuda.set_device(f'cuda:{proc_id % 8}')
        args.use_slurm = True
    else:
        args.use_slurm = False

    eval_model(args)
