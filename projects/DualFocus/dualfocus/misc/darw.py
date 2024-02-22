import copy
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def wrap(string, max_width):
    lines = string.split('\n')
    new_lines = []
    for line in lines:
        line = textwrap.fill(line, max_width)
        new_lines.append(line)
    string = '\n'.join(new_lines)
    return string


def draw_box_to_img(img_pil, box):
    draw = ImageDraw.Draw(img_pil)
    color = tuple([0, 255, 0])
    x1, y1, x2, y2 = box
    draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=5)
    return img_pil


def plot_boxes_to_image(img,
                        meta,
                        resize_height=800,
                        resize_width=600,
                        font_size=20,
                        font_type='/mnt/petrelfs/share_data/caoyuhang/calibri.ttf',
                        wrap_num=60,
                        box_width=3,
                        out_dir=None):
    if isinstance(img, str):
        img_pil = Image.open(img)
    elif isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)
    else:
        img_pil = img
    img_pil = img_pil.convert('RGB')
    ori_h = img_pil.height
    ori_w = img_pil.width

    new_w = max(img_pil.width, resize_width)
    new_h = max(img_pil.height, resize_height)
    new_img_pil = Image.new('RGB', (new_w, new_h), (255, 255, 255))
    new_img_pil.paste(img_pil, (0, 0, img_pil.width, img_pil.height))
    img_pil = new_img_pil

    # img_pil = resizeimage.resize_height(img_pil, resize_height)
    # img_pil = resizeimage.resize_width(img_pil, resize_width)

    tgt = copy.deepcopy(meta)
    image_h = img_pil.height
    image_w = img_pil.width

    boxes = tgt["boxes"]
    boxes = np.array(boxes).reshape(-1, 4)
    if np.all(boxes <= 1):
        boxes[:, ::2] *= ori_w
        boxes[:, 1::2] *= ori_h
    labels = meta.get('labels', None)
    if labels is None:
        labels = ['' for _ in boxes]

    draw = ImageDraw.Draw(img_pil)
    mask = Image.new("L", img_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    interval = font_size
    font = ImageFont.truetype(font_type, interval)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        color = tuple([0, 255, 0])
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=box_width)
        # draw.text((x0, y0), str(label), fill=color)

        # font = ImageFont.load_default(size=75)
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color, width=box_width)
        draw.text((x0, y0), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=3)

    if 'caps' in meta:
        img_np = np.array(img_pil)
        width = img_pil.width
        height = img_pil.height
        dummy_patch = np.zeros([height, 600, 3]) + 255.
        dummy_patch = np.asanyarray(dummy_patch, dtype=np.uint8)
        img_np = np.concatenate([dummy_patch, img_np], axis=1)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        cur_text_height = 0
        for i, cap in enumerate(meta['caps']):
            cap = wrap(cap, wrap_num)
            draw.text((0, cur_text_height), cap, (0, 0, 0), font=font)
            cur_text_height += draw.textsize(cap, font)[1]
            if not cap.endswith('\n'):
                draw.text((0, cur_text_height), '\n', (0, 0, 0), font=font)
                cur_text_height += 0.2 * draw.textsize('\n', font)[1]

    return img_pil
