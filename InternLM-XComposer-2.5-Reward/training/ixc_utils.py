import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def padding_336(b, R=336):
    width, height = b.size
    tar = int(np.ceil(height / R) * R)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding],
        fill=[255, 255, 255])
    return b


def R560_HD18_Identity_transform(img, resolution=560, hd_num=18):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1

    scale_low = min(np.ceil(width * 1.5 / resolution), scale)
    scale_up = min(np.ceil(width * 1.5 / resolution), scale)
    scale = random.randrange(scale_low, scale_up + 1)

    new_w = int(scale * resolution)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(
        img,
        [new_h, new_w],
    )
    img = padding_336(img, resolution)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img
