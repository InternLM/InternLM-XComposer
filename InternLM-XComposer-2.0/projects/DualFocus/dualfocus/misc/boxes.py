import numpy as np
import re


def norm_bbox(bbox, height, width):
    bbox = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    bbox[:, ::2] /= width
    bbox[:, 1::2] /= height
    return bbox

def denorm_bbox(bbox, height, width):
    bbox = np.array(bbox).reshape(-1, 4)
    bbox[:, ::2] *= width
    bbox[:, 1::2] *= height
    return bbox


def enlarge_box(bbox, height, width, enlarge_pixel=25):
    bbox = np.array(bbox, dtype=np.float32).reshape((-1, 4))
    height, width = float(height), float(width)
    bbox[:, 0] -= enlarge_pixel
    bbox[:, 2] += enlarge_pixel
    bbox[:, 1] -= enlarge_pixel
    bbox[:, 3] += enlarge_pixel

    bbox[:, ::2] = np.clip(bbox[:, ::2], 0., width - 1)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0., height - 1)
    return bbox


def expand_box(bbox, height, width):
    assert not np.all(bbox <= 1.)
    bbox = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    height, width = float(height), float(width)
    cxcy = (bbox[:, 2:] + bbox[:, :2]) / 2.
    whs = bbox[:, 2:] - bbox[:, :2]
    max_w_h = np.max(whs, axis=-1)
    expanded_x1y1 = cxcy - max_w_h / 2.
    expanded_x2y2 = cxcy + max_w_h / 2.
    expanded_bbox = np.concatenate([expanded_x1y1, expanded_x2y2], axis=-1)
    expanded_bbox[:, ::2] = np.clip(expanded_bbox[:, ::2], 0., width - 1)
    expanded_bbox[:, 1::2] = np.clip(expanded_bbox[:, 1::2], 0., height - 1)
    return expanded_bbox


def box2str(box):
    box = list(box)
    x1, y1, x2, y2 = box
    box_str = f'[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]'
    return box_str


def find_boxes(string):
    pat = re.compile('\d+\.?\d*,\s*\d+\.?\d*,\s*\d+\.?\d*,\s*\d+\.?\d*')
    boxes_str = pat.findall(string)
    boxes = []
    for box_str in boxes_str:
        box = box_str.split(',')
        box = [float(x) for x in box]
        boxes.append(box)
    return np.array(boxes).reshape(-1, 4)


def find_boxes_qw(string):
    pat = re.compile('\d+,\d+')
    boxes_str = pat.findall(string)
    boxes = []
    for box_str in boxes_str:
        box = box_str.split(',')
        box = [float(x) for x in box]
        boxes.extend(box)
    return np.array(boxes).reshape(-1, 4)
