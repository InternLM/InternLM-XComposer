import os
import json
import random
import numpy as np

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def conv2text(sources):
    BEGIN_SIGNAL = " "
    END_SIGNAL = "\n"
    conversation = ''

    for sentence in sources:
        from_str = sentence["from"]
        if from_str.lower() == "human" or from_str.lower() == "user":
            from_str = '<|User|>'
            temp = (BEGIN_SIGNAL + from_str + ": " + 
                    sentence["value"] + END_SIGNAL)
        else:
            from_str = '<|Bot|>'
            temp = (BEGIN_SIGNAL + from_str + ": " + 
                    sentence["value"] + END_SIGNAL)
        conversation += temp
    conversation = conversation.replace('<image>', '').strip()
    conversation = conversation.split('<|User|>:', 1)[1].strip()

    return conversation

class ImageProcessor:
    def __init__(self, image_size=224):

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        item = Image.open(item).convert("RGB")
        return self.transform(item)
        
class Mix_dataset(Dataset):
    def __init__(self, vl_data_path, txt_data_path, inner_batch_size=1, img_size=224, local_rank=0):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super(Mix_dataset, self).__init__()
        print (f"init mix data at rank {local_rank}")
        self.datasets = []
        self.data_names = []
        self.data_num = []

        self.inner_batch_size = inner_batch_size
        self.set_seed = False
        self.local_rank = local_rank

        batch_size = inner_batch_size 
        vl_data_set = Sample_dataset(vl_data_path, batch_size, has_img = True)
        self.datasets.append(vl_data_set)
        self.data_num.append(len(vl_data_set))
        
        batch_size = inner_batch_size 
        txt_data_set = Sample_dataset(txt_data_path, batch_size, has_img = False)
        self.datasets.append(txt_data_set)
        self.data_num.append(len(txt_data_set))


        self.data_ratio = [float(ratio) / sum(self.data_num) for ratio in self.data_num]


    def __len__(self):
        return int(np.sum(self.data_num)/self.inner_batch_size)

    def __getitem__(self, index):
        if not self.set_seed:
            random.seed(index)
            self.set_seed = True
            print (f"Set seed {index} for rank {self.local_rank}")

        data_idx = random.choices(range(len(self.data_ratio)), self.data_ratio, k=1)[0]
        sample = self.datasets[data_idx].get_item()

        return dict(
            samples = sample
        )

class Sample_dataset(Dataset):
    def __init__(self, data_path, batch_size, has_img=True):
        self.raw_data = json.load(open(data_path, 'r'))
        print (f'load {len(self.raw_data)} data from {data_path}')
        self.batch_size = batch_size
        self.vis_processor = ImageProcessor()
        self.text_processor = conv2text
        self.has_img = has_img

    def __len__(self):
        return len(self.raw_data)

    def __get_item__(self, i):
        conv_text = conv2text(self.raw_data[i]["conversations"])
        sample = dict(
            text_input = conv_text,
        )
        if self.has_img:
            image_file = self.raw_data[i]['image']
            image = self.vis_processor(image_file)
            sample['image'] = image
        else:
            sample['image'] = torch.zeros(3,224,224)

        return sample

    def get_item(self,):
        text_input = ''
        images = []
        sp_token = '<HF_SPLIT_TOKEN>'
        for i in range(self.batch_size):
            idx = random.randrange(len(self.raw_data))
            sample = self.__get_item__(idx)
            text_input += sp_token + sample['text_input']
            images.append(sample['image'].unsqueeze(0))
        images = torch.cat(images, dim=0)
        text_input = text_input.split(sp_token, 1)[-1]
        sample = {
            "image": images,
            "text_input": text_input,
            'data_type': 'multi' if self.has_img else 'text',
            'sp_token': sp_token
        }
        return sample

