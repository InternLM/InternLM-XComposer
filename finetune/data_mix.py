import random

import numpy as np
from ixc_utils import R560_HD18_Identity_transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def conv2text(sources):
    END_HUMAN = '[UNUSED_TOKEN_145]\n'
    END_BOT = '[UNUSED_TOKEN_145]\n'
    conversation = ''

    for idx, sentence in enumerate(sources):
        BEGIN_SIGNAL = ''

        from_str = sentence['from']
        if from_str.lower() == 'human' or from_str.lower() == 'user':
            from_str = '[UNUSED_TOKEN_146]user\n'
            temp = (
                BEGIN_SIGNAL + from_str + sentence['value'].strip() +
                END_HUMAN)
        else:
            from_str = '[UNUSED_TOKEN_146]assistant\n'
            temp = (
                BEGIN_SIGNAL + from_str + sentence['value'].strip() + END_BOT)
        conversation += temp

    return conversation + '</s>'


class ImageProcessorHD:

    def __init__(self, resolution=560, hd_num=18):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)
        self.resolution = resolution
        self.hd_num = hd_num

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        item = Image.open(item).convert('RGB')
        return self.transform(
            R560_HD18_Identity_transform(
                item, resolution=self.resolution, hd_num=self.hd_num))


class Mix_dataset(Dataset):

    def __init__(self,
                 json_datas,
                 batch_size=1,
                 local_rank=0,
                 resolution=560,
                 hd_num=18):
        """vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file."""
        super().__init__()
        print(f'init mix data at rank {local_rank}')
        self.datasets_text, self.datasets_multi = [], []
        self.data_num_text, self.data_num_multi = [], []

        self.batch_size = batch_size
        self.set_seed = False
        self.local_rank = local_rank
        for _, d in json_datas.items():
            if 'image' in d[0].keys():
                has_img = True
            else:
                has_img = False
            sub_data_set = Sample_dataset(
                d, batch_size, has_img=has_img, hd_num=hd_num)
            if has_img:
                self.datasets_multi.append(sub_data_set)
                self.data_num_multi.append(len(sub_data_set))
            else:
                self.datasets_text.append(sub_data_set)
                self.data_num_text.append(len(sub_data_set))

        self.data_ratio_multi = [
            float(ratio) / sum(self.data_num_multi)
            for ratio in self.data_num_multi
        ]
        self.data_ratio_text = [
            float(ratio) / sum(self.data_num_text)
            for ratio in self.data_num_text
        ]
        self.data_num = np.sum(self.data_num_multi) + np.sum(
            self.data_num_text)
        self.use_multi = 0

    def __len__(self):
        return int(np.sum(self.data_num) / self.batch_size)

    def __getitem__(self, index):
        if not self.set_seed:
            random.seed(index)
            self.set_seed = True
            print(f'Set seed {index} for rank {self.local_rank}')

        if len(self.datasets_multi) == 0 and len(self.datasets_text) == 0:
            raise ValueError(
                'Both _multi and _text are empty. Cannot sample any data.')

        if len(self.datasets_multi) > 0 and (self.use_multi < self.batch_size
                                             or len(self.datasets_text) == 0):
            data_idx = random.choices(
                range(len(self.data_ratio_multi)),
                weights=self.data_ratio_multi,
                k=1)[0]
            sample = self.datasets_multi[data_idx].get_item()
        elif len(self.datasets_text) > 0:
            data_idx = random.choices(
                range(len(self.data_ratio_text)),
                weights=self.data_ratio_text,
                k=1)[0]
            sample = self.datasets_text[data_idx].get_item()
        else:
            raise ValueError('Unable to select a dataset for sampling.')

        self.use_multi += 1
        if self.use_multi > self.batch_size * 2:
            self.use_multi = 0
        return dict(samples=sample)


class Sample_dataset(Dataset):

    def __init__(self,
                 raw_data,
                 batch_size,
                 has_img=True,
                 resolution=560,
                 hd_num=18):
        self.raw_data = raw_data
        print(f'load {len(self.raw_data)} data')
        self.batch_size = batch_size
        self.vis_processor = ImageProcessorHD(
            resolution=resolution, hd_num=hd_num)
        self.text_processor = conv2text
        self.has_img = has_img

    def __len__(self):
        return len(self.raw_data)

    def __get_item__(self, i):
        conv_text = conv2text(self.raw_data[i]['conversations'])
        sample = dict(text_input=conv_text, )
        if self.has_img:
            image_file = self.raw_data[i]['image']
            if type(image_file) == str:
                image = self.vis_processor(image_file)
            elif type(image_file) == list:
                image = [self.vis_processor(i) for i in image_file]
            else:
                raise NotImplementedError('Image format not supported')
            sample['image'] = image
        else:
            sample['image'] = None

        return sample

    def get_item(self, ):
        text_input = []
        images = []
        for i in range(self.batch_size):
            idx = random.randrange(len(self.raw_data))
            sample = self.__get_item__(idx)
            text_input.append(sample['text_input'])
            if sample['image'] is None:
                pass
            else:
                images_batch = []
                if type(sample['image']) is list:
                    for im in sample['image']:
                        images_batch.append(im.unsqueeze(0))
                else:
                    images_batch.append(sample['image'].unsqueeze(0))
                images.append(images_batch)
        sample = {
            'text_input': text_input,
            'data_type': 'multi' if self.has_img else 'text',
        }
        if self.has_img:
            sample['image'] = images
        return sample
