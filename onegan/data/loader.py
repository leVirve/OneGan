# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os
import glob

import torch
import scipy.misc
from PIL import Image
from torchvision.datasets.folder import is_image_file


def load_image(path):
    return Image.open(path)


def collect_images(path):
    return sorted([e for e in glob.glob(os.path.join(path, '*.*')) if is_image_file(e)])


def save_batched_images(img_tensors, folder=None, filenames=None):
    os.makedirs(folder, exist_ok=True)

    for fname, img in zip(filenames, img_tensors):
        path = os.path.join(folder, '%s.png' % fname)
        scipy.misc.imsave(path, img)


class BaseDastaset(torch.utils.data.Dataset):

    def initialize(self, phase):
        if self.debug:
            num_split = int(len(self.sources) * 0.1)
            self.sources = self.sources[:num_split]
            self.targets = self.targets[:num_split]

        num_split = int(len(self.sources) * 0.8)
        if phase == 'train':
            self.sources = self.sources[:num_split]
            self.targets = self.targets[:num_split]
        else:
            self.sources = self.sources[num_split:]
            self.targets = self.targets[num_split:]
        return self


def create_dataloader(dataset, phase, **kwargs):
    shuffle = phase == 'train'
    return torch.utils.data.DataLoader(dataset.initialize(phase), shuffle=shuffle, **kwargs)
