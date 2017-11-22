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

    def _initialize(self, files, phase):
        self.phase = phase
        if self.debug:
            num_split = int(len(files) * 0.1)
            files = files[:num_split]
        num_split = int(len(files) * 0.8)
        files = files[:num_split] if phase == 'train' else files[num_split:]
        return files

    def to_loader(self, phase=None, **kwargs):
        self.initialize(phase)
        loader = torch.utils.data.DataLoader(self, shuffle=phase == 'train', **kwargs)
        return loader
