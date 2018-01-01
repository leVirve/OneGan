# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os
import glob

import torch
from PIL import Image
from torchvision.datasets.folder import is_image_file


__all__ = [
    'BaseDataset', 'SourceToTargetDataset',
    'load_image', 'collect_images'
]


def load_image(path):
    return Image.open(path)


def collect_images(path):
    return sorted([e for e in glob.glob(os.path.join(path, '*.*')) if is_image_file(e)])


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, phase):
        self.phase = phase

    def _split_data(self, files, phase, debug=False):
        if debug:
            num_split = int(len(files) * 0.1)
            files = files[:num_split]
        num_split = int(len(files) * 0.8)
        files = files[:num_split] if phase == 'train' else files[num_split:]
        return files

    def to_loader(self, pin_memory=True, **kwargs):
        shuffle = self.phase == 'train'
        return torch.utils.data.DataLoader(
            self, shuffle=shuffle, pin_memory=pin_memory, **kwargs)


class SourceToTargetDataset(BaseDataset):

    def __init__(self, phase, source_folder, target_folder, transform=None, debug=False):
        super().__init__(phase)
        self.sources = self._split_data(collect_images(source_folder), phase, debug=debug)
        self.targets = self._split_data(collect_images(target_folder), phase, debug=debug)
        assert len(self.sources) == len(self.targets)
        self.transform = transform

    def __getitem__(self, index):
        source = load_image(self.sources[index]).convert('RGB')
        target = load_image(self.targets[index]).convert('RGB')
        return self.transform(source), self.transform(target)

    def __len__(self):
        return len(self.sources)
