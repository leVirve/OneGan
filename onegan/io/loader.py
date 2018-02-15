# Copyright (c) 2017-present Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os
import glob
import logging

import torch
from PIL import Image
from torchvision.datasets.folder import is_image_file


__all__ = [
    'BaseDataset', 'SourceToTargetDataset',
    'load_image', 'collect_images', 'universal_collate_fn'
]

default_collate = torch.utils.data.dataloader.default_collate


def load_image(path):
    return Image.open(path)


def collect_images(path):
    return sorted([e for e in glob.glob(os.path.join(path, '*.*')) if is_image_file(e)])


def universal_collate_fn(batch):

    def _collate(data):
        try:
            return default_collate(data)
        except RuntimeError:
            return data

    return {key: _collate([d[key] for d in batch]) for key in batch[0]}


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, phase, args=None):
        """ Base dataset with to_loader method

        Args:
            phase (str): should be `train` or `val` to indicate the phase
            args (argparse.Namespace): parsed arguments from onegan.option.Parser
        """
        self.phase = phase
        self.args = args

    @property
    def logger(self):
        """ :Logger: logger for specific succeeding class """
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(type(self).__name__)
        return self._logger

    def _split_data(self, files, phase, debug=False):
        if debug:
            num_split = int(len(files) * 0.1)
            files = files[:num_split]
        num_split = int(len(files) * 0.8)
        files = files[:num_split] if phase == 'train' else files[num_split:]
        return files

    def to_loader(self, **kwargs):
        """ Dispatch method for torch.utils.data.DataLoader

        Args:
            **kwargs: args for DataLoader()
        """

        # default settings
        params = {
            'pin_memory': True,
            'shuffle': self.phase == 'train',
            'batch_size': self.args.batch_size if self.args else 0,
            'num_workers': self.args.worker if self.args else 0,
        }
        params.update(kwargs)

        return torch.utils.data.DataLoader(self, **params)


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
