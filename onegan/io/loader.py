# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os
import glob
import logging

from torch.utils import data
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS


__all__ = [
    'BaseDataset', 'collect_images', 'universal_collate_fn'
]

default_collate = data.dataloader.default_collate

_log = logging.getLogger('onegan.io')


def collect_images(path):
    return sorted([
        e for e in glob.glob(os.path.join(path, '*.*'))
        if has_file_allowed_extension(e, IMG_EXTENSIONS)])


def universal_collate_fn(batch):

    def _collate(data):
        try:
            return default_collate(data)
        except RuntimeError:
            return data

    return {key: _collate([d[key] for d in batch]) for key in batch[0]}


class BaseDataset(data.Dataset):

    def __init__(self, phase=None, args=None):
        """ Base dataset with to_loader method

        Args:
            phase (str): should be `train` or `val` to indicate the phase
            args (argparse.Namespace): parsed arguments from onegan.option.Parser
        """
        self.phase = phase
        self.args = args
        _log.debug(f'BaseDataset <Initialized: phase={phase}>')

    def to_loader(self, **kwargs):
        """ Dispatch method for torch.utils.data.DataLoader

        Args:
            **kwargs: args for DataLoader()
        """

        # default settings
        params = {
            'pin_memory': True,
            'shuffle': self.phase == 'train' if self.phase else False,
            'batch_size': self.args.batch_size if self.args else 0,
            'num_workers': self.args.worker if self.args else 0,
        }
        params.update(kwargs)

        return data.DataLoader(self, **params)
