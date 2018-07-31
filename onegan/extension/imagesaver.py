# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import pathlib

import scipy.misc

from onegan.visualizer.image import img_normalize
from .base import Extension, unique_experiment_name


class ImageSaver(Extension):
    """ Smarter batched image saver

    Args:
        savedir (str): root folder for image saving (default: `exp/results/`)
        name (str): subfolder name for current experiment (default: `default`)
    """

    def __init__(self, savedir='exp/results/', name='default'):
        self.root_savedir = savedir
        self.name = name

    @property
    def savedir(self):
        if not hasattr(self, '_savedir'):
            self._savedir = unique_experiment_name(self.root_savedir, self.name)
            os.makedirs(self._savedir, exist_ok=True)
        return pathlib.Path(self._savedir)

    def image(self, img_tensors, filenames, normalized=True):
        """
        Args:
            img_tensors: batched tensor [batch, (channel,) height, width]
            filenames: corresponding batched (list of) filename strings
            normalized: whether to normalize the image before saving
        """
        if img_tensors.dim() == 4:
            img_tensors = img_tensors.permute(0, 2, 3, 1)

        for fname, img in zip(filenames, img_tensors):
            name, ext = os.path.splitext(fname)
            ext = '.png' if ext not in ['.png', '.jpg'] else ext
            path = os.path.join(self.savedir, name + ext)

            if normalized:
                img = img_normalize(img)

            scipy.misc.imsave(path, img.cpu().numpy())
