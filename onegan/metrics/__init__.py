# Copyright (c) 2018 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from math import log10

import torch.nn.functional as F
import numpy as np

from onegan.metrics import semantic_segmentation  # noqa
from onegan.utils import img_normalize


def psnr(output, target, img_range=None):
    ''' calculate the PSNR
        output: Variable, range in (0, 1)
        target: Variable, range in (0, 1)
    '''

    def normalize(t):
        return img_normalize(t, img_range=img_range)

    psnrs = np.array([
        10 * log10(1 / F.mse_loss(normalize(pred), normalize(targ)).data[0] + 1e-9)
        for pred, targ in zip(output, target)])
    return psnrs.mean()
