# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from math import log10

import numpy as np
import torch.nn as nn


class PsnrMixin():

    def __init__(self):
        self.mse = nn.MSELoss()

    def __call__(self, output, target):

        def normalize(img):
            mm, mx = img.min(), img.max()
            return img.add(-mm).div(mx - mm)

        psnrs = np.array([
            10 * log10(1 / self.mse(normalize(pred), normalize(targ)).data[0])
            for pred, targ in zip(output, target)])
        return {'acc/psnr': psnrs.mean()}


class AccuracyMetric():

    def __init__(self):
        pass
