# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from . import functional


class TransformPipeline:

    def __init__(self, target_size=None, color_jitter=None):
        self.target_size = target_size
        self.color_jitter = color_jitter or \
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def new_random_state(self):
        self.random = random.random() > 0.5
        self.random_angle = np.random.uniform(-5, 5)

    def _transform(self, x, transform_fn):
        if not self.random:
            return x
        return transform_fn(x)

    def load_image(self, path):
        return functional.load_image(path)

    def resize(self, x, mode='bilinear'):
        return functional.image_resize(x, self.target_size, mode)

    def colorjiiter(self, x):
        return self.color_jitter(x)

    def fliplr(self, x, func=None):
        if func:
            fn = func
        elif 'numpy' == type(x).__module__:
            fn = np.fliplr
        elif isinstance(x, Image.Image):
            fn = T.functional.hflip
        return self._transform(x, fn)

    def rotate(self, x):
        return self._transform(x, lambda x: T.functional.rotate(x, self.random_angle))

    def to_tensor(self,
                  x,
                  im2float=True,
                  normalize=True, mean=[.5, .5, .5], std=[.5, .5, .5]):

        if not im2float:
            # fake the data type as float32 to `T.functional.to_tensor()`
            x = np.array(x, dtype='f')

        if isinstance(x, np.ndarray) and x.ndim == 2:
            x = np.expand_dims(x, axis=2)

        y = T.functional.to_tensor(x)

        if im2float and normalize:
            y = F.normalize(y, mean, std)

        return y if im2float else y.long()
