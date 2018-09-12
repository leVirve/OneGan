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
    """Deprecated."""

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
        return functional.pil_open(path)

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


class StateCompose(T.Compose):
    """Composes several transforms together with state-awareness.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.random_state = None

    def __call__(self, target, random_state=None):
        """Apply all transforms onto target with specific random state.
        """
        self.random_state = random.getstate()
        if random_state:
            random.setstate(random_state)

        for t in self.transforms:
            if not random and 'random' in t.__class__.__name__.lower():
                continue
            target = t(target)
        return target


class LoadPILImage(object):
    """Open a image file into PIL Image.

    Args:
    mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
        If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
        1. If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
        2. If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
        3. If the input has 1 channel, the ``mode`` is determined by the data type (i,e,
        ``int``, ``float``, ``short``).
    """

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, path):
        return functional.pil_open(path).convert(self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int or str, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR`` (or ``'bilinear'``)
    """

    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return functional.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = T._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
