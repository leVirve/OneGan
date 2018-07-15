# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torchvision.transforms as T
from PIL import Image


interpolations = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
}


def load_image(path):
    return Image.open(path)


def image_resize(x, target_size, mode='bilinear'):
    return T.functional.resize(x, target_size, interpolations[mode])
