# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torchvision.transforms as T
from PIL import Image


_str_to_pil_interpolations = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
}


def pil_open(path):
    return Image.open(path)


def _resize(x, size, interpolation='bilinear'):
    r"""Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int or str, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR`` (or ``'bilinear'``)
    Returns:
        PIL Image: Resized image.
    """
    mode = interpolation
    if isinstance(interpolation, str):
        mode = _str_to_pil_interpolations[interpolation]

    return T.functional.resize(x, size, mode)


resize = _resize
