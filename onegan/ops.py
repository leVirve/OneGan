# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import numpy as np
import torch.nn.functional as F


class VisionConv2d:

    def __init__(self, kernel, padding=1, dilation=1, name='VisionConv2d'):
        """
            Args:
                kernel: str or np.ndarray
        """
        if isinstance(kernel, str):
            kernel = {
                'laplacian': laplacian_kernel,
                'sobel_vertical': sobel_vertical_kernel,
                'sobel_horizontal': sobel_horizontal_kernel,
            }[kernel]()
        assert kernel.ndim == 2, 'Plain Vision Kernel should be 2D'
        self.kernel = torch.from_numpy(kernel[np.newaxis, np.newaxis, :])
        self.padding = padding
        self.dilation = dilation
        self.name = name

    def __call__(self, x: torch.tensor):
        assert x.dim() == 4, 'input tensor should be 4D'
        return F.conv2d(x, self.kernel.to(x), padding=self.padding, dilation=self.dilation)


class VisionConv3d:

    def __init__(self, kernel, channel=3, padding=1, dilation=1, name='VisionConv3d'):
        """
            Args:
                kernel: str or np.ndarray
        """
        self.vision_conv2d = VisionConv2d(kernel, padding, dilation, name)

    def __call__(self, x: torch.tensor):
        assert x.dim() == 4, 'input tensor should be 4D'

        in_channels = x.size(1)
        outputs = [
            self.vision_conv2d(x[:, i:i + 1, ...])
            for i in range(in_channels)
        ]
        return torch.cat(outputs, dim=1)


def laplacian_kernel():
    return np.array([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]], dtype='f')


def sobel_vertical_kernel():
    return np.array([[1, 2, 1],
                     [0,  0, 0],
                     [-1, -2, -1]], dtype='f')


def sobel_horizontal_kernel():
    return np.array([[1, 2, 1],
                     [0,  0, 0],
                     [-1, -2, -1]], dtype='f').T
