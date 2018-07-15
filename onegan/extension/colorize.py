# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import numpy as np

from .base import Extension


class Colorizer(Extension):
    # TODO: should move to visualizer.image?

    def __init__(self, colors, num_output_channel=3):
        self.colors = self.normalized_color(colors)
        self.num_label = len(colors)
        self.num_channel = num_output_channel

    @staticmethod
    def normalized_color(colors):
        colors = np.array(colors, 'float32')
        if colors.max() > 1:
            colors = colors / 255
        return colors

    def apply(self, label):
        if label.dim() == 4:
            label = label.squeeze(1)
        assert label.dim() == 3
        n, h, w = label.size()

        canvas = torch.zeros(n, h, w, self.num_channel)
        for lbl_id in range(self.num_label):
            if canvas[label == lbl_id].size(0):
                canvas[label == lbl_id] = torch.from_numpy(self.colors[lbl_id])

        return canvas.permute(0, 3, 1, 2)
