# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch

import onegan
from onegan.utils import to_var


def normalize(x):
    mm, mx = x.min(), x.max()
    x = x.add_(-mm).div_(mx - mm)
    x = x.add_(-0.5).div_(0.5)
    return x


def test_psnr():
    dummy_output = to_var(normalize(torch.rand(10, 3, 128, 128)))
    dummy_target = to_var(normalize(torch.rand(10, 3, 128, 128)))

    accuracy = onegan.metrics.Psnr()
    psnr = accuracy(dummy_output, dummy_target)
    assert psnr
