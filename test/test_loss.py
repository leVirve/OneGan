# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn.functional as F

import onegan.loss as L
from onegan.external import pix2pix
from onegan.utils import device


conditional = True
g = pix2pix.define_G(3, 3, 64, 'unet_128', init_type='xavier').to(device())
d = pix2pix.define_D(6 if conditional else 3, 64, 'basic', init_type='xavier').to(device())


def test_loss_terms():
    source = torch.randn(10, 3, 128, 128, device=device())
    target = torch.randn(10, 3, 128, 128, device=device())
    output = g(source)

    real = L.conditional_input(source, target, conditional)
    fake = L.conditional_input(source, output, conditional)

    L.adversarial_ce_loss(F.sigmoid(d(fake)), 1)
    L.adversarial_ls_loss(d(fake), 1)
    L.adversarial_w_loss(d(fake), True)
    L.gradient_penalty(d, real, fake)
