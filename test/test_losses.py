# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn.functional as F

import onegan.losses as losses
from onegan.external import pix2pix
from onegan.utils import to_var, to_device, cuda_available


conditional = True
g = pix2pix.define_G(3, 3, 64, 'unet_128', init_type='xavier')
d = pix2pix.define_D(6 if conditional else 3, 64, 'basic', init_type='xavier')
if cuda_available:
    g, d = g.cuda(), d.cuda()


def test_loss_terms():
    source = to_device(torch.FloatTensor(10, 3, 128, 128))
    target = to_device(torch.FloatTensor(10, 3, 128, 128))
    source, target = to_var(source), to_var(target)
    output = g(source)

    real = losses.conditional_input(source, target, conditional)
    fake = losses.conditional_input(source, output, conditional)

    losses.adversarial_ce_loss(F.sigmoid(d(fake)), 1)
    losses.adversarial_ls_loss(d(fake), 1)
    losses.adversarial_w_loss(d(fake), True)
    losses.gradient_penalty(d, real, fake)
