# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn

import onegan
from onegan.external import pix2pix
from onegan.utils import to_var, to_device, cuda_available


conditional = True
g = pix2pix.define_G(3, 3, 64, 'unet_128', init_type='xavier')
d = pix2pix.define_D(6 if conditional else 3, 64, 'basic', init_type='xavier')
gan_criterion = (onegan.losses.GANLoss(conditional)
                 .add_term('smooth', nn.L1Loss(), weight=100)
                 .add_term('adv', onegan.losses.AdversarialLoss(d), weight=1)
                 .add_term('gp', onegan.losses.GradientPaneltyLoss(d), weight=0.25))

if cuda_available:
    g, d = g.cuda(), d.cuda()


def test_loss_terms():
    source = to_device(torch.FloatTensor(10, 3, 128, 128))
    target = to_device(torch.FloatTensor(10, 3, 128, 128))
    source, target = to_var(source), to_var(target)
    output = g(source)

    g_loss, g_terms = gan_criterion.g_loss(source, output, target)
    d_loss, d_terms = gan_criterion.d_loss(source, output, target)

    assert g_terms.keys()
    assert d_terms.keys()
