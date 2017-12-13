# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from torch.autograd import Variable, grad

from onegan.utils import to_var, to_device


def l1_loss(x, y):
    return nn.functional.l1_loss(x, y)


def adversarial_ce_loss(x, value: float):
    ''' x: output tensor of discriminator
        value: float
    '''
    label = to_var(torch.FloatTensor(x.size()).fill_(value), requires_grad=False)
    return nn.functional.binary_cross_entropy(x, label)


def adversarial_ls_loss(x, value: float):
    ''' x: output tensor of discriminator
        value: float
    '''
    label = to_var(torch.FloatTensor(x.size()).fill_(value), requires_grad=False)
    return nn.functional.mse_loss(x, label)


def adversarial_w_loss(x, value: bool):
    ''' x: output tensor of discriminator
        value: True -> -1, False -> 1
    '''
    return -torch.mean(x) if value else torch.mean(x)


def gradient_penalty(dnet, target, pred):
    alpha = to_device(torch.rand(target.size(0), 1, 1, 1)).expand_as(target)
    interp = Variable(alpha * target.data + (1 - alpha) * pred.data, requires_grad=True)

    output = dnet(interp)
    grads = grad(outputs=output, inputs=interp,
                 grad_outputs=to_device(torch.ones(output.size())),
                 create_graph=True, retain_graph=True)[0]

    return ((grads.view(grads.size(0), -1).norm(dim=1) - 1) ** 2).mean()


def conditional_input(source, another, conditional):
    return torch.cat((source, another), dim=1) if conditional else another
