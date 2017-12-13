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


def a_loss(loss_terms: dict):
    loss = [v for k, v in loss_terms.items()]
    return loss


class AdversarialLoss():

    def __init__(self, dnet, real_label=1, fake_label=0, use_lsgan=True):
        self.dnet = dnet
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def __call__(self, x, real_fake):
        x = self.dnet(x)
        value = self.real_label if real_fake else self.fake_label
        label = to_var(torch.FloatTensor(x.size()).fill_(value), requires_grad=False)
        return self.loss(x, label)


class GradientPaneltyLoss():

    def __init__(self, dnet):
        self.dnet = dnet

    def __call__(self, target, pred):
        batch_size = target.size(0)
        alpha = to_device(torch.rand(batch_size, 1)
                          .expand(batch_size, target.nelement() // batch_size)
                          .contiguous()
                          .view(target.size()))
        interp = Variable(alpha * target.data + (1 - alpha) * pred.data, requires_grad=True)

        output = self.dnet(interp)
        grads = grad(outputs=output, inputs=interp,
                     grad_outputs=to_device(torch.ones(output.size())),
                     create_graph=True, retain_graph=True)[0]

        return ((grads.view(batch_size, -1).norm(dim=1) - 1) ** 2).mean()


class CombinedLossMixin():

    def add_term(self, name, criterion, weight):
        if isinstance(criterion, AdversarialLoss):
            self.adv_name = name
            self.adv_criterion = criterion
            self.adv_weight = weight
        elif isinstance(criterion, GradientPaneltyLoss):
            self.gp_name = name
            self.gp_criterion = criterion
            self.gp_weight = weight
        else:
            self.smooth_name = name
            self.smooth_criterion = criterion
            self.smooth_weight = weight
        return self


class GANLoss(CombinedLossMixin):

    def __init__(self, conditional=False):
        self.conditional = conditional

    def conditional_input(self, source, another):
        return torch.cat((source, another), dim=1) if self.conditional else another

    def g_loss(self, source, output, target):
        fake = self.conditional_input(source, output)
        return self._g_loss(output, target, fake)

    def d_loss(self, source, output, target):
        fake = self.conditional_input(source, output)
        real = self.conditional_input(source, target)
        return self._d_loss(real, fake)

    def _g_loss(self, output, target, fake):
        smooth = self.smooth_criterion(output, target)
        adv = self.adv_criterion(fake, True)

        loss = self.smooth_weight * smooth + self.adv_weight * adv
        terms = {f'g/{self.adv_name}': adv,
                 f'g/{self.smooth_name}': smooth,
                 f'g/loss': loss}

        return loss, terms

    def _d_loss(self, real, fake):
        adv_real = self.adv_criterion(real, True)
        adv_fake = self.adv_criterion(fake, False)

        loss = (adv_real + adv_fake) * 0.5
        terms = {'d/real': adv_real,
                 'd/fake': adv_fake,
                 'd/loss': loss}

        if self.gp_weight:
            gp = self.gp_criterion(real, fake)
            loss += self.gp_weight * gp
            terms.update({f'd/{self.gp_name}': gp, 'd/loss': loss})

        return loss, terms
