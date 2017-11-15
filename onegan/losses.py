# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from torch.autograd import Variable, grad


class AdversarialLoss():

    def __init__(self, dnet, real_label=1, fake_label=0, use_lsgan=True):
        self.dnet = dnet
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def __call__(self, x, real_fake):
        x = self.dnet(x)
        value = self.real_label if real_fake else self.fake_label
        label = Variable(torch.FloatTensor(x.size()).fill_(value).cuda(), requires_grad=False)
        return self.loss(x, label)


class GradientPaneltyLoss():

    def __init__(self, dnet):
        self.dnet = dnet

    def __call__(self, target, pred):
        batch_size = target.size(0)
        alpha = (torch.rand(batch_size, 1)
                      .expand(batch_size, target.nelement() // batch_size)
                      .contiguous()
                      .view(target.size()).cuda())
        interp = Variable(alpha * target.data + (1 - alpha) * pred.data, requires_grad=True)

        output = self.dnet(interp)
        grads = grad(outputs=output, inputs=interp,
                     grad_outputs=torch.ones(output.size()).cuda(),
                     create_graph=True, retain_graph=True)[0]

        return ((grads.view(batch_size, -1).norm(dim=1) - 1) ** 2).mean()


class CombinedLossMixin():

    def add_term(self, name, criterion, weight):
        if isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
            self.smooth_name = name
            self.smooth_criterion = criterion
            self.smooth_weight = weight
        elif isinstance(criterion, AdversarialLoss):
            self.adv_name = name
            self.adv_criterion = criterion
            self.adv_weight = weight
        elif isinstance(criterion, GradientPaneltyLoss):
            self.gp_name = name
            self.gp_criterion = criterion
            self.gp_weight = weight
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
