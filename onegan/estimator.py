# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tqdm

import onegan.losses as losses
from onegan.utils import to_var
from onegan.extensions import Logger, History


class Estimator:

    def __init__(self, model, optimizer):
        self.model = model
        self.optim = optimizer

    def run(self, train_loader, validate_loader, epochs):
        for epoch in range(epochs):
            self.gnet.train(), self.dnet.train()
            self.train(train_loader, epoch, History(length=len(train_loader)))
            self.gnet.eval(), self.dnet.eval()
            self.evaluate(validate_loader, epoch, History(length=len(validate_loader)))
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        if not hasattr(self, 'saver'):
            return
        self.saver.save(epoch)


class OneGANEstimator(Estimator):

    def __init__(self, model, optimizer, metric, saver, name):
        self.gnet, self.dnet = model
        self.g_optim, self.d_optim = optimizer
        self.metric = metric
        self.saver = saver
        self.name = name

        self.logger = Logger(name=name, max_num_images=30)
        self.saver.register_trainer(self)

        self.build_criterion()

    def build_criterion(self):
        self.recon_weight = 10
        self.recon_loss = losses.l1_loss
        self.adv_loss = losses.adversarial_ce_loss

    def foward_d(self, source, output, target):
        fake = losses.conditional_input(source, output, self.conditional)
        real = losses.conditional_input(source, target, self.conditional)
        d_fake = self.adv_loss(self.dnet(fake), False)
        d_real = self.adv_loss(self.dnet(real), True)
        d_terms = {'d/real': d_real, 'd/fake': d_fake, 'd/loss': d_real + d_fake}
        d_loss = d_terms['d/loss']
        return d_loss, d_terms

    def foward_g(self, source, output, target):
        recon = self.recon_loss(output, target)
        adv = self.adv_loss(self.dnet(losses.conditional_input(source, output, self.conditional)), True)
        g_terms = {'g/recon': recon, 'g/adv': adv, 'g/loss': recon * self.recon_weight + adv}
        g_loss = g_terms['g/loss']
        return g_loss, g_terms

    def train(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader)
        for i, (source, target) in enumerate(progress):
            source, target = to_var(source), to_var(target)
            output = self.gnet(source)

            d_loss, d_terms = self.foward_d(source, output.detach(), target)
            g_loss, g_terms = self.foward_g(source, output, target)
            acc = self.metric(output, target)

            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            progress.set_description('Epoch#%d' % (epoch + 1))
            progress.set_postfix(history.add({**g_terms, **d_terms}, {'acc/psnr': acc}))

            self.logger.image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='train_')

        self.logger.scalar(history.metric(), epoch)

    def evaluate(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader, leave=False)
        for i, (source, target) in enumerate(progress):
            source, target = to_var(source, volatile=True), to_var(target, volatile=True)
            output = self.gnet(source)

            _, d_terms = self.foward_d(source, output.detach(), target)
            _, g_terms = self.foward_g(source, output, target)
            acc = self.metric(output, target)

            progress.set_description('Evaluate')
            progress.set_postfix(history.add({**g_terms, **d_terms}, {'acc/psnr': acc}, log_suffix='_val'))

            self.logger.image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='val_')

        self.logger.scalar(history.metric(), epoch)


class OneWGANEstimator(OneGANEstimator):

    def __init__(self, model, optimizer, metric, saver, name):
        super().__init__(model, optimizer)
        self.critic_iters = 5

    def build_criterion(self):
        self.recon_weight = 10
        self.gp_weight = 10
        self.recon_loss = losses.l1_loss
        self.adv_loss = losses.adversarial_w_loss

    def foward_d(self, source, output, target):
        fake = losses.conditional_input(source, output, self.conditional)
        real = losses.conditional_input(source, target, self.conditional)
        d_fake = self.adv_loss(self.dnet(fake), False)
        d_real = self.adv_loss(self.dnet(real), True)
        d_gp = losses.gradient_panelty(self.dnet, real, fake)
        d_terms = {'d/real': d_real, 'd/fake': d_fake, 'd/gp': d_gp, 'd/loss': d_real + d_fake + self.gp_weight * d_gp}
        d_loss = d_terms['d/loss']
        return d_loss, d_terms

    def train(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader)

        def fetch_data():
            source, target = next(progress)
            return to_var(source), to_var(target)

        # TODO: need go through and modify to work
        for _ in range(len(progress)):

            for _ in range(self.critic_iters):
                source, target = fetch_data()
                output = self.gnet(source).detach()
                d_loss, d_terms = self.foward_d(source, output, target)

                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

            source, target = fetch_data()
            output = self.gnet(source)
            g_loss, g_terms = self.foward_g(source, output, target)

            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            acc = self.metric(output, target)
            progress.set_description('Epoch#%d' % (epoch + 1))
            progress.set_postfix(history.add({**g_terms, **d_terms}, {'acc/psnr': acc}))

            self.logger.image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='train_')

        self.logger.scalar(history.metric(), epoch)
