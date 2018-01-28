# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging

import tqdm

import onegan.losses as losses
from onegan.utils import to_var
from onegan.extension import History, TensorBoardLogger, Checkpoint, GANCheckpoint


class Estimator:

    def __init__(self, model, optimizer, metric, name, **kwargs):
        self.model = model
        self.optim = optimizer
        self.metric = metric
        self.name = name

        # TODO: make these extensions optional
        if self.name:
            self.saver = Checkpoint(name=name, save_epochs=kwargs.get('save_epochs', 5))
            self.logger = TensorBoardLogger(name=name, max_num_images=kwargs.get('max_num_images', 30))

    def run(self, train_loader, validate_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
            self.train(train_loader, epoch, History())
            self.model.eval()
            self.evaluate(validate_loader, epoch, History())
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        if not hasattr(self, 'saver'):
            return
        self.saver.save(self, epoch)


class OneEstimator:

    def __init__(self, model, optimizer=None, lr_scheduler=None, logger=None, saver=None, name=None):
        self.model = model
        self.optimizer = optimizer
        self.saver = saver
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.name = name

        self.history = History()
        self.state = {}
        self._log = logging.getLogger(f'OneGAN.{name}')
        self._log.info(f'OneEstimator<{name}> is initialized')

    def run(self, train_loader, validate_loader, update_fn, inference_fn, epochs):
        for epoch in range(epochs):
            self.state['epoch'] = epoch

            self.train(train_loader, update_fn)
            self.logger.scalar(self.history.metric(), epoch)

            self.evaluate(validate_loader, inference_fn)
            self.logger.scalar(self.history.metric(), epoch)

            self.save_checkpoint()
            self.adjust_learning_rate(self.history.metric()['loss/loss_val'])
            self._log.info(f'OneEstimator<{self.name}> epoch#{epoch} end')

    def load_checkpoint(self, weight_path, resume=False):
        if not hasattr(self, 'saver') or self.saver is None:
            return
        self.saver.load(self, weight_path, resume)

    def save_checkpoint(self):
        if not hasattr(self, 'saver') or self.saver is None:
            return
        self.saver.save(self, self.state['epoch'])

    def adjust_learning_rate(self, monitor_val):
        if not hasattr(self, 'lr_scheduler') or self.lr_scheduler is None:
            return
        self.lr_scheduler.step(monitor_val)

    def train(self, data_loader, update_fn):
        self.model.train()
        self.history.clear()

        progress = tqdm.tqdm(data_loader)
        progress.set_description(f'Epoch#{self.state["epoch"] + 1}')

        for data in progress:
            loss, accuracy = update_fn(self.model, data)
            progress.set_postfix(self.history.add({**loss, **accuracy}))
            self.optimizer.zero_grad()
            loss['loss/loss'].backward()
            self.optimizer.step()
        return self.history.metric()

    def evaluate(self, data_loader, inference_fn):
        self.model.eval()
        self.history.clear()

        progress = tqdm.tqdm(data_loader)
        progress.set_description('Evaluate')

        for data in progress:
            log_values = inference_fn(self.model, data)
            loss, accuracy = log_values if isinstance(log_values, tuple) else (log_values, {})
            progress.set_postfix(self.history.add({**loss, **accuracy}, log_suffix='_val'))
        return self.history.metric()


class OneGANEstimator(Estimator):

    def __init__(self, model, optimizer, metric, name, **kwargs):
        self.gnet, self.dnet = model
        self.g_optim, self.d_optim = optimizer
        self.metric = metric
        self.name = name

        self.saver = GANCheckpoint(name=name, save_epochs=kwargs.get('save_epochs', 5))
        self.logger = TensorBoardLogger(name=name, max_num_images=kwargs.get('max_num_images', 30))

        self.build_criterion()

    def build_criterion(self):
        self.recon_weight = 10
        self.recon_loss = losses.l1_loss
        self.adv_loss = losses.adversarial_ls_loss

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
            progress.set_postfix(history.add({**g_terms, **d_terms, 'acc/psnr': acc}))

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
            progress.set_postfix(history.add({**g_terms, **d_terms, 'acc/psnr': acc}, log_suffix='_val'))

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
            progress.set_postfix(history.add({**g_terms, **d_terms, 'acc/psnr': acc}))

            self.logger.image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='train_')

        self.logger.scalar(history.metric(), epoch)
