# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tqdm

import onegan.losses as losses
from onegan.utils import History, Logger, to_var
from onegan.extensions import ImageSummaryExtention, HistoryExtention


class Estimator():

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optim = optimizer
        self.criterion = criterion

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


class GANEstimator(Estimator):

    def __init__(self, model, optimizer, criterion):
        self.gnet, self.dnet = model
        self.g_optim, self.d_optim = optimizer
        self.criterion = criterion


class OneGANEstimator(GANEstimator):

    def __init__(self, model, optimizer, criterion, metric, saver, name):
        super().__init__(model, optimizer, criterion)
        self.metric = metric
        self.saver = saver
        self.name = name

        logger = Logger(name=name)
        self.summary_history = HistoryExtention(logger)
        self.summary_image = ImageSummaryExtention(logger, summary_num_images=30)
        self.saver.register_trainer(self)

    def train(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader)
        for i, (source, target) in enumerate(progress):
            source, target = to_var(source), to_var(target)
            output = self.gnet(source)

            d_loss, d_terms = self.criterion.d_loss(source=source, output=output.detach(), target=target)
            g_loss, g_terms = self.criterion.g_loss(source=source, output=output, target=target)
            acc_terms = self.metric(output, target)

            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            progress.set_description('Epoch#%d' % (epoch + 1))
            progress.set_postfix(history.add({**g_terms, **d_terms}, {**acc_terms}))

            self.summary_image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='train_')

        self.summary_history(history.metric(), epoch)

    def evaluate(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader, leave=False)
        for i, (source, target) in enumerate(progress):
            source, target = to_var(source, volatile=True), to_var(target, volatile=True)
            output = self.gnet(source)

            _, d_terms = self.criterion.d_loss(source=source, output=output, target=target)
            _, g_terms = self.criterion.g_loss(source=source, output=output, target=target)
            acc_terms = self.metric(output, target)

            progress.set_description('Evaluate')
            progress.set_postfix(history.add({**g_terms, **d_terms}, {**acc_terms}, log_suffix='_val'))

            self.summary_image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='val_')

        self.summary_history(history.metric(), epoch)


class OneWGANEstimator(GANEstimator):

    def __init__(self, model, optimizer, metric, saver, name):
        super().__init__(model, optimizer, None)
        self.metric = metric
        self.saver = saver
        self.name = name
        self.critic_iters = 5

        logger = Logger(name=name)
        self.summary_history = HistoryExtention(logger)
        self.summary_image = ImageSummaryExtention(logger, summary_num_images=30)
        self.saver.register_trainer(self)

    def foward_d(self, source, output, target):
        fake = losses.conditional_input(source, output, self.conditional)
        real = losses.conditional_input(source, target, self.conditional)
        d_fake = losses.adversarial_w_loss(self.dnet(fake), False)
        d_real = losses.adversarial_w_loss(self.dnet(real), True)
        d_gp = losses.gradient_panelty(self.dnet, real, fake)
        d_terms = {'d/real': d_real, 'd/fake': d_fake, 'd/gp': d_gp, 'd/loss': d_real + d_fake + 0.25 * d_gp}
        d_loss = d_terms['d/loss']
        return d_loss, d_terms

    def foward_g(self, source, output, target):
        l1 = losses.l1_loss(output, target)
        adv = losses.adversarial_w_loss(self.dnet(losses.conditional_input(source, output, self.conditional)), True)
        g_terms = {'g/smooth': l1, 'g/adv': adv, 'g/loss': l1 * 100 + adv}
        g_loss = g_terms['g/loss']
        return g_loss, g_terms

    def train(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader)

        def fetch_data():
            source, target = next(progress)
            return to_var(source), to_var(target)

        for _ in range(len(progress)):

            for _ in range(self.critic_iters):
                source, target = fetch_data()
                output = self.gnet(source).detach()
                d_loss, d_terms = self.foward_d(source, target, output)

                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

            source, target = fetch_data()
            output = self.gnet(source)
            g_loss, g_terms = self.foward_g(source, target, output)

            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            acc_terms = self.metric(output, target)
            progress.set_description('Epoch#%d' % (epoch + 1))
            progress.set_postfix(history.add({**g_terms, **d_terms}, {**acc_terms}))

            self.summary_image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='train_')

        self.summary_history(history.metric(), epoch)

    def evaluate(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader, leave=False)
        for i, (source, target) in enumerate(progress):
            source, target = to_var(source, volatile=True), to_var(target, volatile=True)
            output = self.gnet(source)

            _, d_terms = self.foward_d(source, target, output)
            _, g_terms = self.foward_g(source, target, output)
            acc_terms = self.metric(output, target)

            progress.set_description('Evaluate')
            progress.set_postfix(history.add({**g_terms, **d_terms}, {**acc_terms}, log_suffix='_val'))

            self.summary_image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='val_')

        self.summary_history(history.metric(), epoch)
