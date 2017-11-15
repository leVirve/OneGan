# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tqdm
from torch.autograd import Variable

from onegan.utils import History, Logger
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
            source, target = Variable(source).cuda(), Variable(target).cuda()
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
            source, target = Variable(source, volatile=True).cuda(), Variable(target, volatile=True).cuda()
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
