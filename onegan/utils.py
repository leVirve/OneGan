# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import uuid
from collections import defaultdict

import torch
import tensorboardX


class History():

    def __init__(self, length):
        self.len = length
        self.losses = defaultdict(float)
        self.accuracies = defaultdict(float)

    def add(self, losses, accuracies, log_suffix=''):
        display = {}
        for name, acc in accuracies.items():
            self.accuracies[f'{name}{log_suffix}'] += acc
            display[name] = '%.02f' % acc
        for name, loss in losses.items():
            self.losses[f'{name}{log_suffix}'] += loss.data[0]
            display[name] = '%.03f' % loss.data[0]
        return display

    def metric(self):
        terms = {name: acc / self.len for name, acc in self.accuracies.items()}
        terms.update({name: loss / self.len for name, loss in self.losses.items()})
        return terms


class Checkpoint():

    def __init__(self, save_epochs=20):
        self.save_epochs = save_epochs

    def register_trainer(self, trainer):
        self.trainer = trainer
        self.root = 'output/weight/%s' % trainer.name
        os.makedirs(self.root, exist_ok=True)

    def load(self, net_path=None, resume=False):
        epoch = self._load(net_path, self.trainer.model, self.trainer.optim)
        if resume:
            self.trainer.start_epoch = epoch

    def save(self, epoch):
        if (epoch + 1) % self.save_epochs:
            return
        self._save(f'net-{epoch}.pth', self.trainer.model, self.trainer.optim, epoch)

    def _load(self, path, model, optim):
        if not path:
            return None
        ckpt = torch.load(path)
        assert ckpt['arch'] == model.__class__.__name__
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        return ckpt['epoch']

    def _save(self, name, model, optim, epoch):
        path = os.path.join(self.root, name)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch + 1,
            'arch': model.__class__.__name__
        }, path)


class GANCheckpoint(Checkpoint):

    def __init__(self, save_epochs=20):
        super().__init__(save_epochs)

    def load(self, gnet_path=None, dnet_path=None, resume=False):
        epoch = self._load(dnet_path, self.trainer.dnet, self.trainer.d_optim)
        epoch = self._load(gnet_path, self.trainer.gnet, self.trainer.g_optim)
        if resume:
            self.trainer.start_epoch = epoch

    def save(self, epoch):
        if (epoch + 1) % self.save_epochs:
            return
        name = '%%s-%d.pth' % (epoch)
        self._save(name % 'dnet', self.trainer.dnet, self.trainer.d_optim, epoch)
        self._save(name % 'gnet', self.trainer.gnet, self.trainer.g_optim, epoch)


class Logger(object):

    def __init__(self, log_root='logs', name=None):
        self.logger_dir = self.get_unique_dir(log_root, name)
        self.summary_num_images = 0
        self.image_base_tag_counter = 0
        self._phase_state = None

    def scalar(self, kw_scalars, step):
        [self.writer.add_scalar(tag, value, step) for tag, value in kw_scalars.items()]

    def image(self, kw_images, step, prefix):
        if self._phase_state != prefix:
            self.image_base_tag_counter = 0
            self._phase_state = prefix

        if self.image_base_tag_counter >= self.summary_num_images:
            return

        [self.writer.add_image(f'{prefix}{tag}/{self.image_base_tag_counter + i}', img_normalize(image), step)
         for tag, images in kw_images.items()
         for i, image in enumerate(images)]

        num_summaried_img = len(next(iter(kw_images.values())))
        self.image_base_tag_counter += num_summaried_img

    @property
    def writer(self):
        if not hasattr(self, '_writer'):
            self._writer = tensorboardX.SummaryWriter(self.logger_dir)
        return self._writer

    def get_unique_dir(self, log_root, name):
        target_path = os.path.join(log_root, name)
        if os.path.exists(target_path):
            name = f'{name}_' + uuid.uuid4().hex[:6]
        return os.path.join(log_root, name)


def img_normalize(img):
    mm, mx = img.min(), img.max()
    return img if mm == mx else img.add_(-mm).div_(mx - mm)
