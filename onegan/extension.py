# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
from collections import defaultdict

import torch
from torch.autograd import Variable
import tensorboardX

from onegan.utils import unique_experiment_name, img_normalize


def check_state(f):
    def wrapper(instance, kw_images, epoch, prefix):
        if instance._phase_state != prefix:
            instance._tag_base_counter = 0
            instance._phase_state = prefix
        return f(instance, kw_images, epoch, prefix)
    return wrapper


def check_num_images(f):
    def wrapper(instance, kw_images, epoch, prefix):
        if instance._tag_base_counter >= instance.max_num_images:
            return
        num_summaried_img = len(next(iter(kw_images.values())))
        result = f(instance, kw_images, epoch, prefix)
        instance._tag_base_counter += num_summaried_img
        return result
    return wrapper


class Extension:
    pass


class Logger(Extension):

    def __init__(self, logdir='logs', name='default', max_num_images=20):
        self.logdir = unique_experiment_name(logdir, name)
        self.max_num_images = max_num_images
        self._tag_base_counter = 0
        self._phase_state = None

    def scalar(self, kw_scalars, epoch):
        [self.writer.add_scalar(tag, value, epoch) for tag, value in kw_scalars.items()]

    @check_state
    @check_num_images
    def image(self, kw_images, epoch, prefix):
        [self.writer.add_image(f'{prefix}{tag}/{self._tag_base_counter + i}', img_normalize(image), epoch)
         for tag, images in kw_images.items()
         for i, image in enumerate(images)]

    def clear_state(self):
        self.image_base_tag_counter = 0
        self._phase_state = None

    @property
    def writer(self):
        if not hasattr(self, '_writer'):
            self._writer = tensorboardX.SummaryWriter(self.logdir)
        return self._writer


class History(Extension):

    def __init__(self):
        self.count = 0
        self.meters = defaultdict(float)

    def add(self, kwvalues, n=1, log_suffix=''):
        display = {}
        for name, value in kwvalues.items():
            val = value.data[0] if isinstance(value, Variable) else value
            self.meters[f'{name}{log_suffix}'] += val
            display[name] = f'{val:.03f}'
        self.count += n
        return display

    def metric(self):
        return {name: val / self.count for name, val in self.meters.items()}


class Checkpoint(Extension):

    def __init__(self, savedir='output/checkpoint/', name='default', save_epochs=20):
        self.root_savedir = savedir
        self.name = name
        self.save_epochs = save_epochs

    @property
    def savedir(self):
        if not hasattr(self, '_savedir'):
            self._savedir = unique_experiment_name(self.root_savedir, self.name)
            os.makedirs(self._savedir, exist_ok=True)
        return self._savedir

    def load(self, trainer, net_path=None, resume=False):
        epoch = self._load(net_path, trainer.model, trainer.optim)
        if resume:
            trainer.start_epoch = epoch

    def save(self, trainer, epoch):
        if (epoch + 1) % self.save_epochs:
            return
        self._save(f'net-{epoch}.pth', trainer.model, trainer.optim, epoch)

    def _load(self, path, model, optim):
        if not path:
            return None
        ckpt = torch.load(path)
        assert ckpt['arch'] == model.__class__.__name__
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        return ckpt['epoch']

    def _save(self, name, model, optim, epoch):
        path = os.path.join(self.savedir, name)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch + 1,
            'arch': model.__class__.__name__
        }, path)


class GANCheckpoint(Checkpoint):

    def load(self, trainer, gnet_path=None, dnet_path=None, resume=False):
        epoch = self._load(dnet_path, trainer.dnet, trainer.d_optim)
        epoch = self._load(gnet_path, trainer.gnet, trainer.g_optim)
        if resume:
            trainer.start_epoch = epoch

    def save(self, trainer, epoch):
        if (epoch + 1) % self.save_epochs:
            return
        name = '%%s-%d.pth' % (epoch)
        self._save(name % 'dnet', trainer.dnet, trainer.d_optim, epoch)
        self._save(name % 'gnet', trainer.gnet, trainer.g_optim, epoch)
