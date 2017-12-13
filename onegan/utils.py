# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import uuid

import torch
from torch.autograd import Variable

cuda_available = torch.cuda.is_available()


def to_device(x):
    if cuda_available:
        return x.cuda()
    return x


def to_var(x, **kwargs):
    var = Variable(x, **kwargs)
    if cuda_available:
        return var.cuda()
    return var


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


def get_unique_subfolder(root, name):
    target_path = os.path.join(root, name)
    if os.path.exists(target_path):
        name = f'{name}_' + uuid.uuid4().hex[:6]
    return os.path.join(root, name)


def img_normalize(img):
    mm, mx = img.min(), img.max()
    return img if mm == mx else img.add_(-mm).div_(mx - mm)
