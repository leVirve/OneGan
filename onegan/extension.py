# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.misc
import tensorboardX
import torch
from torch.autograd import Variable

from onegan.utils import (export_checkpoint_weight, img_normalize,
                          unique_experiment_name)


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


class TensorBoardLogger(Extension):

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
        '''
            Args:
                kw_images: 4-D tensor [batch, channel, height, width]
                epoch: step for TensorBoard logging
                prefix: prefix string for tag
        '''
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


class ImageSaver(Exception):

    def __init__(self, savedir='output/results/', name='default'):
        self.name = name
        self.savedir = unique_experiment_name(savedir, name)
        self._create_folder()

    def _create_folder(self):
        os.makedirs(self.savedir, exist_ok=True)

    def image(self, img_tensors, filenames):
        '''
            Args:
                img_tensors: batched tensor [batch, (channel,) height, width]
        '''
        if img_tensors.dim() == 4:
            img_tensors = img_tensors.permute(0, 2, 3, 1)

        for fname, img in zip(filenames, img_tensors):
            path = os.path.join(self.savedir, fname)
            scipy.misc.imsave(path, img.cpu().numpy())


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

    def clear(self):
        self.count = 0
        self.meters = defaultdict(float)


class WeightSearcher(Extension):

    def __init__(self, weight_path):
        self.weight_path = Path(weight_path)

    def get_weight(self):
        if self.weight_path.is_file():
            yield export_checkpoint_weight(self.weight_path, remove_module=False), self.weight_path
        for weight_path in self.weight_path.glob('*.pth'):
            yield export_checkpoint_weight(weight_path, remove_module=False), weight_path


class Checkpoint(Extension):

    def __init__(self, savedir='output/checkpoints/', name='default', save_epochs=20):
        self.root_savedir = savedir
        self.name = name
        self.save_epochs = save_epochs

    @property
    def savedir(self):
        if not hasattr(self, '_savedir'):
            self._savedir = unique_experiment_name(self.root_savedir, self.name)
            os.makedirs(self._savedir, exist_ok=True)
        return self._savedir

    def apply(weight_path, model):
        state_dict = export_checkpoint_weight(weight_path, remove_module=False)
        model.load_state_dict(state_dict)

    def load(self, trainer, net_path=None, resume=False):
        ckpt = self._load(net_path)
        assert ckpt['arch'] == trainer.model.__class__.__name__
        trainer.model.load_state_dict(ckpt['model'])
        if resume:
            trainer.optimizer.load_state_dict(ckpt['optimizer'])
            return ckpt['epoch']

    def save(self, trainer, epoch):
        if (epoch + 1) % self.save_epochs:
            return
        self._save(f'net-{epoch}.pth', trainer.model, trainer.optimizer, epoch)

    def _load(self, path):
        if not path:
            return None
        return torch.load(path)

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


class Colorizer(Extension):

    def __init__(self, colors, num_output_channel=3):
        self.colors = self.normalized_color(colors)
        self.num_label = len(colors)
        self.num_channel = num_output_channel

    @staticmethod
    def normalized_color(colors):
        colors = np.array(colors)
        if colors.max() > 1:
            colors = colors / 255
        return colors

    def apply(self, label):
        if label.dim() == 3:
            label = label.unsqueeze(1)
        assert label.dim() == 4
        batch, _, h, w = label.size()
        canvas = torch.zeros(batch, self.num_channel, h, w)

        for channel in range(self.num_channel):
            for lbl_id in range(self.num_label):
                mask = label == lbl_id  # N x 1 x h x w
                channelwise_mask = torch.cat(
                    channel * [torch.zeros_like(mask)] +
                    [mask] +
                    (self.num_channel - 1 - channel) * [torch.zeros_like(mask)], dim=1)
                canvas[channelwise_mask] = self.colors[lbl_id][channel]

        return canvas
