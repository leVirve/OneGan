# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import logging
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

    @property
    def logger(self):
        """ :Logger: logger for specific succeeding class """
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(type(self).__name__)
        return self._logger


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
            name, ext = os.path.splitext(fname)
            ext = '.png' if ext not in ['.png', '.jpg'] else ext
            path = os.path.join(self.savedir, name + ext)
            scipy.misc.imsave(path, img.cpu().numpy())


class History(Extension):

    def __init__(self):
        self.count = defaultdict(int)
        self.meters = defaultdict(lambda: defaultdict(float))

    def add(self, kwvalues, n=1, log_suffix=''):
        display = {}
        for name, value in kwvalues.items():
            val = value.item() if isinstance(value, Variable) else value
            self.meters[log_suffix][f'{name}{log_suffix}'] += val
            display[name] = f'{val:.03f}'
        self.count[log_suffix] += n
        return display

    def get(self, key):
        return self.metric()[key]

    def metric(self):
        result = {}
        for key in self.count.keys():
            cnt = self.count[key]
            out = {name: val / cnt for name, val in self.meters[key].items()}
            result.update(out)
        return result

    def clear(self):
        self.count = defaultdict(int)
        self.meters = defaultdict(lambda: defaultdict(float))


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
        return Path(self._savedir)

    def get_checkpoint_dir(self, unique=False):
        if unique:
            return self.savedir
        return Path(self.savedir) / self.name

    def load_trained_model(self, weight_path, remove_module=False):
        """ another loader method for `model`

        It can recover changed model module from dumped `latest.pt` and load
        pre-trained weights.

        Arge:
            weight_path (str): full path or short weight name to the dumped weight
            remove_module (bool)
        """
        folder = self.get_checkpoint_dir()
        if str(folder) not in weight_path:
            folder = Path(weight_path).parent

        ckpt = torch.load(folder / 'latest.pt')
        full_model = ckpt['model']

        if 'latest.pt' not in weight_path:
            state_dict = export_checkpoint_weight(weight_path, remove_module)
            full_model.load_state_dict(state_dict)

        return full_model

    def load(self, path=None, model=None, remove_module=False, resume=False):
        """ load method for `model` and `optimizor`

        If `resume` is True, full `model` and `optimizer` modules will be returned;
        or the loaded model will be returned.

        Args:
            path (str): full path to the dumped weight or full module
            model (nn.Module)
            remove_module (bool)
            resume (bool)

        Return:
            - dict() of dumped data inside `latest.pt`
            - OrderedDict() of `state_dict`
            - nn.Module of input model with loaded state_dict
            - nn.Module of dumped full module with loaded state_dict
        """
        if resume:
            latest_ckpt = torch.load(path)
            return latest_ckpt

        try:
            state_dict = export_checkpoint_weight(path, remove_module)
            if model is None:
                return state_dict
            model.load_state_dict(state_dict)
            return model
        except KeyError:
            self.logger.warn('Use fallback solution: load `latest.pt` as module')
            return self.load_trained_model(path, remove_module)

    def save(self, model, optimizer, epoch):
        """ save method for `model` and `optimizor`

        Args:
            model (nn.Module)
            optimizer (nn.Module)
            epoch (int): epoch step of training
        """
        if (epoch + 1) % self.save_epochs:
            return

        folder = self.get_checkpoint_dir(unique=True)
        torch.save({'weight': model.state_dict()}, folder / f'net-{epoch}.pt')
        torch.save({
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch + 1
        }, folder / 'latest.pt')

    def get_weights(self, weight_path, model=None, remove_module=False):
        """ model weights searcher

        Args:
            weight_path (str): the path to single weight file or the folder of weights
            model (nn.Module)
        """

        weight_path = Path(weight_path)
        if weight_path.is_file():
            path = str(weight_path)
            payload = self.load(path, model=model, remove_module=remove_module)
            yield payload, path

        paths = list(weight_path.glob('*.pt'))
        if weight_path.is_dir():
            assert len(paths), 'Weights folder contains nothing.'

        for path in paths:
            path = str(path)
            if 'latest.pt' in path:
                continue
            payload = self.load(path, model=model, remove_module=remove_module)
            model = payload  # use corrected model_def
            yield payload, path


class GANCheckpoint(Checkpoint):

    def load(self, trainer, gnet_path=None, dnet_path=None, resume=False):
        epoch = self._load(dnet_path, trainer.dnet, trainer.d_optim)
        epoch = self._load(gnet_path, trainer.gnet, trainer.g_optim)
        if resume:
            trainer.start_epoch = epoch

    def save(self, trainer, epoch):
        if (epoch + 1) % self.save_epochs:
            return
        self._save('dnet-{epoch}.pth', trainer.model_d, trainer.optim_d, epoch)
        self._save('gnet-{epoch}.pth', trainer.model_g, trainer.optim_g, epoch)


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
