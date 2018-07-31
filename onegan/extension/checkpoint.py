# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
from pathlib import Path
from collections import OrderedDict

import torch

from .base import Extension, unique_experiment_name


def export_checkpoint_weight(checkpoint_path, remove_module=True):

    def clean_module(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        return new_state_dict

    ckpt = torch.load(checkpoint_path)
    weight = ckpt['weight']
    return clean_module(weight) if remove_module else weight


class Checkpoint(Extension):
    r""" Checkpoint manager for model saving and restoring.

    Args:
        rootdir (str): the root folder for checkpoint manager (default: ``exp/checkpoints``).
        name (str): subfolder name for current experiment (default: ``default``).
        save_interval (int): interval of epochs to save the checkpoint (default: 10)
    """

    def __init__(self, rootdir='exp/checkpoints/', name='default', save_interval=10):
        self.rootdir = rootdir
        self.name = name
        self.save_interval = save_interval

    @property
    def savedir(self):
        if not hasattr(self, '_savedir'):
            self._savedir = unique_experiment_name(self.rootdir, self.name)
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

        Args:
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
        """ load method for `model` and `optimizer`

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

    def save(self, model, optimizer=None, epoch=None):
        """ save method for `model` and `optimizer`

        Args:
            model (nn.Module)
            optimizer (nn.Module)
            epoch (int): epoch step of training
        """
        if (epoch + 1) % self.save_interval:
            return

        folder = self.get_checkpoint_dir(unique=True)
        torch.save({'weight': model.state_dict()}, folder / f'net-{epoch}.pt')
        torch.save({
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch + 1
        }, folder / 'latest.pt')

    def get_weights(self, weight_path, model=None, remove_module=False, path_only=False):
        """ model weights searcher

        Args:
            weight_path (str): the path to single weight file or the folder of weights
            model (nn.Module): if given, the model will be filled with state_dict
            remove_module (bool): remove the `module.` string from the keys of state_dict
            path_only (bool): if true, the return value will be only path to weights
        Returns:
            - payload, path: if model is given, payload will be loaded model else will be state_dict
            - path: the path to the weight
        """

        weight_path = Path(weight_path)
        if weight_path.is_file():
            path = str(weight_path)
            if path_only:
                yield path
            payload = self.load(path, model=model, remove_module=remove_module)
            return payload, path

        paths = list(weight_path.glob('*.pt'))
        if weight_path.is_dir():
            assert len(paths), 'Weights folder contains nothing.'

        for path in paths:
            path = str(path)
            if 'latest.pt' in path:
                continue
            if path_only:
                yield path
                continue
            payload = self.load(path, model=model, remove_module=remove_module)
            model = payload  # use corrected model_def
            yield payload, path


# deprecated
class GANCheckpoint(Checkpoint):

    def load(self, trainer, gnet_path=None, dnet_path=None, resume=False):
        epoch = self._load(dnet_path, trainer.dnet, trainer.d_optim)
        epoch = self._load(gnet_path, trainer.gnet, trainer.g_optim)
        if resume:
            trainer.start_epoch = epoch

    def save(self, trainer, epoch):
        if (epoch + 1) % self.save_interval:
            return
        self._save('dnet-{epoch}.pth', trainer.model_d, trainer.optim_d, epoch)
        self._save('gnet-{epoch}.pth', trainer.model_g, trainer.optim_g, epoch)
