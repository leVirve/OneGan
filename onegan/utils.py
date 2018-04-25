# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import time
from collections import OrderedDict
from datetime import datetime
from functools import wraps

import scipy.misc
import torch
from torch.autograd import Variable


default_device_name = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x


def device():
    return torch.device(default_device_name)


def set_device(device):
    global default_device_name
    default_device_name = device


def unique_experiment_name(root, name):
    target_path = os.path.join(root, name)
    # TODO: fix bug in Tensorboard logger and checkpoint unique name
    if os.path.exists(target_path):
        name = f'{name}_' + datetime.now().strftime('%m-%dT%H-%M')

    _name = globals().get('experiment_name')
    if _name is None or _name[:-12] != name:
        global experiment_name
        experiment_name = name

    return os.path.join(root, experiment_name)


def img_normalize(img, val_range=None):
    ''' Normalize the tensor into (0, 1)

    Args:
        tensor: torch.Tensor
        val_range: tuple of (min_val, max_val)
    Returns:
        img: normalized tensor
    '''
    t = img.clone()
    if val_range:
        mm, mx = val_range[0], val_range[1]
    else:
        mm, mx = t.min(), t.max()
    try:
        return t.add_(-mm).div_(mx - mm)
    except RuntimeError:
        return img


def save_batched_images(img_tensors, folder=None, filenames=None):
    os.makedirs(folder, exist_ok=True)

    for fname, img in zip(filenames, img_tensors):
        path = os.path.join(folder, fname)
        scipy.misc.imsave(path, img.cpu().numpy())


def export_checkpoint_weight(checkpoint_path, remove_module=True):

    def clean_module(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        return new_state_dict

    ckpt = torch.load(checkpoint_path)
    weight = ckpt['weight']
    return clean_module(weight) if remove_module else weight


def timeit(f):

    @wraps(f)
    def wrap(*args, **kw):
        s = time.time()
        result = f(*args, **kw)
        e = time.time()
        print('--> %s(), cost %2.4f sec' % (f.__name__, e - s))
        return result

    return wrap
