# Copyright (c) 2017 Salas Lin (leVirve)
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


def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    if is_variable(x):
        return x.data.cpu().numpy()
    return x


def set_device_mode(device='gpu'):
    assert device in ('cpu', 'gpu')
    global cuda_available
    cuda_available = device == 'gpu'


def is_variable(x):
    return 'variable' in str(type(x))


def unique_experiment_name(root, name):
    target_path = os.path.join(root, name)
    if os.path.exists(target_path):
        name = f'{name}_' + datetime.now().strftime('%m-%dT%H-%M')

    _name = globals().get('experiment_name')
    if _name is None or _name[:-6] != name:
        global experiment_name
        experiment_name = name

    return os.path.join(root, experiment_name)


def img_normalize(img, img_range=None):
    ''' normalize the tensor into (0, 1)
        tensor: Tensor or Variable
        img_range: tuple of (min_val, max_val)
    '''
    t = img.clone()
    if img_range:
        mm, mx = img_range[0], img_range[1]
    else:
        mm, mx = t.min(), t.max()
    if isinstance(img, Variable):
        return t.add(-mm).div(mx - mm)
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
    weight = ckpt['model']
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
