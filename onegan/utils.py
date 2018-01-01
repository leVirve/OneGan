# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import uuid

import torch
import scipy.misc
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


def unique_experiment_name(root, name):
    target_path = os.path.join(root, name)
    if os.path.exists(target_path):
        name = f'{name}_' + uuid.uuid4().hex[:6]

    if 'experiment_name' not in globals():
        global experiment_name
        experiment_name = name

    return os.path.join(root, experiment_name)


def img_normalize(img):
    mm, mx = img.min(), img.max()
    return img if mm == mx else img.add_(-mm).div_(mx - mm)


def save_batched_images(img_tensors, folder=None, filenames=None):
    os.makedirs(folder, exist_ok=True)

    for fname, img in zip(filenames, img_tensors):
        path = os.path.join(folder, '%s.png' % fname)
        scipy.misc.imsave(path, img_normalize(img.cpu().numpy()))
