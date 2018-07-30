# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from onegan import visualizer   # noqa
from onegan import metrics   # noqa
from onegan import extension  # noqa

import onegan.loss  # noqa
import onegan.estimator  # noqa
import onegan.models  # noqa
import onegan.io  # noqa
import onegan.ops  # noqa
import onegan.option  # noqa
import onegan.external  # noqa


__version__ = '0.5.0a'


# environment

import torch


default_device_name = 'cuda' if torch.cuda.is_available() else 'cpu'


def device():
    """ return the current default global device """
    return torch.device(default_device_name)


def set_device(device):
    """ set the current default global device """
    global default_device_name
    default_device_name = device
