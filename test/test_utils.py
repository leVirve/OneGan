# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os

import torch

import onegan
from onegan.visualizer import img_normalize
from onegan.extension import unique_experiment_name
# from onegan.utils import (
#     device, set_device, img_normalize,
#     unique_experiment_name
# )


def test_img_normalize():
    dummy_tensor = torch.rand(3, 64, 64).add_(-0.5).div_(0.5)

    normalized_tensor = img_normalize(dummy_tensor)
    assert normalized_tensor.min() == 0
    assert normalized_tensor.max() == 1

    dummy_tensor = torch.randn(3, 8, 8)
    normalized_tensor = img_normalize(dummy_tensor, val_range=(-2, 2))
    assert normalized_tensor.min() < 0
    assert normalized_tensor.max() > 1

    dummy_zero_tensor = torch.zeros(8, 8)
    img_normalize(dummy_zero_tensor)


def test_device():
    # default device
    assert onegan.device().type == ('cuda' if torch.cuda.is_available() else 'cpu')

    # change to cpu
    onegan.set_device('cpu')
    assert onegan.device().type == 'cpu'

    if torch.cuda.is_available():
        # change back to gpu
        onegan.set_device('cuda')
        assert onegan.device().type == 'cuda'

        # change back to gpu:0
        onegan.set_device('cuda:0')
        assert onegan.device().type == 'cuda' and onegan.device().index == 0


def test_unique_experiment_name_same_name_experiments():
    name_1 = unique_experiment_name('.', 'one87')
    os.makedirs(name_1, exist_ok=True)
    name_2 = unique_experiment_name('.', 'one87')
    os.removedirs(name_1)
    assert name_1 != name_2


def test_unique_experiment_name_different_experiments():
    name_1 = unique_experiment_name('.', 'one87')
    name_2 = unique_experiment_name('.', 'another87')
    assert name_1 != name_2
