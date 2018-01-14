# Copyright (c) 2018 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os

import pytest
import torch

from onegan.utils import (
    to_device, to_var, img_normalize, set_device_mode,
    unique_experiment_name, timeit
)


def test_img_normalize():
    dummy_img_tensor = torch.rand(3, 64, 64)
    transformed_tensor = dummy_img_tensor.add_(-0.5).div_(0.5)
    normalized_tensor = img_normalize(transformed_tensor)
    assert normalized_tensor.min() == 0
    assert normalized_tensor.max() == 1

    normalized_var = img_normalize(to_var(dummy_img_tensor))
    assert normalized_var.data.min() == 0
    assert normalized_var.data.max() == 1

    dummy_tensor = torch.LongTensor(3, 8, 8).random_(-2, to=2).float()
    normalized_tensor = img_normalize(dummy_tensor, img_range=(-2, 2))
    assert normalized_tensor.min() >= 0
    assert normalized_tensor.max() <= 1

    dummy_zero_tensor = torch.zeros(8, 8)
    img_normalize(dummy_zero_tensor)


def test_set_device_mode():
    with pytest.raises(AssertionError):
        set_device_mode('878787')


def test_to_device():
    dummy_tensor = torch.rand(8, 7)

    if torch.cuda.is_available():
        assert hasattr(to_device(dummy_tensor), 'get_device')
        set_device_mode('gpu')
        assert hasattr(to_device(dummy_tensor), 'get_device')

    set_device_mode('cpu')
    with pytest.raises(Exception):
        to_device(dummy_tensor).get_device()


def test_to_var():
    dummy_tensor = torch.rand(8, 7)

    if torch.cuda.is_available():
        assert hasattr(to_var(dummy_tensor), 'get_device')
        set_device_mode('gpu')
        assert hasattr(to_var(dummy_tensor), 'get_device')

    set_device_mode('cpu')
    with pytest.raises(Exception):
        to_var(dummy_tensor).get_device()


def test_unique_experiment_name_same_name_experiments():
    uname_1 = unique_experiment_name('.', 'one87')
    os.makedirs(uname_1, exist_ok=True)
    uname_2 = unique_experiment_name('.', 'one87')
    os.removedirs(uname_1)
    assert uname_1 != uname_2


def test_unique_experiment_name_different_experiments():
    uname_1 = unique_experiment_name('.', 'one87')
    uname_2 = unique_experiment_name('.', 'another87')
    assert uname_1 != uname_2


def test_timeit():
    @timeit
    def dummy_foo(arg='87'):
        print(arg)
