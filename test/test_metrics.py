# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch

import onegan
from onegan.utils import to_var, img_normalize


def test_psnr():
    dummy_output = to_var(img_normalize(torch.rand(4, 3, 64, 64)))
    dummy_target = to_var(img_normalize(torch.rand(4, 3, 64, 64)))

    psnr = onegan.metrics.psnr(dummy_output, dummy_target)
    assert psnr


def test_semantic_segmentation_confusion():
    num_class = 7
    gt = to_var(torch.LongTensor(4, 3).random_(to=num_class))
    pred = to_var(torch.LongTensor(4, 3).random_(to=num_class))

    confusion = onegan.metrics.semantic_segmentation_confusion(pred, gt, num_class)
    assert confusion.sum() == 4 * 3


def test_semantic_segmentation_iou():
    num_class = 7
    gt = to_var(torch.LongTensor(4, 3).random_(0, to=2))
    pred = to_var(torch.LongTensor(4, 3).random_(3, to=4))

    confusion = onegan.metrics.semantic_segmentation_confusion(pred, gt, num_class)
    iou = onegan.metrics.semantic_segmentation_iou(confusion)
    assert len(iou) == num_class
