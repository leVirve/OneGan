# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch

import onegan
from onegan.utils import img_normalize


def test_psnr():
    dummy_output = img_normalize(torch.rand(4, 3, 64, 64))
    dummy_target = img_normalize(torch.rand(4, 3, 64, 64))

    psnr = onegan.metrics.psnr(dummy_output, dummy_target)
    assert psnr


def test_semantic_segmentation_confusion():
    num_class = 7
    gt = torch.LongTensor(4, 3).random_(to=num_class)
    pred = torch.LongTensor(4, 3).random_(to=num_class)

    # input torch.tensor
    confusion = onegan.metrics.semantic_segmentation.confusion_table(
        pred, gt, num_class)
    assert confusion.sum() == 4 * 3

    # input np.ndarray
    confusion = onegan.metrics.semantic_segmentation.confusion_table(
        pred.cpu().numpy(), gt.cpu().numpy(), num_class)
    assert confusion.sum() == 4 * 3


def test_semantic_segmentation_iou():
    num_class = 7
    gt = torch.LongTensor(4, 3).random_(0, to=2)
    pred = torch.LongTensor(4, 3).random_(3, to=4)

    confusion = onegan.metrics.semantic_segmentation.confusion_table(pred, gt, num_class)
    iou = onegan.metrics.semantic_segmentation.intersection_over_union(confusion)
    assert len(iou) == num_class


def test_semantic_segmentation_metric():
    num_class = 3
    gt = torch.LongTensor(3, 3).random_(to=num_class)
    pred = torch.LongTensor(3, 3).random_(to=num_class)

    metric = onegan.metrics.semantic_segmentation.Metric(num_class=num_class, only_scalar=True, prefix='acc/')
    scalar_result = metric(pred, gt)
    metric = onegan.metrics.semantic_segmentation.Metric(num_class=num_class)
    full_result = metric(pred, gt)
    assert scalar_result.keys() != full_result.keys()
