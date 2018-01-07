# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from math import log10

import torch.nn.functional as F
import numpy as np

from onegan.utils import img_normalize


def psnr(output, target, img_range=None):
    ''' calculate the PSNR
        output: Variable, range in (0, 1)
        target: Variable, range in (0, 1)
    '''

    def normalize(t):
        return img_normalize(t, img_range=img_range)

    psnrs = np.array([
        10 * log10(1 / F.mse_loss(normalize(pred), normalize(targ)).data[0] + 1e-9)
        for pred, targ in zip(output, target)])
    return psnrs.mean()


def semantic_segmentation_confusion(preds, labels, num_class: int):
    ''' calculate the confision matrix
        preds: Variable
        labels: Variable

        *credit: refer from [chainer/chainercv] eval_semantic_segmentation.py
    '''
    confusion = np.zeros(num_class * num_class, dtype=np.int64)

    for pred, label in zip(preds, labels):
        pred = pred.view(-1)
        label = label.view(-1)
        mask = label < 255
        hist = num_class * label[mask] + pred[mask]
        confusion += np.bincount(hist.data.cpu().numpy(), minlength=num_class ** 2)

    return confusion.reshape((num_class, num_class))


def semantic_segmentation_iou(confusion: np.ndarray):
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    return np.diag(confusion) / (iou_denominator + 1e-9)


class SemanticSegmentationMetric():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, output, target):
        confusion = semantic_segmentation_confusion(output, target, num_class=self.num_classes)
        iou = semantic_segmentation_iou(confusion)
        pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
        class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

        return {'iou': iou, 'miou': np.nanmean(iou),
                'pixel_accuracy': pixel_accuracy,
                'class_accuracy': class_accuracy,
                'mean_class_accuracy': np.nanmean(class_accuracy)}
