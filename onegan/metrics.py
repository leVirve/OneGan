# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from math import log10

import numpy as np
import torch.nn.functional as F


def psnr(output, target):

    def normalize(img):
        mm, mx = img.min(), img.max()
        return img.add(-mm).div(mx - mm)

    psnrs = np.array([
        10 * log10(1 / (F.mse_loss(normalize(pred), normalize(targ)).data[0] + 1e-6))
        for pred, targ in zip(output, target)])
    return psnrs.mean()


class Segmentation():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, output, target):
        ''' Refer from: [chainer/chainercv] eval_semantic_segmentation.py
        '''
        def to_numpy(x):
            return x.data.cpu().numpy()

        confusion = self.semantic_confusion(to_numpy(output), to_numpy(target))
        iou = self.semantic_iou(confusion)
        pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
        class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

        return {'acc/miou': np.nanmean(iou),
                'acc/pixel': pixel_accuracy,
                'acc/mean_class': np.nanmean(class_accuracy)}

    def semantic_iou(self, confusion):
        iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
        return np.diag(confusion) / iou_denominator

    def semantic_confusion(self, pred_labels, gt_labels):
        n_class = self.num_classes
        confusion = np.zeros((n_class, n_class), dtype=np.int64)

        for pred_label, gt_label in zip(pred_labels, gt_labels):
            pred_label = pred_label.flat
            gt_label = gt_label.flat
            mask = 255 > gt_label
            confusion += np.bincount(
                n_class * gt_label[mask].astype(int) + pred_label[mask],
                minlength=n_class**2).reshape((n_class, n_class))
        return confusion


class AccuracyMetric():

    def __init__(self):
        pass
