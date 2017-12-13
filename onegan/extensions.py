# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from collections import defaultdict

import tensorboardX

from onegan.utils import get_unique_subfolder, img_normalize


def check_state(f):
    def wrapper(instance, kw_images, epoch, prefix):
        if instance._phase_state != prefix:
            instance._tag_base_counter = 0
            instance._phase_state = prefix
        return f(instance, kw_images, epoch, prefix)
    return wrapper


def check_num_images(f):
    def wrapper(instance, kw_images, epoch, prefix):
        if instance._tag_base_counter >= instance.max_num_images:
            return
        num_summaried_img = len(next(iter(kw_images.values())))
        instance._tag_base_counter += num_summaried_img
        return f(instance, kw_images, epoch, prefix)
    return wrapper


class Extension:
    pass


class Logger(Extension):

    def __init__(self, logdir='logs', name='default', max_num_images=20):
        self.logdir = get_unique_subfolder(logdir, name)
        self.max_num_images = max_num_images
        self._tag_base_counter = 0
        self._phase_state = None

    def scalar(self, kw_scalars, epoch):
        [self.writer.add_scalar(tag, value, epoch) for tag, value in kw_scalars.items()]

    @check_state
    @check_num_images
    def image(self, kw_images, epoch, prefix):
        [self.writer.add_image(f'{prefix}{tag}/{self._tag_base_counter + i}', img_normalize(image), epoch)
         for tag, images in kw_images.items()
         for i, image in enumerate(images)]

    def clear_state(self):
        self.image_base_tag_counter = 0
        self._phase_state = None

    @property
    def writer(self):
        if not hasattr(self, '_writer'):
            self._writer = tensorboardX.SummaryWriter(self.logdir)
        return self._writer


class History(Extension):

    def __init__(self, length):
        self.len = length
        self.losses = defaultdict(float)
        self.accuracies = defaultdict(float)

    def add(self, losses, accuracies, log_suffix=''):
        display = {}
        for name, acc in accuracies.items():
            self.accuracies[f'{name}{log_suffix}'] += acc
            display[name] = '%.02f' % acc
        for name, loss in losses.items():
            self.losses[f'{name}{log_suffix}'] += loss.data[0]
            display[name] = '%.03f' % loss.data[0]
        return display

    def metric(self):
        terms = {name: acc / self.len for name, acc in self.accuracies.items()}
        terms.update({name: loss / self.len for name, loss in self.losses.items()})
        return terms