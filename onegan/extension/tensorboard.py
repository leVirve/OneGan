# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorboardX

from onegan.visualizer import image as oneimage
from .base import Extension, unique_experiment_name


def check_state(f):
    def wrapper(instance, kw_images, epoch, prefix=''):
        if instance._phase_state != prefix:
            instance._tag_base_counter = 0
            instance._phase_state = prefix
        return f(instance, kw_images, epoch, prefix)
    return wrapper


def check_num_images(f):
    def wrapper(instance, kw_images, epoch, prefix=''):
        if instance._tag_base_counter >= instance.max_num_images:
            return
        num_summaried_img = len(next(iter(kw_images.values())))
        result = f(instance, kw_images, epoch, prefix)
        instance._tag_base_counter += num_summaried_img
        return result
    return wrapper


class TensorBoard(Extension):
    r""" Convenient TensorBoard wrapping tensorboardX

    Args:
        logdir (str): the root folder for tensorboard logging events (default: 'exp/logs')
        name (str): subfolder name for current event writer (default: `default`)
        max_num_images (int): number of images to log on the image panel (default: 20)
    """

    def __init__(self, logdir='exp/logs', name='default', max_num_images=20):
        self.logdir = unique_experiment_name(logdir, name)
        self.max_num_images = max_num_images

        # internal usage
        self._tag_base_counter = 0
        self._phase_state = None

    @property
    def writer(self):
        if not hasattr(self, '_writer'):
            self._writer = tensorboardX.SummaryWriter(self.logdir)
        return self._writer

    def clear(self):
        """ Manually clear the state of logger """
        self._tag_base_counter = 0
        self._phase_state = None

    def scalar(self, scalar_dict, epoch):
        """
        Args:
            scalar_dict: :class:`dict` of scalars
            epoch: step for TensorBoard logging
        """
        [self.writer.add_scalar(tag, value, epoch) for tag, value in scalar_dict.items()]

    @check_state
    @check_num_images
    def image(self, images_dict, epoch, prefix=''):
        """
        Args:
            images_dict: :class:`dict` of tensors [batch, channel, height, width]
            epoch: step for TensorBoard logging
            prefix: prefix string for tag
        """
        images_dict = self.remove_empty_pair(images_dict)
        [self.writer.add_image(f'{prefix}{tag}/{self._tag_base_counter + i}', oneimage.img_normalize(image), epoch)
         for tag, images in images_dict.items()
         for i, image in enumerate(images)]

    def histogram(self, tensors_dict, epoch, bins='auto'):
        """
        Args:
            tensors_dict: :class:`dict` of tensors
            epoch: step for TensorBoard logging
            bins: `bins` for tensorboarSdX.add_histogram
        """
        [self.writer.add_histogram(f'{tag}', tensor, epoch, bins=bins) for tag, tensor in tensors_dict.items()]

    @staticmethod
    def remove_empty_pair(dict_results) -> dict:
        return {k: v for k, v in dict_results.items() if v is not None}
