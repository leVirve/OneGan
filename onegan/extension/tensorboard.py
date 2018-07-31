# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorboardX

from onegan.visualizer import image as oneimage
from .base import Extension, unique_experiment_name


class TensorBoardLogger(Extension):
    r""" Convenient TensorBoard wrapping on tensorboardX

    Args:
        logdir (str): the root folder for tensorboard logging events (default: ``exp/logs``).
        name (str): subfolder name for current event writer (default: ``default``).
        max_num_images (int): number of images to log on the image panel (default: 20).

    Attributes:
        writer (:class:`tensorboardX.SummaryWriter`): internal wrapped writer.
        _tag_base_counter (int): internal image counter for :py:meth:`image` logging.
        _phase_state (str): internal state from the argument ``prefix`` of :py:meth:`image`.
    """

    def __init__(self, logdir='exp/logs', name='default', max_num_images=20):
        self.logdir = unique_experiment_name(logdir, name)
        self.max_num_images = max_num_images

        # internal usage
        self._tag_base_counter = 0
        self._phase_state = 'none'

    @property
    def writer(self):
        if not hasattr(self, '_writer'):
            self._writer = tensorboardX.SummaryWriter(self.logdir)
        return self._writer

    def clear(self):
        """ Manually clear the internal state ``_tag_base_counter`` and
            counter ``_phase_state``.
        """
        self._tag_base_counter = 0
        self._phase_state = 'none'

    def scalar(self, scalar_dict, epoch) -> None:
        """ Log scalar onto scalars panel.

        Args:
            scalar_dict (dict): :class:`dict` of scalars
            epoch (int): step for TensorBoard logging
        """
        [self.writer.add_scalar(tag, value, epoch) for tag, value in scalar_dict.items()]

    def image(self, images_dict, epoch, prefix='') -> None:
        """ Log image tensors onto images panel.

        Only ``max_num_images`` of image tensors will be logged, and while the ``prefix`` changed
        the internal image counter will be cleared automatically.

        Args:
            images_dict (dict): :class:`dict` of :class:`torch.Tensor`
            epoch (int): step for TensorBoard logging
            prefix (str): prefix string appended to the image tag.

        Shape:
            Each tensor in ``images_dict`` should in the shape of :math:`(N, C, H, W)`.
        """
        images_dict = self.remove_empty_pair(images_dict)

        # check state
        if self._phase_state != prefix:
            self._tag_base_counter = 0
            self._phase_state = prefix

        # check_num_images
        if self._tag_base_counter >= self.max_num_images:
            return
        num_summaried_img = len(next(iter(images_dict.values())))
        self._tag_base_counter += num_summaried_img

        [self.writer.add_image(f'{prefix}{tag}/{self._tag_base_counter + i}', oneimage.img_normalize(image), epoch)
         for tag, images in images_dict.items()
         for i, image in enumerate(images)]

    def histogram(self, tensors_dict, epoch, bins='auto') -> None:
        """ Log histogram onto histograms and distributions panels.

        Args:
            tensors_dict (dict): :class:`dict` of :class:`torch.Tensor`
            epoch (int): step for TensorBoard logging
            bins (str): `bins` for :py:meth:`tensorboardX.SummaryWriter.add_histogram`
        """
        [self.writer.add_histogram(f'{tag}', tensor, epoch, bins=bins) for tag, tensor in tensors_dict.items()]

    @staticmethod
    def remove_empty_pair(dict_results) -> dict:
        return {k: v for k, v in dict_results.items() if v is not None}
