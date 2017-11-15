# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


class Extention():

    def __init__(self, logger):
        self.logger = logger


class ImageSummaryExtention(Extention):

    def __init__(self, logger, summary_num_images=20):
        super().__init__(logger)
        self.logger.summary_num_images = summary_num_images

    def __call__(self, kwimages, epoch, prefix=''):
        self.logger.image(kwimages, prefix=prefix, step=epoch)


class HistoryExtention(Extention):

    def __init__(self, logger):
        super().__init__(logger)

    def __call__(self, kwscalars, epoch):
        self.logger.scalar(kwscalars, step=epoch)
