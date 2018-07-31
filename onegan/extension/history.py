# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from collections import defaultdict

import torch

from .base import Extension


class History(Extension):
    """ History meters for training losses and accuracies.

    :py:meth:`clear` will be called at the beginning of every epoch in
    :class:`onegan.estimator.OneEstimator`.
    """

    def __init__(self):
        self.count = defaultdict(int)
        self.meters = defaultdict(lambda: defaultdict(float))
        self._metric = None

    @property
    def metric(self):
        """ Accumulated results in the form of :class:`dict`. """
        if self._metric is None:
            result = {}
            for phase, cnt in self.count.items():
                out = {name: val / cnt for name, val in self.meters[phase].items()}
                result.update(out)
            self._metric = result
        return self._metric

    def update(self, kwvalues, n=1, log_suffix='') -> dict:
        """
        Args:
            kwvalues (dict): values to be accumulated and recorded.
            n (int): number of the records (default: 1).
            log_suffix (str): suffix for the tag (default: `''`).

        Returns:
            display: the scalar to be shown in the form for displaying in ``tqdm``.
        """
        display = {}
        for name, value in kwvalues.items():
            val = value.item() if torch.is_tensor(value) else value
            display[name] = f'{val:.03f}'
            self.meters[log_suffix][f'{name}{log_suffix}'] += val
        self.count[log_suffix] += n
        return display

    def add(self, kwvalues, n=1, log_suffix='') -> dict:
        """ Deprecated. Use :py:meth:`update` instead. """
        self.update(kwvalues, n=1, log_suffix='')

    def get(self, key):
        """ Get the accumulated result by the ``key`` from :py:attr:`metric`. """
        return self.metric.get(key)

    def clear(self) -> None:
        """ Clear the state of history. """
        self.count = defaultdict(int)
        self.meters = defaultdict(lambda: defaultdict(float))
        self._metric = None
