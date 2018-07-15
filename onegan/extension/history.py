# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from collections import defaultdict

import torch

from .base import Extension


class History(Extension):

    def __init__(self):
        self.count = defaultdict(int)
        self.meters = defaultdict(lambda: defaultdict(float))

    def add(self, kwvalues, n=1, log_suffix=''):
        display = {}
        for name, value in kwvalues.items():
            val = value.item() if torch.is_tensor(value) else value
            self.meters[log_suffix][f'{name}{log_suffix}'] += val
            display[name] = f'{val:.03f}'
        self.count[log_suffix] += n
        return display

    def get(self, key):
        return self.metric().get(key)

    def metric(self):
        result = {}
        for key in self.count.keys():
            cnt = self.count[key]
            out = {name: val / cnt for name, val in self.meters[key].items()}
            result.update(out)
        return result

    def clear(self):
        self.count = defaultdict(int)
        self.meters = defaultdict(lambda: defaultdict(float))
