# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from collections import defaultdict

import torch

from onegan.io import save_mat
from .base import Extension


class TensorCollector(Extension):
    """ Collect batched tensors

    Attributes:
        collection: :class:`dict` of collected tensors
    """

    def __init__(self):
        self.collection = defaultdict(list)

    def clear(self):
        """ Clear the internal collection """
        self.collection = defaultdict(list)

    def add(self, name, x):
        """ Add (`+`) to the `collection` list """
        self.collection[name] += x

    def append(self, name, x):
        """ Append (`.append`) to the `collection` list """
        self.collection[name].append(x)

    def save_mat(self, name: str, data: dict = None):
        """ Save the concatenated tensors into `mat` file

        Args:
            name: (str) saved output name
            data: (dict) data for saving
        """
        if data is None:
            data = {}
            for key, value in self.collection.items():
                if torch.is_tensor(value[0]):
                    data[key] = torch.cat(value, dim=0).numpy()
                else:
                    data[key] = value
        save_mat(name, data)
