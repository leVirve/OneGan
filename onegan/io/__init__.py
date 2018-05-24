# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import onegan.io.loader  # noqa
import onegan.io.transform  # noqa

from onegan.io.loader import *  # noqa


def save_mat(name, data):
    """ Save data into *.mat file
    """
    import scipy.io as io
    io.savemat(name, data)
