# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging


class Extension:

    @property
    def logger(self):
        """ logger for specific succeeding class """
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(type(self).__name__)
        return self._logger


def unique_experiment_name(root, name):
    import os
    from datetime import datetime

    target_path = os.path.join(root, name)
    # TODO: fix bug in Tensorboard logger and checkpoint unique name
    if os.path.exists(target_path):
        name = f'{name}_' + datetime.now().strftime('%m-%dT%H-%M')

    _name = globals().get('experiment_name')
    if _name is None or _name[:-12] != name:
        global experiment_name
        experiment_name = name

    return os.path.join(root, experiment_name)
