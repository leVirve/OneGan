# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import argparse

import yaml


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


class Parser():

    def __init__(self, description='', config=None):
        self.config_file = config
        self.parser = argparse.ArgumentParser(description=description)
        self._add_default_option()

    def parse(self, args=None, namespace=None):
        """ Parse the arguments from command-line and config file.
        The arguments will be overwritten by command-line ones.
        """
        global cfg
        namespace = cfg if namespace is None else namespace

        file_cfg = self._load_option_config(self.config_file)
        cli_cfg = self.parser.parse_args(args, namespace=namespace)
        for key, v in file_cfg.items():
            if isinstance(v, dict):
                # support two-level attr-dict
                attr_dict = AttrDict(v)
                setattr(cli_cfg, key, attr_dict)
            if key not in cli_cfg or not getattr(cli_cfg, key):
                setattr(cli_cfg, key, v)

        return cli_cfg

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def _add_default_option(self):
        trainer_option(self.parser)

    def _load_option_config(self, path):
        if not path:
            return {}
        with open(path) as f:
            return yaml.load(f)


def trainer_option(parser):
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--worker', type=int)


cfg = AttrDict()
