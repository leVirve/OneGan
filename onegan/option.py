# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import argparse

import yaml


class Parser():

    def __init__(self, description, config=None):
        self.config_file = config
        self.parser = argparse.ArgumentParser(description=description)
        self._add_default_option()

    def parse(self, args=None):
        base_cfg = self._load_option_config(self.config_file)
        cli_cfg = self.parser.parse_args(args)
        for k, v in base_cfg.items():
            if k not in cli_cfg or not getattr(cli_cfg, k):
                setattr(cli_cfg, k, v)
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
