from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .default import _C as config

import importlib

class cfg_dict(object):
    def __init__(self, d):
        self.__dict__ = d
        self.get = d.get

def set_cfg_from_file_city(cfg_path):
    spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec_loader = spec.loader.exec_module(cfg_file)
    cfg = cfg_file.cfg
    return cfg_dict(cfg)


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
