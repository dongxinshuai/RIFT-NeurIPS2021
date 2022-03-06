# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np

from .AdvPLM import AdvPLM

def setup(opt):
    
    if opt.model == "adv_plm":
        model =AdvPLM(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model
