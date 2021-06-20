import sys
import csv
import random
import numpy as np

from config.params import *
from model.train_bignn import train_bignn
from model.test_bignn import test_bignn
from model.bignn import bignn
from common.loader import ld_pfpo_ds
from common.session import get_dev
from common.otc import is_const, is_unary

from torch import load as th_load, device as th_dev





#
def load_bignn(path=None):
    global m

    if (MOD_PATH is None):
        m = bignn(OP_ENC_DIM, H_DIM, CLASSES)
        return m

    mod_dev    = get_dev() if USE_GPU else th_dev('cpu')
    state_dict = th_load(MOD_PATH, map_location=mod_dev)

    m = bignn(OP_ENC_DIM, H_DIM, CLASSES)
    m.to(mod_dev)
    m.load_hier_state(state_dict)
    return m 


#
#
#
if __name__ == '__main__':
    if not(len(sys.argv) == 2):
        raise RuntimeError('Expected dataset path as command line argument')

    ds    = ld_pfpo_ds(sys.argv[1])   
    model = load_bignn(MOD_PATH)
    train_bignn(model, ds)
    #test_bignn(ds, sys.argv[1], model)









