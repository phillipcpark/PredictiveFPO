import csv
import argparse
from json import load
from numpy import arange
import common.otc as otc
from model.bignn import bignn
from random import shuffle
from common.pfpo_utils import get_dev
import torch as th

# read CL arguments
def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", help="path to JSON config for job", required=True, \
                        dest="config_path", metavar="") 
    parser.add_argument("-ds", help="path to dataset for directory", required=True, \
                        dest="ds_path", metavar="") 
    parser.add_argument("-m", help="mode 0: train, 1:test", required=True, type=int, \
                        dest="mode", metavar="") 
    args = vars(parser.parse_args())
    return args

# read JSON from path
def ld_config(path)->dict:
    with open(path, "r") as config_fo:
        config = load(config_fo)
    return config 

# 
def ld_pfpo_ds(path, config):
    f_hand = open(path + "/ds.csv", 'r')  
    reader = csv.reader(f_hand, delimiter=',')
    next(reader) 

    # indexed by graph
    g_edges     = []
    unary_masks = []
    
    # graphs indexed along axis 0 
    feats   = [[]]
    labels  = [[]]
    g_idxs  = []

    curr_g_idx    = 0
    curr_edges    = [] 
    curr_is_unary = []

    exec_lists = [[]]

    for row in reader:        
        if (row[0] == ''):         
            # new graph
            if (len(g_edges) <= curr_g_idx):
                g_edges.append(curr_edges)
                unary_masks.append(curr_is_unary)

                exec_lists.append([])

            #set label to ignore class if not const/var/func                 
            counts = [0 for i in range(len(curr_is_unary))]
            for edge in curr_edges:
                counts[edge[0]] += 1

            for nidx in range(len(counts)):
                if (counts[nidx]<2 and not(curr_is_unary[nidx])) or feats[-1][nidx][0]==0: 
                    labels[-1][nidx] = config['model']['ignore_class'] 

            curr_edges    = [] 
            curr_is_unary = []

            g_idxs.append(curr_g_idx)
            feats.append([])
            labels.append([])
                   
        else:       
            attrs = [int(elem) if not(elem == '') else None for elem in row ]
            curr_g_idx = attrs[0]         

            feats[-1].append([attrs[2], attrs[5]])           
            exec_lists[-1].append([attrs[1], attrs[2], attrs[3], attrs[4]])
 
            # cases are unary op node, constant node, and binary op node 
            if (otc.is_unary(attrs[2])):
                curr_is_unary.append(True)
                curr_edges.append([attrs[3], attrs[1]])
            elif otc.is_const(attrs[2]):
                curr_is_unary.append(False)
            else:
                curr_is_unary.append(False)
                curr_edges.append([attrs[3], attrs[1]])
                curr_edges.append([attrs[4], attrs[1]])

            # ds labels are in {0: -2 types, 1: -1 type, 2: keep type, 3: +1 type, 4: +2 types};
            # only tuning down 1 type, from 64->32 
            labels[-1].append(1 if (attrs[6]<2) else 0)

    g_edges.append(curr_edges)
    unary_masks.append(curr_is_unary)
    g_idxs.append(curr_g_idx)

    # shuffle 
    shuff_idxs = arange(len(feats)) 
    shuffle(shuff_idxs)

    return {'g_edges':g_edges, 'feats':feats, 'labels':labels, 'unary_masks':unary_masks, 'g_idxs':g_idxs, 'shuff_idxs':shuff_idxs, 'exec_lists':exec_lists}

#
def ld_bignn(config):
    # global for now, to make accessible to flask POSTs 
    global m
    m = bignn(otc.OP_ENC_DIM, config['model']['hidden_dim'], config['model']['classes'], \
              config['model']['tie_mp_params'], config['model']['message_passing_steps'])

    path = config['pretrained']['model_path']
    if (path is None):
        return m

    mod_dev    = get_dev() if config['use_gpu'] else th.device('cpu')
    state_dict = th.load(path, map_location=mod_dev)

    m.to(mod_dev)
    m.load_hier_state(state_dict)
    return m 




