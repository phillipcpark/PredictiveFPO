import sys
import csv
import random
from params import *

import numpy as np
from tr_eval_model import *

  
#
def load_predfpo_ds(path):
    f_hand = open(path, 'r')  
    reader = csv.reader(f_hand, delimiter=',')
    next(reader) 

    # indexed by graph
    g_edges     = []
    unary_masks = []
    
    # all graphs combined along axis 0 
    feats = [[]]
    labels  = [[]]
    g_idxs  = []

    curr_g_idx    = 0
    curr_edges    = [] 
    curr_is_unary = []

    for row in reader:        
        if (row == GRAPH_DELIM):         
            # new graph
            if (len(g_edges) <= curr_g_idx):
                g_edges.append(curr_edges)
                unary_masks.append(curr_is_unary)

            curr_edges    = [] 
            curr_is_unary = []

            g_idxs.append(curr_g_idx)
            feats.append([])
            labels.append([])
                   
        else:       
            attrs = [int(elem) if not(elem == '') else None for elem in row ]
            curr_g_idx = attrs[0]         
 
            if (is_unary(attrs[2])):
                curr_is_unary.append(True)
                curr_edges.append([attrs[3], attrs[1]])

            elif is_const(attrs[2]):
                curr_is_unary.append(False)

            else:
                curr_is_unary.append(False)
                curr_edges.append([attrs[3], attrs[1]])
                curr_edges.append([attrs[4], attrs[1]])

            feats[-1].append([attrs[2], attrs[5]])

            if (is_const(attrs[2])):
                labels[-1].append(IGNORE_CLASS)
            else:
                labels[-1].append(attrs[6])

    g_edges.append(curr_edges)
    unary_masks.append(curr_is_unary)
    g_idxs.append(curr_g_idx)

    # shuffle 
    shuff_idxs = np.arange(len(feats)) 
    random.shuffle(shuff_idxs)

    return g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs

#
if __name__ == '__main__':
    assert(len(sys.argv) > 1), "missing path to ds"
   
    # FIXME really need to write a DataLoader...
    g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs = load_predfpo_ds(sys.argv[1])

    bid_mpgnn = train_mpgnn(g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs)
    mpgnn_test_eval(bid_mpgnn,g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs)



