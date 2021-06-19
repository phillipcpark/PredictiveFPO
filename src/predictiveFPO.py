import sys
import csv
import random
from collections import Counter
import numpy as np

from params import *
from otc import is_const, is_unary
from train_test_bignn import train_bignn, test_bignn

# 
def ld_predfpo_ds(path):
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
        if (row == GRAPH_DELIM):         
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
                    labels[-1][nidx] = IGNORE_CLASS                          

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
            if (is_unary(attrs[2])):
                curr_is_unary.append(True)
                curr_edges.append([attrs[3], attrs[1]])
            elif is_const(attrs[2]):
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
    shuff_idxs = np.arange(len(feats)) 
    random.shuffle(shuff_idxs)

    return {'g_edges':g_edges, 'feats':feats, 'labels':labels, 'unary_masks':unary_masks, 'g_idxs':g_idxs, 'shuff_idxs':shuff_idxs, 'exec_lists':exec_lists}


#
#
#
if __name__ == '__main__':
    if not(len(sys.argv) == 2):
        raise RuntimeError('Expected dataset path as command line argument')

    ds    = ld_predfpo_ds(sys.argv[1])   
    #bignn = train_bignn(ds)
    #test_bignn(ds, sys.argv[1], bignn)
    test_bignn(ds, sys.argv[1])









