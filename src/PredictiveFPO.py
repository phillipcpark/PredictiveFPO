import sys
import csv
import random

from collections import Counter

import numpy as np
from tr_eval_model import *
from otc import *
from params import *

#
def load_nootcfeat_ds(path):
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

    # filter repeated graphs
    feats_filt  = [[feats[0][idx][0] for idx in range(len(feats[0]))]]
    labels_filt = [[tune_prec(feats[0][idx][1], labels[0][idx]) if not(labels[0][idx] == IGNORE_CLASS) else IGNORE_CLASS \
                                                                for idx in range(len(feats[0]))]]
    last_g_idx = g_idxs[0]
            
    for ex_idx in range(1, len(g_idxs)):
        if (last_g_idx == g_idxs[ex_idx]):
            continue

        last_g_idx = g_idxs[ex_idx]

        feats_filt.append([feats[ex_idx][idx][0] for idx in range(len(feats[ex_idx]))])           

        labels_filt.append([tune_prec(feats[ex_idx][idx][1], labels[ex_idx][idx]) if not(labels[ex_idx][idx] == IGNORE_CLASS) else IGNORE_CLASS \
                                                                for idx in range(len(feats[ex_idx]))])
 
    # shuffle 
    shuff_idxs = np.arange(len(feats_filt)) 
    random.shuffle(shuff_idxs)
    return g_edges, feats_filt, labels_filt, unary_masks, np.arange(len(g_edges)), shuff_idxs

  
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

           
            elif (COARSE_TUNE):
                classes = 5 #NOTE hardcoded
                gt_prec = max(attrs[5] + (attrs[6] - int(classes-1)/int(2)), 0) 

                if (SP_TARGET):
                    labels[-1].append(1 if (gt_prec == 0) else 0)
                else:
                    labels[-1].append(1 if attrs[6] < int(classes-1)/int(2) else 0)

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
def output_targ_stats(labels):
    targs = []
    for label in labels:
        targs += label

    counts = Counter(targs)
    total = len(targs)

    print("\n**DS target class props")
    for i in range(CLASSES):
        print(str(i) + ": " + str(float(counts[i])/total)) 
    print("")


#
if __name__ == '__main__':
    assert(len(sys.argv) > 1), "missing path to ds"

    g_edges = feats =  labels = unary_masks = g_idxs = shuff_idxs = None
   
    if (SINGLE_GRAPH_TARG):
        g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs = load_nootcfeat_ds(sys.argv[1])
    else: 
        g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs = load_predfpo_ds(sys.argv[1])
    
    output_targ_stats(labels)

    #FIXME FIXME
    bid_mpgnn = train_mpgnn(g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs)

    mpgnn_test_eval(bid_mpgnn,g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs)









