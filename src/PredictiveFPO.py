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
    f_hand = open(path + "/ds.csv", 'r')  
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
                classes = 5 #NOTE hardcoded: assumes DS has 5 classes
                gt_prec = max(attrs[5] + (attrs[6] - int(classes-1)/int(2)), 0) 

                if (SP_TARGET):
                    #FIXME FIXME FIXME should label be '0' if init type is '0'? 
                    labels[-1].append(1 if (gt_prec == 0) else 0)
                else:
                    labels[-1].append(1 if ((attrs[5] + (attrs[6] - int(classes-1)/int(2))) < attrs[5]) else 0)

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


# estimate P(y|depth)
def analyze_depth(g_edges, g_idxs, feats, unary_masks, labels, shuff_idxs):

    # delineates the overall height of graphs
    overall_depths_sp = {}
    overall_depths_all = {}

    for i in range(35): 
        d_sp_count  = {}
        d_all_count = {}
        for j in range(35):
            d_sp_count[j] = 0
            d_all_count[j] = 0
        overall_depths_sp[i] = d_sp_count
        overall_depths_all[i] = d_all_count            
      
    g_idx = None
    for _idx in range(len(shuff_idxs)):
        ex_idx = shuff_idxs[_idx]
        if (g_idxs[ex_idx] == g_idx):
            continue
        g_idx = g_idxs[ex_idx]

        g = batch_graphs_from_idxs([0], [g_edges[g_idx]], [unary_masks[g_idx]], [0],[feats[ex_idx]], use_gpu=False)        

        curr_d = 0
        top_ord = dgl.topological_nodes_generator(g)   
        for d in top_ord:
            curr_d += 1

        g_d    = curr_d
        curr_d = 0
        for d in top_ord:
            for node in d:
                n = node.detach().numpy()                
                if not(feats[ex_idx][n][0] == 0):                 
                    overall_depths_all[g_d][curr_d] += 1

                    if (labels[ex_idx][n] == 1 and not(feats[ex_idx][n][1] == 0)):
                        overall_depths_sp[g_d][curr_d] += 1
            curr_d += 1         

    # print per graph sz
    #for g_d in overall_depths_sp.keys():
    #    print("\n**graphs w/ depth " + str(g_d))

    #    for indiv_d in overall_depths_sp[g_d].keys():
    #        if (overall_depths_all[g_d][indiv_d] == 0):
    #            continue 
    #        print("\t" + str(indiv_d) + ": " + str(float( overall_depths_sp[g_d][indiv_d] ) / overall_depths_all[g_d][indiv_d]) + \
    #              " -total: " + str(overall_depths_all[g_d][indiv_d]))           

    # print for alll graphs 
    comb_depths_sp = {}
    comb_depths_all = {}

    for i in range(40): 
        comb_depths_sp[i] = 0
        comb_depths_all[i] = 0

    for g_d in overall_depths_sp.keys():
        for indiv_d in overall_depths_sp[g_d].keys():
            comb_depths_sp[indiv_d] += overall_depths_sp[g_d][indiv_d] 
            comb_depths_all[indiv_d] += overall_depths_all[g_d][indiv_d] 

    total_nodes = 0
    for k in comb_depths_all.keys():
        total_nodes += comb_depths_all[k]

    print("depth,prop_sp,prop_all")
    for depth in comb_depths_sp.keys():
        if (comb_depths_all[depth] == 0):
            continue
        print(str(depth) + "," + str(float(comb_depths_sp[depth]) / comb_depths_all[depth]) + ", " + str(float(comb_depths_all[depth])/total_nodes))
 


#
if __name__ == '__main__':
    assert(len(sys.argv) > 1), "missing path to ds"

    g_edges = feats =  labels = unary_masks = g_idxs = shuff_idxs = None
   
    if (SINGLE_GRAPH_TARG):
        g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs = load_nootcfeat_ds(sys.argv[1])
    else: 
        g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs = load_predfpo_ds(sys.argv[1])

    
    #output_targ_stats(labels)
    #analyze_depth(g_edges, g_idxs, feats, unary_masks, labels, shuff_idxs)
    #analyze_depth(g_edges, g_idxs, feats, unary_masks, labels, shuff_idxs)


    bid_mpgnn = train_mpgnn(g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs)
    mpgnn_test_eval(bid_mpgnn,g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs, sys.argv[1])









