import sys
import csv
import random
from params import *
from gnn import *

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
import sys
import re
import copy

#
def is_const(opcode):
    if (opcode == ops['CONST']):
        return True
    return False

#
def is_unary(opcode):
    if (opcode == ops['SIN'] or opcode == ops['COS']):
        return True
    else:
        return False
  
#
def load_ds(path):
    f_hand = open(path, 'r')  
    reader = csv.reader(f_hand, delimiter=',')
    next(reader) 

    # indexed by graph
    g_edges     = []
    unary_masks = []
    pred_masks  = []
    
    # indexed by example, graphs concat on axis (mapped to graphs using idxs) 
    feats = [[]]
    labels  = [[]]
    g_idxs  = []

    curr_g_idx    = 0
    curr_edges    = [] 
    curr_is_unary = []
    curr_p_mask   = []

    nodes = {}

    gs = 0

    for row in reader:        
        if (row == GRAPH_DELIM):         
            # was new graph
            if (len(g_edges) <= curr_g_idx):
                g_edges.append(curr_edges)
                unary_masks.append(curr_is_unary)
                pred_masks.append(curr_p_mask)

            curr_edges    = [] 
            curr_is_unary = []
            curr_p_mask   = []

            g_idxs.append(curr_g_idx)
            feats.append([])
            labels.append([])
                   
        else:       
            attrs = [int(elem) if not(elem == '') else None for elem in row ]
            curr_g_idx = attrs[0]         
 
            if (is_unary(attrs[2])):
                curr_is_unary.append(True)
                curr_edges.append([attrs[3], attrs[1]])
                curr_p_mask.append([True])

            elif is_const(attrs[2]):
                curr_is_unary.append(False)
                curr_p_mask.append([False])

            else:
                curr_is_unary.append(False)
                curr_edges.append([attrs[3], attrs[1]])
                curr_edges.append([attrs[4], attrs[1]])
                curr_p_mask.append([True])

            feats[-1].append(OP_ENC[attrs[2]][attrs[5]])
            labels[-1].append(attrs[6])

    g_edges.append(curr_edges)
    unary_masks.append(curr_is_unary)
    pred_masks.append(curr_p_mask)
    g_idxs.append(curr_g_idx)

    return g_edges, feats, labels, unary_masks, pred_masks, g_idxs

# evaluate predictions on all masked nodes
def eval_all_tst_preds(predicts, tst_labels, tst_pmask):
    correct = total = 0
    for p_idx in range(len(predicts)):
        if (tst_pmask[p_idx][0] == False):
            continue  
        pred_class = np.argmax(np.array(predicts[p_idx].detach())) 
        true_class = np.array(tst_labels[p_idx].detach())

        if (pred_class == true_class):
            correct += 1
        total += 1            
    print('test all masked score: ' + str(float(correct) / total))

# evaluating accuracy only on decisions taken from max-probability heuristic
def eval_strat_tst_preds(model, graphs, labels, pred_masks): 
    correct = total = 0

    for g_idx in range(len(graphs)):
        predicts = model(graphs[g_idx])  

        print("\n")

        maxarg = None
        for op_idx in range(len(predicts)):
            if (pred_masks[g_idx][op_idx][0] == True):
                if (maxarg == None or np.amax(predicts[op_idx].detach().numpy()) > \
                                      np.amax(predicts[maxarg].detach().numpy())):
                   maxarg = op_idx

                if (g_idx < 5):
                    print(str(predicts[op_idx].detach().numpy()) + " true-> " + str(labels[g_idx][op_idx]))

        if (maxarg == None):
            continue

        pred_class = np.argmax(predicts[maxarg].detach().numpy())
        true_class = labels[g_idx][maxarg]

        if (pred_class == true_class):
            correct += 1
        total += 1                    
           
    print("test max-prob predict score: " + str(float(correct)/total))



        

# 
def create_graph(edges, feats, is_unary_op):
    edge_src    = th.tensor([e[0] for e in edges])
    edge_target = th.tensor([e[1] for e in edges])    
   
    graph = dgl.DGLGraph((edge_src, edge_target))

    graph.ndata['node_feats']  = th.tensor(feats)
    graph.ndata['is_unary_op'] = th.tensor(is_unary_op)
    return graph

#
def treval_fptc_gnn():
    graph_edges, feats, labels, unary_masks, pred_masks, graph_idxs = load_ds(sys.argv[1])

    shuff_idxs = np.arange(len(feats)) 
    random.shuffle(shuff_idxs)

    example_count = len(feats)
    BAT_COUNT = int((example_count * TR_DS_PROP) / BAT_SZ)
    model     = fptc_gnn(OP_ENC_DIM, H_DIM)
    optimizer = th.optim.Adagrad(model.parameters(), lr=L_RATE)
          
    for epoch in range(EPOCHS):
        print("\n**epoch " + str(epoch))

        #graphs in each batch combined into single monolithic graph
        for bat_idx in range(BAT_COUNT):            
            graphs = []

            for ex_idx in range(bat_idx * BAT_SZ, (bat_idx + 1) * BAT_SZ):         
                g_idx = graph_idxs[shuff_idxs[ex_idx]]

                ex_graph = create_graph(graph_edges[g_idx],\
                                        feats[shuff_idxs[ex_idx]],\
                                        unary_masks[g_idx])
                 
                graphs.append(ex_graph)

            graphs_bat  = dgl.batch(graphs)
            curr_labels = curr_pmask = None 

            #FIXME ensure ordering over graphs is consistent
            labels_bat = []
            pmasks_bat = [] 
            for g_idx in range(bat_idx*BAT_SZ, (bat_idx+1)*BAT_SZ):
                labels_bat += labels[shuff_idxs[g_idx]]         
                pmasks_bat += pred_masks[graph_idxs[shuff_idxs[g_idx]]] 
 
            predicts = model(graphs_bat)           

            #default prediction (class 0) used for nodes that are not 'True' in the mask, so that they do not contribute to loss            
            default_preds  = th.tensor( [[True] + [False for i in range(CLASSES-1)] for node in range(len(predicts))] ) 
            default_labels = th.tensor( [0 for node in range(len(predicts))] )   
      
            predicts    = th.where(th.tensor(pmasks_bat), predicts, default_preds.float())
            curr_labels = th.where(th.squeeze(th.tensor(pmasks_bat)), th.tensor(labels_bat), default_labels) 

            comp_loss = nn.CrossEntropyLoss()
            loss      = comp_loss(predicts, th.tensor(curr_labels))

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            print("**Epoch {:05d} | Loss {:.16f} |".format(epoch, loss.item()))

    #test set
    tst_graphs = []

    for ex_idx in range(BAT_COUNT*BAT_SZ, BAT_COUNT*BAT_SZ+1024):         
        g_idx = graph_idxs[shuff_idxs[ex_idx]]

        ex_graph = create_graph(graph_edges[g_idx],\
                                feats[shuff_idxs[ex_idx]],\
                                unary_masks[g_idx])
         
        tst_graphs.append(ex_graph)

    tst_graphs_bat  = dgl.batch(tst_graphs)

    #FIXME ensure ordering over graphs is consistent
    tst_labels_bat = []
    tst_pmask_bat = [] 

    tst_labels = []
    tst_pmask = []

    for ex_idx in range(BAT_COUNT*BAT_SZ, BAT_COUNT*BAT_SZ+1024): #example_count):
        glob_idx = shuff_idxs[ex_idx]

        tst_labels_bat += labels[glob_idx]         
        tst_pmask_bat += pred_masks[graph_idxs[glob_idx]] 

        tst_labels.append(labels[glob_idx])
        tst_pmask.append(pred_masks[graph_idxs[glob_idx]])

    tst_labels_bat = th.tensor(tst_labels_bat)
    predicts   = model(tst_graphs_bat) 

    eval_all_tst_preds(predicts, tst_labels_bat, tst_pmask_bat)
    eval_strat_tst_preds(model, tst_graphs, tst_labels, tst_pmask) 

#
if __name__ == '__main__':
    assert(len(sys.argv) > 1), "missing path to ds"

    treval_fptc_gnn()






