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
from collections import Counter

#FIXME FIXME delete
from quick_eval import *


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

            #feats[-1].append(OP_ENC[attrs[2]][attrs[5]])
            feats[-1].append([attrs[2], attrs[5]])

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

    if (total > 0):
        print('\ttest all masked score: ' + str(float(correct) / total))
    else:
        print('\tpmasks were all 0')



# evaluating accuracy only on decisions taken from max-probability heuristic
def eval_strat_tst_preds(model, graphs, labels, pred_masks): 
    correct = total = 0

    for g_idx in range(len(graphs)):
        predicts, _ = model(graphs[g_idx])  

        maxarg = None
        for op_idx in range(len(predicts)):
            if (pred_masks[g_idx][op_idx][0] == True):
                if (maxarg == None or np.amax(predicts[op_idx].detach().numpy()) > \
                                      np.amax(predicts[maxarg].detach().numpy())):
                   maxarg = op_idx

                #print(str(np.argmax(predicts[op_idx].detach().numpy())) + ", " + str(labels[g_idx][op_idx]))
        #print("")

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
def tune_prec(orig, tune_rec):
    rec = tune_rec - int((CLASSES-1)/2)

    if rec < 1:
        return max(0, orig+rec)
    else:
        return min(2, orig+rec) 

 
# balance classes through setting prediction masks
def balance_classes(pred_masks, gt_otc, steps=3):
    orig = copy.deepcopy(pred_masks)
    
    for step in range(steps):
        total =  0
        labs = []

        for op_idx in range(len(gt_otc)):
            if (pred_masks[op_idx][0] == True):
                labs.append(gt_otc[op_idx] ) 
                total += 1 

        counts = Counter(labs)

        samp_weights = {}   
        for k in counts.keys():
            samp_weights[k] = 1.0 - float(counts[k]) / total

        for op_idx in range(len(pred_masks)):
            if (pred_masks[op_idx][0] == True):
                p_keep = samp_weights[gt_otc[op_idx]]
                keep   = np.random.choice([0,1], p=[1.0-p_keep, p_keep])    
 
                if (keep == 0): 
                    pred_masks[op_idx][0] = False


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
            labels_bat = []
            pmasks_bat = [] 

            gt_otc = []

            for ex_idx in range(bat_idx * BAT_SZ, (bat_idx + 1) * BAT_SZ):         

                glob_idx = shuff_idxs[ex_idx]
                g_idx = graph_idxs[glob_idx]

                ex_feats = feats[glob_idx]

                print("g_idx: " + str(g_idx) + " feats: " + str(len(ex_feats)))

                ex_graph = create_graph(graph_edges[g_idx],\
                                        [OP_ENC[feat[0]][feat[1]] for feat in ex_feats],\
                                        unary_masks[g_idx])
                
                labels_bat += labels[glob_idx]         

                pmasks_bat += [[pred_masks[graph_idxs[glob_idx]][op_idx][0]] for \
                               op_idx in range(len(pred_masks[graph_idxs[glob_idx]]))] 

                gt_otc += [ tune_prec(ex_feats[i][1], labels[glob_idx][i]) for i in range(len(ex_feats)) ]                   
                graphs.append(ex_graph)

            balance_classes(pmasks_bat, gt_otc) 

            graphs_bat = dgl.batch(graphs) 
            predicts, _ = model(graphs_bat)           

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


    tst_graphs = []

    #FIXME FIXME
    example_count = 1024
    ex_errs = []

    # difference in prec % after tuning
    all_prec_freq_deltas = {32:[], 64:[], 80:[]}
    comp_succ_freq_deltas = {32:[], 64:[], 80:[]}
    mean_succ_freq_deltas = {32:[], 64:[], 80:[]}

    all_prec_improv = []
    comp_prec_improv = []
    mean_prec_improv = []

    all_count = avg_good_count = all_good_count =  0

    for ex_idx in range(BAT_COUNT*BAT_SZ, BAT_COUNT*BAT_SZ + example_count):         
        glob_idx = shuff_idxs[ex_idx]
        g_idx    = graph_idxs[glob_idx]
        edges    = graph_edges[g_idx] 
        ex_feats    = feats[glob_idx] # tuples of [opcode, prec]

        ex_graph = create_graph(graph_edges[g_idx],\
                                [OP_ENC[feat[0]][feat[1]] for feat in ex_feats],\
                                unary_masks[g_idx])
        tst_graphs.append(ex_graph)
         
        predicts, top_ord = model(ex_graph) 

        exec_list = []
        otc_orig       = []
        otc_tuned      = []

        for step in top_ord:
            #there may be multiple nodes at same position in top order
            for n in step:
                rec = int(np.argmax(predicts[n].detach()))
                parents = [int(v) for v in ex_graph.in_edges(n)[0]]
                if (len(parents) < 2):
                    if (len(parents) < 1): 
                        parents.append(None) 
                    parents.append(None) 

                exec_list.append([int(n), ex_feats[n][0], parents[0], parents[1]])

                otc_orig.append(precs_inv[ex_feats[n][1]])
                otc_tuned.append(precs_inv[tune_prec(ex_feats[n][1], rec)])
                            
                #counts[ tune_prec(ex_feats[n][1], rec) ] += 1                
                #print(str(exec_list_orig[-1]))
                #print(" ->p = " + str(rec) + " -> "+  str(exec_list_tun[-1]))            
        #print("")
       

        #
        # run tuned otc
        #
        inputs = gen_stratified_inputs(exec_list, input_samp_sz, inputs_mag) 

        errs = []
        gt_otc = gen_spec_otc(exec_list, precs_inv[2])

        # discard test program if shadow not valid
        valid = True
        for ins in inputs:
            result = sim_prog(exec_list, ins, otc_tuned)
            shad_result = sim_prog(exec_list, ins, gt_otc) 
            err = relative_error(result, shad_result)
            errs.append(err)
            if (shad_result == None):
                valid = False
                break
        if not (valid):
            continue

        if (np.mean(errs) < err_thresh):
            avg_good_count += 1
        if (np.amax(errs) < err_thresh):
            all_good_count += 1
        all_count += 1

        print("**avg rel error for tuned otc: " + str(np.mean(errs)))
        print("**max err for tuned otc: " + str(np.amax(errs)) + "\n")
        ex_errs.append(np.mean(errs)) 
    
        # precision counts
        total = len(otc_tuned)

        orig_counts = {32:0, 64:0, 80:0}
        for prec in otc_orig:
            orig_counts[prec] += 1

        tun_counts = {32:0, 64:0, 80:0}
        for prec in otc_tuned:
            tun_counts[prec] += 1                


        # only do counts for tuned programs that succeed        
        for k in tun_counts.keys():
            all_prec_freq_deltas[k].append(float(tun_counts[k])/total - float(orig_counts[k])/total)
            all_prec_improv.append( np.sum(otc_orig) - np.sum(otc_tuned) )

            if (np.mean(errs) < err_thresh):
                mean_succ_freq_deltas[k].append(float(tun_counts[k])/total - float(orig_counts[k])/total)
                mean_prec_improv.append( np.sum(otc_orig) - np.sum(otc_tuned) )

                if (np.amax(errs) < err_thresh):
                    comp_succ_freq_deltas[k].append(float(tun_counts[k])/total - float(orig_counts[k])/total)    
                    comp_prec_improv.append( np.sum(otc_orig) - np.sum(otc_tuned) )
    
    print("\n\t\t**proportion within thresh, on average across inputs " + str(float(avg_good_count) / float(all_count))) 
    print("\t\t**proportion within thresh, for all inputs " + str(float(all_good_count) / float(all_count))) 

    print("\tall avg precision proportion deltas ")
    for k in all_prec_freq_deltas.keys():
        print(str(k) + " : " + str(np.mean(all_prec_freq_deltas[k]))) 
    print("\n\tcomplete sol avg precision proportion deltas ")
    for k in comp_succ_freq_deltas.keys():
        print(str(k) + " : " + str(np.mean(comp_succ_freq_deltas[k]))) 
    print("\n\tmean sol avg precision proportion deltas ")
    for k in mean_succ_freq_deltas.keys():
        print(str(k) + " : " + str(np.mean(mean_succ_freq_deltas[k]))) 

    print("\n\n\tall avg precision improvement " + str(np.mean(all_prec_improv)))
    print("\tcomplete sol avg precision improvement " + str(np.mean(comp_prec_improv)))
    print("\tmean sol avg precision improvement " + str(np.mean(mean_prec_improv)))


    #for p in counts.keys():
    #    print(str(p) + " - " + str(counts[p]))
     
    tst_graphs_bat  = dgl.batch(tst_graphs)

    #FIXME ensure ordering over graphs is consistent
    tst_labels_bat = []
    tst_pmask_bat = [] 

    tst_labels = []
    tst_pmask = []

 
    for ex_idx in range(BAT_COUNT*BAT_SZ, BAT_COUNT*BAT_SZ+example_count):
        glob_idx = shuff_idxs[ex_idx]

        tst_labels_bat += labels[glob_idx]         
        tst_pmask_bat += pred_masks[graph_idxs[glob_idx]] 

        tst_labels.append(labels[glob_idx])
        tst_pmask.append(pred_masks[graph_idxs[glob_idx]])
       
    tst_labels_bat = th.tensor(tst_labels_bat)
    predicts, _   = model(tst_graphs_bat) 
   
    eval_all_tst_preds(predicts, tst_labels_bat, tst_pmask_bat)
    eval_strat_tst_preds(model, tst_graphs, tst_labels, tst_pmask) 

    
      


#
if __name__ == '__main__':
    assert(len(sys.argv) > 1), "missing path to ds"

    treval_fptc_gnn()






