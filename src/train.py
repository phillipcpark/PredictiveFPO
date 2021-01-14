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

from tr_helper import *
from fp_funcs import *
from apps import *


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
    
    # indexed by example, graphs concat on axis (mapped to graphs using idxs) 
    feats = [[]]
    labels  = [[]]
    g_idxs  = []

    curr_g_idx    = 0
    curr_edges    = [] 
    curr_is_unary = []


    for row in reader:        
        if (row == GRAPH_DELIM):         
            # was new graph
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

    return g_edges, feats, labels, unary_masks, g_idxs

# evaluate predictions
def pred_acc(predicts, labels):

    correct = total = 0
    for p_idx in range(len(predicts)):
        if (labels[p_idx].detach() == IGNORE_CLASS):
            continue  

        pred_class = np.argmax(np.array(predicts[p_idx].detach())) 
        true_class = np.array(labels[p_idx].detach())

        if (pred_class == true_class):
            correct += 1
        total += 1            

    if (total > 0):
        return float(correct) / total
    else:
        print('\tpmasks were all 0')
        return 0.0


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


#
def get_class_weights(labels):
    counts = Counter(labels)
    weights = []

    total = 0

    for k in counts.keys():
        if not (k == 6):
            total += counts[k]        

    for i in range(CLASSES):
        if (i not in counts.keys()):
            weights.append(1.0)
        else:
            #weights.append(float(abs(np.log(float(total) / (CLASSES * counts[i])))))      
            weights.append(float(total) / (CLASSES * counts[i]))      

    return th.tensor(weights)



#
def treval_fptc_gnn():
    graph_edges, feats, labels, unary_masks, graph_idxs = load_ds(sys.argv[1])

    shuff_idxs = np.arange(len(feats)) 
    random.shuffle(shuff_idxs)

    example_count = len(feats)
    BAT_COUNT = int((example_count * TR_DS_PROP) / BAT_SZ) 
    VALID_SZ  = int(example_count * VAL_DS_PROP)  

    #fwd_model = fwd_gnn(OP_ENC_DIM, H_DIM, LAYERS)
    #bwd_model = bwd_gnn(OP_ENC_DIM, H_DIM, LAYERS)
    #comb_model = comb_state(OP_ENC_DIM, H_DIM, LAYERS, CLASSES)

    fwd_model = fwd_gnn_resid(OP_ENC_DIM, H_DIM)
    bwd_model = bwd_gnn_resid(OP_ENC_DIM, H_DIM)
    comb_model = comb_state_resid(OP_ENC_DIM, H_DIM)

    #fwd_model = fwd_gnn_dense(OP_ENC_DIM, H_DIM)
    #bwd_model = bwd_gnn_dense(OP_ENC_DIM, H_DIM)
    #comb_model = comb_state_dense(OP_ENC_DIM, H_DIM)

    optimizer = th.optim.Adagrad(list(fwd_model.parameters()) + list(bwd_model.parameters()) + list(comb_model.parameters()), lr=L_RATE)
    #optimizer = th.optim.Adagrad(bwd_model.parameters(), lr=L_RATE)


    for epoch in range(EPOCHS):
        ep_loss = []
        ep_acc  = []

        #graphs in each batch combined into single monolithic graph
        for bat_idx in range(BAT_COUNT):            
            optimizer.zero_grad()

            graphs = []
            labels_bat = []

            for ex_idx in range(bat_idx * BAT_SZ, (bat_idx + 1) * BAT_SZ):         
                glob_idx = shuff_idxs[ex_idx]
                g_idx = graph_idxs[glob_idx]

                ex_feats = feats[glob_idx]
                ex_graph = create_graph(graph_edges[g_idx],\
                                        [OP_ENC[feat[0]][feat[1]] for feat in ex_feats],\
                                        unary_masks[g_idx])
                
                labels_bat += labels[glob_idx]         

                graphs.append(ex_graph)

            graphs_bat = dgl.batch(graphs) 
            rev_bat = dgl.batch([g.reverse(share_ndata=True, share_edata=True) for g in graphs])

            fwd_predicts, _ = fwd_model(graphs_bat)
            bwd_predicts, _ = bwd_model(rev_bat)
            predicts        = comb_model(fwd_predicts, bwd_predicts)
            #predicts = bwd_predicts

            # embeddings
            #if (epoch == EPOCHS - 1 and bat_idx == 0):
            #    for p_idx in range(len(bwd_predicts)):
            #        if not(labels_bat[p_idx] == IGNORE_CLASS):
            #            print(str(bwd_predicts[p_idx][0].item()) + "," + str(bwd_predicts[p_idx][1].item()))
            #    sys.exit(0)

                
            #class_weights = get_class_weights(labels_bat) 

            #comp_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_CLASS, size_average=True) 
            comp_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS, size_average=True) 

            loss      = comp_loss(predicts, th.tensor(labels_bat))
         
            ep_loss.append(loss.item())
            ep_acc.append(pred_acc(predicts, th.tensor(labels_bat)))

            loss.backward() 
            optimizer.step()
            #print("**Epoch {:05d} | Loss {:.16f} |".format(epoch, loss.item()))

        # validation set
        graphs = []
        labels_bat = []
        for ex_idx in range(BAT_COUNT * BAT_SZ, BAT_COUNT * BAT_SZ + VALID_SZ):         
            glob_idx = shuff_idxs[ex_idx]
            g_idx = graph_idxs[glob_idx]

            ex_feats = feats[glob_idx]
            ex_graph = create_graph(graph_edges[g_idx],\
                                    [OP_ENC[feat[0]][feat[1]] for feat in ex_feats],\
                                    unary_masks[g_idx])
            
            labels_bat += labels[glob_idx]         
            graphs.append(ex_graph)

        graphs_bat = dgl.batch(graphs) 
        rev_bat = dgl.batch([g.reverse(share_ndata=True, share_edata=True) for g in graphs])

        fwd_predicts, _ = fwd_model(graphs_bat)
        bwd_predicts, _ = bwd_model(rev_bat)
        predicts        = comb_model(fwd_predicts, bwd_predicts)

        comp_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS, size_average=True) 

        val_loss      = comp_loss(predicts, th.tensor(labels_bat))
        val_acc = pred_acc(predicts, th.tensor(labels_bat))

        if (epoch == 0):
            print("\nepoch,tr_loss,tr_acc,val_loss,val_acc")   
                
        print(str(epoch) + "," + str(np.mean(ep_loss)) + "," + str(np.mean(ep_acc)) + \
              "," + str(val_loss.item()) + "," + str(val_acc))

    tst_graphs = []
    rev_graphs = []

    #FIXME FIXME
    example_count = 1024
    ex_errs = []
    all_errs = []

    # difference in prec % after tuning
    all_prec_freq_deltas = {32:[], 64:[], 80:[]}
    comp_succ_freq_deltas = {32:[], 64:[], 80:[]}
    mean_succ_freq_deltas = {32:[], 64:[], 80:[]}

    all_prec_improv = []
    comp_prec_improv = []
    mean_prec_improv = []

    all_count = avg_good_count = all_good_count =  0

    tune_counts = {0:0, 1:0, 2:0, 3:0, 4:0}

    for ex_idx in range(BAT_COUNT*BAT_SZ + VALID_SZ, BAT_COUNT*BAT_SZ + VALID_SZ + example_count):         
        glob_idx = shuff_idxs[ex_idx]
        g_idx    = graph_idxs[glob_idx]
        edges    = graph_edges[g_idx] 
        ex_feats    = feats[glob_idx] # tuples of [opcode, prec]

        ex_graph = create_graph(graph_edges[g_idx],\
                                [OP_ENC[feat[0]][feat[1]] for feat in ex_feats],\
                                unary_masks[g_idx])
        rev_g = ex_graph.reverse(share_ndata=True, share_edata=True)

        tst_graphs.append(ex_graph)
        rev_graphs.append(rev_g)
         
        fwd_embeds, fwd_top_ord = fwd_model(ex_graph) 
        bwd_embeds, bwd_top_ord = bwd_model(rev_g) 

        #NOTE sigmoid pulled out from model
        predicts = th.sigmoid(comb_model(fwd_embeds, bwd_embeds))
        #predicts = bwd_embeds
 
        # since softmax is taken out of model
        sm = nn.Softmax(dim=-1)
        predicts = sm(predicts)

        exec_list = []
        otc_orig       = []
        otc_tuned      = []

        for step in fwd_top_ord:
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
                                   

        #
        # run tuned otc
        #
        inputs = gen_stratified_inputs(exec_list, input_samp_sz, inputs_mag) 

        errs = []
        triv_errs = []

        gt_otc = gen_spec_otc(exec_list, precs_inv[2])
        triv_otc = gen_spec_otc(exec_list, precs_inv[0])

        # discard test program if shadow not valid
        valid = True
        for ins in inputs:
            result = sim_prog(exec_list, ins, otc_tuned)
            shad_result = sim_prog(exec_list, ins, gt_otc) 
            triv_result = sim_prog(exec_list, ins, triv_otc)

            err = relative_error(result, shad_result)
            errs.append(err)

            triv_err = relative_error(triv_result, shad_result)
            triv_errs.append(triv_err)

            if (shad_result == None):
                valid = False
                break
        if not (valid):
            continue

        # NOTE skip programs that have trivial solution for given threshold
        if (np.amax(triv_errs) < err_thresh):
            continue

        if (np.mean(errs) < err_thresh):
            avg_good_count += 1
        if (np.amax(errs) < err_thresh):
            all_good_count += 1
        all_count += 1


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

                # FIXME FIXME FIXME
                #else:
                #    all_errs.append(errs)

    # FIXME printing errs
    #print("\n" + str(len(all_errs)))
    #for _e in range(len(all_errs)):
    #    print(str(_e) + ',', end='') 
    #print("")

    #for _e in range(len(all_errs[0])):
    #    for e_idx in range(len(all_errs)):
    #        print(str(all_errs[e_idx][_e]) + ",", end='')
    #    print("")
    
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


    print("\ntune counts")
    for key in tune_counts.keys():
        print(str(key) + " " + str(tune_counts[key]))

     
    tst_graphs_bat  = dgl.batch(tst_graphs)
    rev_graphs_bat = dgl.batch(rev_graphs)

    tst_labels_bat = []
 
    for ex_idx in range(BAT_COUNT*BAT_SZ+VALID_SZ, BAT_COUNT*BAT_SZ+VALID_SZ +example_count):
        glob_idx = shuff_idxs[ex_idx]
        tst_labels_bat += labels[glob_idx]         
       
    tst_labels_bat = th.tensor(tst_labels_bat)

    fwd_embeds, _   = fwd_model(tst_graphs_bat) 
    bwd_embeds, _   = bwd_model(rev_graphs_bat)
    predicts        = sm(th.sigmoid(comb_model(fwd_embeds, bwd_embeds)))  
    #predicts = bwd_embeds
 
    print("\n**test accuracy: " + str(pred_acc(predicts, tst_labels_bat)))

    #
    # eval on FPBench apps
    #

    #in_count = 32
    #jet_edges, jet_ops, jet_is_unary, jet_consts = jet_app() 

    #non_consts = len([True for op in jet_ops if not(op == 0)])   
    #inputs = [[rand.uniform(-5.0, 5.0), rand.uniform(-20.0, 5.0)] + jet_consts + \
    #          [0.0 for o in range(non_consts)] for i in range(input_samp_sz)]

    #in_otcs  = [gen_rand_otc(len(jet_ops)) for _in in range(in_count)]

    #all_errs = []

    #for in_otc in in_otcs:
    #    curr_feats = [[jet_ops[op_idx], in_otc[op_idx]] for op_idx in range(len(jet_ops))] 

    #    ex_graph = create_graph(jet_edges,\
    #                            [OP_ENC[feat[0]][feat[1]] for feat in curr_feats],\
    #                            jet_is_unary)
    #    rev_g = ex_graph.reverse(share_ndata=True, share_edata=True)

    #    fwd_embeds, fwd_top_ord = fwd_model(ex_graph) 
    #    bwd_embeds, _           = bwd_model(rev_g)
    #    predicts                = comb_model(fwd_embeds, bwd_embeds)  

    #    exec_list = []
    #    otc_orig  = []
    #    otc_tuned = []

    #    for step in fwd_top_ord:
    #        for n in step:
    #            rec     = int(np.argmax(predicts[n].detach()))
    #            parents = [int(v) for v in ex_graph.in_edges(n)[0]]
    #            if (len(parents) < 2):
    #                if (len(parents) < 1): 
    #                    parents.append(None) 
    #                parents.append(None) 

    #            exec_list.append([int(n), curr_feats[n][0], parents[0], parents[1]])
    #            otc_orig.append(precs_inv[curr_feats[n][1]])
    #            otc_tuned.append(precs_inv[tune_prec(curr_feats[n][1], rec)])

    #    errs = []
    #    gt_otc = gen_spec_otc(exec_list, precs_inv[2])
    #    valid = True

    #    for ins in inputs:
    #        result = sim_prog(exec_list, ins, otc_tuned)
    #        shad_result = sim_prog(exec_list, ins, gt_otc) 
    #        err = relative_error(result, shad_result)
    #        errs.append(err)
    #        if (shad_result == None):
    #            valid = False
    #            break
    #    if not (valid):
    #        continue

    #    all_errs.append(np.mean(errs))


#
if __name__ == '__main__':
    assert(len(sys.argv) > 1), "missing path to ds"

    treval_fptc_gnn()






