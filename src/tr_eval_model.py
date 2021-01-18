from nn_model import *
from eval_metrics import *
from otc import *
from prog_inputs import *
from params import *
from prog_sim import *

from collections import Counter

import sys

# 
def create_dgl_graph(edges, feats, is_unary_op):
    edge_src    = th.tensor([e[0] for e in edges])
    edge_target = th.tensor([e[1] for e in edges])    
   
    graph = dgl.DGLGraph((edge_src, edge_target))

    graph.ndata['node_feats']  = th.tensor(feats)
    graph.ndata['is_unary_op'] = th.tensor(is_unary_op)
    return graph


# g_edges and unary_masks are indexed by graph; feats and labels by example
def batch_graphs(bat_sz, g_idxs, g_edges, unary_masks, feats, labels=None):
    ex_count = min(bat_sz, len(feats))
   
    graphs_bat = []
    labels_bat = []

    for ex_idx in range(ex_count):         
        g_idx    = g_idxs[ex_idx]
        ex_graph = create_dgl_graph(g_edges[g_idx],\
                                    [OP_ENC[feat[0]][feat[1]] for feat in feats[ex_idx]],\
                                     unary_masks[g_idx])        
        graphs_bat.append(ex_graph)

        if not(labels == None):
            labels_bat += labels[ex_idx]         

    graphs_bat = dgl.batch(graphs_bat) 
    labels_bat = th.tensor(labels_bat) if not (labels == None) else None

    return graphs_bat, labels_bat

#
def rev_graph_batch(graphs):
    rev_bat = [g.reverse(share_ndata=True, share_edata=True) for g in graphs]
    return rev_bat
    

#
def get_class_weights(labels):
    counts = Counter(labels)
    weights = []

    beta = 0.9
    total = 0

    for k in counts.keys():
        if not (k == 6):
            total += counts[k]        

    for i in range(CLASSES):
        if (i not in counts.keys()):
            weights.append(1.0)
        else:
            weights.append( (1.0 - beta) / (1.0 - beta**counts[i]) )
            #weights.append(float(total) / (CLASSES * counts[i]))      

    norm_const = np.sum(weights) / float(CLASSES)
    weights = [float(w/norm_const) for w in weights]

    return th.tensor(weights)


#
#
#
def train_mpgnn(g_edges, feats, labels, unary_masks, g_idxs):

    example_count = len(feats)
    BAT_COUNT = int((example_count * TR_DS_PROP) / BAT_SZ) 
    VALID_SZ  = int(example_count * VAL_DS_PROP)  

    bid_mpgnn = {'fwd': fwd_gnn_resid(OP_ENC_DIM, H_DIM),
                 'bwd': bwd_gnn_resid(OP_ENC_DIM, H_DIM),
                 'comb': comb_state_resid(OP_ENC_DIM, H_DIM)}
 
    optimizer = th.optim.Adagrad(list(bid_mpgnn['fwd'].parameters()) + list(bid_mpgnn['bwd'].parameters()) + \
                                 list(bid_mpgnn['comb'].parameters()), lr=L_RATE)

    for epoch in range(EPOCHS):
        ep_loss = []
        ep_acc  = []

        #graphs in each batch combined into single monolithic graph
        for bat_idx in range(BAT_COUNT):            
            optimizer.zero_grad()

            bat_start = bat_idx * BAT_SZ
            bat_end   = bat_start + BAT_SZ
 
            graphs_bat, labels_bat = batch_graphs(BAT_SZ, g_idxs[bat_start:bat_end], g_edges, unary_masks,\
                                                  feats[bat_start:bat_end], labels[bat_start:bat_end])
            rev_bat                = graphs_bat.reverse(share_ndata=True, share_edata=True) 

            fwd_predicts, _ = bid_mpgnn['fwd'](graphs_bat)
            bwd_predicts, _ = bid_mpgnn['bwd'](rev_bat)
            predicts        = bid_mpgnn['comb'](fwd_predicts, bwd_predicts)
                
            #class_weights = get_class_weights(labels_bat.detach().numpy()) 
            #comp_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_CLASS, size_average=True) 
            comp_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS, size_average=True) 
            loss      = comp_loss(predicts, labels_bat)
         
            ep_loss.append(loss.item())
            ep_acc.append(pred_acc(predicts, th.tensor(labels_bat)))

            loss.backward() 
            optimizer.step()

        # validation set
        val_start = BAT_COUNT*BAT_SZ
        val_end   = val_start + VALID_SZ

        val_graphs_bat, val_labels = batch_graphs(VALID_SZ, g_idxs[val_start:val_end], g_edges, unary_masks,\
                                              feats[val_start:val_end], labels[val_start:val_end])
        rev_bat                    = val_graphs_bat.reverse(share_ndata=True, share_edata=True)  

        fwd_predicts, _ = bid_mpgnn['fwd'](val_graphs_bat)
        bwd_predicts, _ = bid_mpgnn['bwd'](rev_bat)
        val_predicts    = bid_mpgnn['comb'](fwd_predicts, bwd_predicts)

        comp_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS, size_average=True) 

        val_loss = comp_loss(val_predicts, val_labels)
        val_acc  = pred_acc(val_predicts, val_labels)

        if (epoch == 0):
            print("\nepoch,tr_loss,tr_acc,val_loss,val_acc")                   
        print(str(epoch) + "," + str(np.mean(ep_loss)) + "," + str(np.mean(ep_acc)) + \
              "," + str(val_loss.item()) + "," + str(val_acc))

    return bid_mpgnn 


#
def mpgnn_test_eval(bid_mpgnn, g_edges, feats, labels, unary_masks, g_idxs):
    example_count = len(feats)
    BAT_COUNT = int((example_count * TR_DS_PROP) / BAT_SZ) 
    VALID_SZ  = int(example_count * VAL_DS_PROP)  

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
        g_idx    = g_idxs[ex_idx]
        ex_feats = feats[ex_idx] # tuples of [opcode, prec]

        ex_graph = create_dgl_graph(g_edges[g_idx],\
                                [OP_ENC[feat[0]][feat[1]] for feat in ex_feats],\
                                unary_masks[g_idx])
        rev_g = ex_graph.reverse(share_ndata=True, share_edata=True)

        tst_graphs.append(ex_graph)
        rev_graphs.append(rev_g)
         
        fwd_embeds, fwd_top_ord = bid_mpgnn['fwd'](ex_graph) 
        bwd_embeds, bwd_top_ord = bid_mpgnn['bwd'](rev_g) 

        sm = nn.Softmax(dim=-1)
        predicts = sm(th.sigmoid(bid_mpgnn['comb'](fwd_embeds, bwd_embeds)))

        for pred in predicts:
            tune_counts[np.argmax(pred.detach().numpy())] += 1

        exec_list = []
        otc_orig  = []
        otc_tuned = []

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

        # skip if trivial result viable
        for ins in inputs:
            shad_result = sim_prog(exec_list, ins, gt_otc) 
            triv_result = sim_prog(exec_list, ins, triv_otc)
            triv_err = abs(relative_error(triv_result, shad_result))
            triv_errs.append(triv_err)

        if (np.amax(triv_errs) < err_thresh):
            print("\t**trivial solution worked for test example; skipping " + str(np.amax(triv_errs)))
            continue

        # discard test program if shadow not valid
        valid = True
        for ins in inputs:
            result = sim_prog(exec_list, ins, otc_tuned)
            shad_result = sim_prog(exec_list, ins, gt_otc) 

            err = abs(relative_error(result, shad_result))
            errs.append(err)

            if (shad_result == None):
                valid = False
                break
        if not (valid):
            print("\t**shadow was invalid for test example")
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

    
    print("\n\t\t**proportion within thresh, on average across inputs " + str(float(avg_good_count) / float(all_count))) 
    print("\t\t**proportion within thresh, for all inputs " + str(float(all_good_count) / float(all_count))) 

    print("\n\tcomplete sol avg precision proportion deltas ")
    for k in comp_succ_freq_deltas.keys():
        print(str(k) + " : " + str(np.mean(comp_succ_freq_deltas[k]))) 
    print("\n\tmean sol avg precision proportion deltas ")
    for k in mean_succ_freq_deltas.keys():
        print(str(k) + " : " + str(np.mean(mean_succ_freq_deltas[k]))) 

    print("\tcomplete sol avg precision improvement " + str(np.mean(comp_prec_improv)))
    print("\tmean sol avg precision improvement " + str(np.mean(mean_prec_improv)))

    print("\ntune counts")
    for key in tune_counts.keys():
        print(str(key) + " " + str(tune_counts[key]))
    
    tst_graphs_bat  = dgl.batch(tst_graphs)
    rev_graphs_bat = dgl.batch(rev_graphs)

    tst_labels_bat = []
 
    for ex_idx in range(BAT_COUNT*BAT_SZ+VALID_SZ, BAT_COUNT*BAT_SZ+VALID_SZ +example_count):
        tst_labels_bat += labels[ex_idx]         
       
    tst_labels_bat = th.tensor(tst_labels_bat)

    fwd_embeds, _   = bid_mpgnn['fwd'](tst_graphs_bat) 
    bwd_embeds, _   = bid_mpgnn['bwd'](rev_graphs_bat)
    predicts        = sm(th.sigmoid(bid_mpgnn['comb'](fwd_embeds, bwd_embeds)))  
    #predicts = bwd_embeds
 
    print("\n**test accuracy: " + str(pred_acc(predicts, tst_labels_bat)))






