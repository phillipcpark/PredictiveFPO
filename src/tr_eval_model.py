from nn_model import *
from eval_metrics import *
from otc import *
from prog_inputs import *
from params import *
from prog_sim import *

from collections import Counter
import sys

import torch as th

# 
def create_dgl_graph(edges, feats, is_unary_op):
    edge_src    = th.tensor([e[0] for e in edges])
    edge_target = th.tensor([e[1] for e in edges])    
   
    graph = dgl.DGLGraph((edge_src, edge_target))

    graph.ndata['node_feats']  = th.tensor(feats)
    graph.ndata['is_unary_op'] = th.tensor(is_unary_op)
 
    return graph


# g_edges and unary_masks are indexed by graph; feats and labels by example
def batch_graphs_from_idxs(idxs, g_edges, unary_masks, g_idxs, feats, labels=None):
   
    graphs_list = []
    labels_bat = []

    #FIXME FIXME
    feat_counts = []
    label_counts = []
    

    for ex_idx in idxs:         
        g_idx    = g_idxs[ex_idx]
        edges    = g_edges[g_idx] 
        ex_feats = feats[ex_idx] 


        feat_counts.append(len(ex_feats))
        label_counts.append(len(labels[ex_idx]))


        ex_graph = None
        if (SINGLE_GRAPH_TARG):
            ex_graph = create_dgl_graph(g_edges[g_idx],\
                                        [OP_ENC_NOPREC[feat] for feat in ex_feats],\
                                         unary_masks[g_idx])
        else:
            ex_graph = create_dgl_graph(g_edges[g_idx],\
                                    [OP_ENC[feat[0]][feat[1]] for feat in ex_feats],\
                                     unary_masks[g_idx])        
        
        graphs_list.append(ex_graph)

        if not(labels == None):
            labels_bat += labels[ex_idx]         
            
    graphs_bat = dgl.batch(graphs_list) 
    labels_bat = th.tensor(labels_bat) if not(labels == None) else None

    if not(labels == None):
        return graphs_bat, labels_bat
    return graphs_bat


#
def rev_graph_batch(graphs):
    rev_bat = dgl.batch([g.reverse(share_ndata=True, share_edata=True) for g in graphs])
    return rev_bat
    

#
def get_class_weights(labels):
    counts = Counter(labels) #labels.detach().numpy())
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
            weights.append(1.0 / counts[i])

            #weights.append( (1.0 - beta) / (1.0 - beta**counts[i]) )
            #weights.append(float(total) / (CLASSES * counts[i]))      

    norm_const = np.sum(weights)
    weights = [float(w/norm_const) for w in weights]

    return th.tensor(weights)

# 
def undersamp_classes(labels, steps=2):
    bal_labels = labels.detach().numpy() 

    for i in range(steps):
        total =  len([l for l in bal_labels if not(l == IGNORE_CLASS)])
        counts = Counter(bal_labels)

        min_cl = counts[0]
        for c in range(1, CLASSES):
            if (counts[c] < min_cl):
                min_cl = c

        new_bal_labels = []

        # random weighted filtering 
        samp_weights = {}   

        for k in counts.keys():
            if (k == IGNORE_CLASS): 
                continue
            samp_weights[k] = 1.0 - float(counts[k]) / total

        for lab_idx in range(len(bal_labels)):
            if not(bal_labels[lab_idx] == IGNORE_CLASS):
                if (bal_labels[lab_idx] == min_cl):
                    new_bal_labels.append(bal_labels[lab_idx])
                    continue

                p_keep = samp_weights[bal_labels[lab_idx]]
                keep   = np.random.choice([0,1], p=[1.0-p_keep, p_keep])                
 
                if (keep == 0):
                    new_bal_labels.append(IGNORE_CLASS) 
                else:
                    new_bal_labels.append(bal_labels[lab_idx]) 
            else:
                new_bal_labels.append(IGNORE_CLASS)    

        bal_labels = new_bal_labels 
    return th.tensor(bal_labels)


#
def get_dev():
    dev = None

    print("\n** CUDA: " + str(th.cuda.is_available()))

    if th.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return th.device(dev)



# check whether or not all-SP and gt solutions work, as well as if GT throws exception
def check_init_sols(exec_list, inputs, orig_otc):

    triv_errs = []
    orig_errs = []
    
    gt_otc   = gen_spec_otc(exec_list, precs_inv[2])
    triv_otc = gen_spec_otc(exec_list, precs_inv[0])

    # skip if trivial result viable
    for ins in inputs:
        shad_result = sim_prog(exec_list, ins, gt_otc) 

        if (shad_result == None):
            #print("\tground truch OTC threw exception")
            return False

        triv_result = sim_prog(exec_list, ins, triv_otc)
        orig_result = sim_prog(exec_list, ins, orig_otc)

        if (triv_result == None):
            return True
        else:
            triv_errs.append(relative_error(triv_result, shad_result))

        # FIXME FIXME FIXMe should we forgo discarding orig OTCs that throw exceptions?
        if (orig_result == None):
            return False
        else:
            orig_errs.append(relative_error(orig_result, shad_result))            
 
    if (np.amax(triv_errs) < err_thresh):
        #print("\ttrivial solution worked for all inputs " + str(np.amax(triv_errs)))
        return False

    accept, gt_thresh_prop = accept_err(orig_errs)
    if not(accept):
        #print("\torig solution didn't work for prop of inputs")
        return False

    accept, gt_thresh_prop = accept_err(triv_errs)
    if (accept):
        #print("\ttrivial worked for prop of inputs")        
        return False   
    return True


#
def update_freq_delta(prec_freq_deltas, otc_tuned, otc_orig):    
    # precision counts
    total = len(otc_tuned)

    orig_counts = {32:0, 64:0, 80:0}
    for prec in otc_orig:
        orig_counts[prec] += 1

    prec_counts = {32:0, 64:0, 80:0} 
    for prec in otc_tuned:
        prec_counts[prec] += 1                

    for k in prec_counts.keys():
        prec_freq_deltas[k].append(float(prec_counts[k])/total - float(orig_counts[k])/total)



    
#
#
#
def train_mpgnn(g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs):

    print("\n*********training phase*********")

    example_count = len(feats)
    BAT_COUNT = int((example_count * TR_DS_PROP) / BAT_SZ) 
    VALID_SZ  = int(example_count * VAL_DS_PROP)  

    feat_dim = OP_ENC_NOPREC_DIM if SINGLE_GRAPH_TARG else OP_ENC_DIM
    gnn = bid_mpgnn(feat_dim, H_DIM, CLASSES)

    optimizer = th.optim.Adagrad(gnn.parameters()) 
    
    for epoch in range(EPOCHS):
        ep_loss = []
        ep_acc  = []

        ep_prec = []
        ep_rec  = []

        #graphs in each batch combined into single monolithic graph
        for bat_idx in range(BAT_COUNT):            
            optimizer.zero_grad()

            bat_idxs                = shuff_idxs[bat_idx*BAT_SZ : (bat_idx+1)*BAT_SZ] 
            graphs_bat, labels_bat  = batch_graphs_from_idxs(bat_idxs, g_edges, unary_masks, g_idxs, 
                                                                    feats, labels)        
            predicts, _ = gnn(graphs_bat)
            
            comp_loss = None 
            if (USE_CL_BAL):
                class_weights = get_class_weights(labels_bat.detach().numpy()) 
                comp_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_CLASS, size_average=True) 
            else:
                comp_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS, size_average=True) 

            # FIXME label filtering
            #labels_bat = undersamp_classes(labels_bat) 

            prec, rec = prec_recall(predicts, labels_bat)
            ep_prec.append(prec)
            ep_rec.append(rec)

            loss      = comp_loss(predicts, labels_bat)
                     
            ep_loss.append(loss.item())
            ep_acc.append(pred_acc(predicts, th.tensor(labels_bat)))

            loss.backward() 
            optimizer.step()

        val_idxs   = shuff_idxs[BAT_COUNT*BAT_SZ : BAT_COUNT*BAT_SZ + VALID_SZ]        
        val_graphs_bat, val_labels = batch_graphs_from_idxs(val_idxs, g_edges, unary_masks,\
                                                                    g_idxs, feats, labels)
        val_predicts, _ = gnn(val_graphs_bat)

        comp_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS, size_average=True) 
        val_loss  = comp_loss(val_predicts, val_labels)
        val_acc   = pred_acc(val_predicts, val_labels)

        if (epoch == 0):
            print("\nepoch,tr_loss,tr_acc,val_loss,val_acc")                   

        print("\n" + str(epoch) + "," + str(np.mean(ep_loss)) + "," + str(np.mean(ep_acc)) + \
              "," + str(val_loss.item()) + "," + str(val_acc))

        # avg prec-rec
        avg_prec = {}
        avg_rec  = {}
        for c in range(CLASSES):
            avg_prec[c] = []
            avg_rec[c]  = [] 
        for i in range(BAT_COUNT):
            for c in range(CLASSES):            
                avg_prec[c].append(ep_prec[i][c])
                avg_rec[c].append(ep_rec[i][c]) 
        for c in range(CLASSES):            
            print("\t" + str(c) + ": " + str(np.mean(avg_prec[c])) + ", " + str(np.mean(avg_rec[c])))

    return gnn 


#
def mpgnn_test_eval(gnn, g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs):

    print("\n**********test phase*************")

    example_count = 1024 #512 #1024

    BAT_COUNT = int((example_count * TR_DS_PROP) / BAT_SZ) 
    VALID_SZ  = int(example_count * VAL_DS_PROP)  

    # difference in prec % after tuning
    prec_freq_deltas = {32:[], 64:[], 80:[]}
    all_count = avg_good_count = all_good_count = all_prop_count =  0

    tune_counts = None
    if (SINGLE_GRAPH_TARG):
        tune_counts = {0:0, 1:0, 2:0} 
    else:
        if (COARSE_TUNE):
            tune_counts = {0:0, 1:0}
        else:
            tune_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    tst_precs = []
    tst_recs = []

    for shuff_idx in range(BAT_COUNT*BAT_SZ + VALID_SZ, BAT_COUNT*BAT_SZ + VALID_SZ + example_count):         

        if (shuff_idx % 200 == 0):#100 == 0):
            print("tst_idx " + str(shuff_idx - (BAT_COUNT*BAT_SZ + VALID_SZ)))

        ex_idx   = shuff_idxs[shuff_idx] 
        g_idx    = g_idxs[ex_idx]
        ex_feats = feats[ex_idx] # tuples of [opcode, prec]

        ex_graph, ex_label = batch_graphs_from_idxs([ex_idx], g_edges, unary_masks, g_idxs, 
                                                     feats, labels)        
        
        predicts, top_order = gnn(ex_graph)

        sm = nn.Softmax(dim=-1)
        predicts = sm(th.sigmoid(predicts))

        exec_list = []
        otc_orig  = []
        otc_tuned = []

        for step in top_order:
            for n in step:
                rec = int(np.argmax(predicts[n].detach()))
                parents = [int(v) for v in ex_graph.in_edges(n)[0]]
                if (len(parents) < 2):
                    if (len(parents) < 1): 
                        parents.append(None) 
                    parents.append(None) 
 
                if (SINGLE_GRAPH_TARG):
                    exec_list.append([int(n), ex_feats[n], parents[0], parents[1]])
                    otc_tuned.append(precs_inv[rec])
                else:
                    exec_list.append([int(n), ex_feats[n][0], parents[0], parents[1]])
                    otc_orig.append(precs_inv[ex_feats[n][1]])              

                    if (COARSE_TUNE):

                        #FIXME FIXME
                        if (SP_TARGET):
                            otc_tuned.append(32 if rec==1 else precs_inv[ex_feats[n][1]])
                        else: #tune down 1 type
                            otc_tuned.append(precs_inv[max(0, ex_feats[n][1]-1)] if rec==1 else precs_inv[ex_feats[n][1]])
 
                    else:
                        otc_tuned.append(precs_inv[tune_prec(ex_feats[n][1], rec)])
                                          
        # run tuned otc
        inputs = gen_stratified_inputs(exec_list, input_samp_sz, inputs_mag) 


        # skip example if either trivial sol works, shadow throws except, orig sol throws except         
        if not(check_init_sols(exec_list, inputs, otc_orig)):
            continue


        # FIXME prec-recall on each test ex
        prec, rec = prec_recall(predicts, ex_label)
        tst_precs.append(prec)
        tst_recs.append(rec)

        ex_errs = []

        gt_otc = gen_spec_otc(exec_list, precs_inv[2])

        for ins in inputs:
            result      = sim_prog(exec_list, ins, otc_tuned)
            shad_result = sim_prog(exec_list, ins, gt_otc) 
            ex_errs.append(relative_error(result, shad_result))

        if (np.mean(ex_errs) < err_thresh):
            avg_good_count += 1
        if (np.amax(ex_errs) < err_thresh):
            all_good_count += 1
        all_count += 1

        # proportion based acceptance
        accept, gt_thresh_prop = accept_err(ex_errs)
        if (accept):
            all_prop_count += 1

        for pred in predicts:
            tune_counts[np.argmax(pred.detach().numpy())] += 1

        if not(SINGLE_GRAPH_TARG):
            update_freq_delta(prec_freq_deltas, otc_tuned, otc_orig)   

    print("\n**test prog count: " + str(all_count) + "\n")     
    print("**proportion within thresh, on average across inputs " + str(float(avg_good_count) / float(all_count))) 
    print("**proportion within thresh, for all inputs " + str(float(all_good_count) / float(all_count))) 
    print("**proportion within thresh, for proportion of inputs " + str(float(all_prop_count) / float(all_count)))

    print("\ntune counts")
    for key in tune_counts.keys():
        print(str(key) + " " + str(tune_counts[key]))

    print("\navg prec-recall across tests graphs")
    avg_prec = {}
    avg_rec  = {}
    for c in range(CLASSES):
        avg_prec[c] = []
        avg_rec[c]  = [] 
    for i in range(len(tst_precs)):
        for c in range(CLASSES):            
            avg_prec[c].append(tst_precs[i][c])
            avg_rec[c].append(tst_recs[i][c]) 
    for c in range(CLASSES):            
        print("\t" + str(c) + ": " + str(np.mean(avg_prec[c])) + ", " + str(np.mean(avg_rec[c])))


    
    #tst_graphs_bat = dgl.batch(tst_graphs)
    #tst_labels_bat = []
 
    #for ex_idx in range(BAT_COUNT*BAT_SZ+VALID_SZ, BAT_COUNT*BAT_SZ+VALID_SZ +example_count):
    #    tst_labels_bat += labels[shuff_idxs[ex_idx]]                
    #tst_labels_bat = th.tensor(tst_labels_bat)

    #predicts,_ = gnn(tst_graphs_bat)
    #sm       = nn.Softmax(dim=-1)
    #predicts = sm(th.sigmoid(predicts))

    #print("\n**test accuracy: " + str(pred_acc(predicts, tst_labels_bat)))

    #
    #prec, rec = prec_recall(predicts, tst_labels_bat)

    ## avg prec-rec
    #for c in range(CLASSES):            
    #    print("\t" + str(c) + ": " + str(prec[c]) + ", " + str(rec[c]))



