from nn_model import *
from eval_metrics import *
from otc import *
from prog_inputs import *
from params import *
from prog_sim import *

from apps import *

import scipy.stats as stats 
from collections import OrderedDict
from collections import Counter
import sys
import torch as th


#FIXME FIXME FIXME 
# -parents get misordered for non commutative ops! 


# 
def create_dgl_graph(edges, feats, is_unary_op, use_gpu):
    if (use_gpu):
        edge_src    = th.tensor([e[0] for e in edges]).to(get_dev())
        edge_target = th.tensor([e[1] for e in edges]).to(get_dev())    
    else:
        edge_src    = th.tensor([e[0] for e in edges])
        edge_target = th.tensor([e[1] for e in edges])    

   
    graph = dgl.DGLGraph((edge_src, edge_target))

    if (use_gpu):
        graph.ndata['node_feats']  = th.tensor(feats).to(get_dev())
        graph.ndata['is_unary_op'] = th.tensor(is_unary_op).to(get_dev())
    else:
        graph.ndata['node_feats']  = th.tensor(feats)
        graph.ndata['is_unary_op'] = th.tensor(is_unary_op)

 
    return graph


# g_edges and unary_masks are indexed by graph; feats and labels by example
def batch_graphs_from_idxs(idxs, g_edges, unary_masks, g_idxs, feats, use_gpu, labels=None):
   
    graphs_list = []
    labels_bat = []
 
    for ex_idx in idxs:         
        g_idx    = g_idxs[ex_idx]
        edges    = g_edges[g_idx] 

        ex_feats = feats[ex_idx]

        ex_graph = None

        if (SINGLE_GRAPH_TARG):
            ex_feats = [OP_ENC_NOPREC[feat] for feat in ex_feats]
            ex_graph = create_dgl_graph(g_edges[g_idx],\
                                        ex_feats,\
                                        unary_masks[g_idx],use_gpu)
        else:
            ex_feats = [OP_ENC[feat[0]][feat[1]] for feat in ex_feats]
            ex_graph = create_dgl_graph(g_edges[g_idx],\
                                        ex_feats,\
                                        unary_masks[g_idx],use_gpu)        
 
        if (use_gpu):
            ex_graph = ex_graph.to(get_dev())

                          
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
        if not (k == IGNORE_CLASS):
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

    if th.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
        print("\n\t**CUDA unavailable")
        sys.exit(0)

    return th.device(dev)



# check whether or not all-SP and gt solutions work, as well as if GT throws exception
def check_init_sols(exec_list, inputs, orig_otc, return_err_props=False, return_errs=False):

    triv_errs = []
    orig_errs = []
    
    gt_otc   = gen_spec_otc(exec_list, precs_inv[2])
    triv_otc = gen_spec_otc(exec_list, precs_inv[0])

    # skip if trivial result viable
    for ins in inputs:
        shad_result = sim_prog(exec_list, ins, gt_otc) 

        if (shad_result == None):
            #print("\tground truch OTC threw exception")
            if return_err_props:
                if (return_errs):
                    return False, None, None, None, None
                return False, None, None
            return False

        triv_result = sim_prog(exec_list, ins, triv_otc)
        orig_result = sim_prog(exec_list, ins, orig_otc)

        if not(triv_result == None):
            triv_errs.append(relative_error(triv_result, shad_result))
        
        # original solution threw exception
        if (orig_result == None):
            if (return_err_props):
                if (return_errs):
                    return False, None, None, None, None
                return False, None, None
            return False
        else:
            orig_errs.append(relative_error(orig_result, shad_result))            
 
    accept, og_gt_prop = accept_err(orig_errs)
    if not(accept):
        #print("\torig solution didn't work for prop of inputs")
        if (return_err_props):
            if (return_errs):
                return False, None, None, None, None
            return False, None, None
        return False

    accept, triv_gt_prop = accept_err(triv_errs)
    if (accept):

        # FIXME FIXME FIXME accepting trivial for now, until we can distinguish failures
        if (return_err_props):
            if (return_errs):
                return True, triv_gt_prop, og_gt_prop, triv_errs, orig_errs
            return True, triv_gt_prop, og_gt_prop
        return True  
            #return False, triv_gt_prop, og_gt_prop
        #return False   

    if (return_err_props):
        if (return_errs):
            return True, triv_gt_prop, og_gt_prop, triv_errs, orig_errs
        return True, triv_gt_prop, og_gt_prop
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


# **copied form Ophoff on github, pytorch issue #8741
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, th.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, th.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

    
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

    if (USE_GPU):
        gnn.to(get_dev())
        optimizer_to(optimizer,get_dev())

  
    for epoch in range(EPOCHS):
        ep_loss = []
        ep_acc  = []

        ep_prec = []
        ep_rec  = []

        comp_loss = None

        #graphs in each batch combined into single monolithic graph
        for bat_idx in range(BAT_COUNT):            
            if (bat_idx % 10 == 0):
                print("\tbat " + str(bat_idx) + "/" + str(BAT_COUNT))

            optimizer.zero_grad()

            bat_idxs                = shuff_idxs[bat_idx*BAT_SZ : (bat_idx+1)*BAT_SZ] 
            graphs_bat, labels_bat  = batch_graphs_from_idxs(bat_idxs, g_edges, unary_masks, g_idxs, 
                                                             feats, USE_GPU, labels)        

            if (USE_GPU):
                graphs_bat.to(get_dev()) 

            class_weights = th.tensor([1.0 for c in range(CLASSES)])
            if (USE_CL_BAL):
                class_weights = get_class_weights(labels_bat.detach().numpy()) 
            comp_loss = nn.CrossEntropyLoss(weight=class_weights,ignore_index=IGNORE_CLASS, size_average=True) 

            if (USE_GPU):
                class_weights.to(get_dev())
                comp_loss = comp_loss.to(get_dev()) 

            predicts, fwd_topo_ord = gnn(graphs_bat, USE_GPU)
            loss = None

            if (USE_GPU):                                        
                loss     = comp_loss(predicts, labels_bat.to(get_dev()))
                predicts = predicts.to("cpu")
            else:
                loss = comp_loss(predicts, labels_bat)

            prec, rec = prec_recall(predicts, labels_bat)
            ep_prec.append(prec)
            ep_rec.append(rec)
                     
            ep_loss.append(loss.cpu().item())
            ep_acc.append(pred_acc(predicts, th.tensor(labels_bat)))

            loss.backward() 
            optimizer.step()

            if (USE_GPU):
                th.cuda.empty_cache()
                th.cuda.synchronize()

        val_idxs   = shuff_idxs[BAT_COUNT*BAT_SZ : BAT_COUNT*BAT_SZ + VALID_SZ]        
        val_graphs_bat, val_labels = batch_graphs_from_idxs(val_idxs, g_edges, unary_masks,\
                                                                    g_idxs, feats, USE_GPU, labels)
            
        val_predicts, _ = gnn(val_graphs_bat, USE_GPU)

        if (USE_GPU):
            val_graphs_bat = val_graphs_bat.to(get_dev())
        val_loss = None

        #FIXME FIXME
        comp_loss.weights = th.tensor([1.0 for i in range(CLASSES)]) 

        if (USE_GPU):
            val_loss  = comp_loss(val_predicts, val_labels.to(get_dev()))
            val_predicts = val_predicts.to("cpu")
        else:
            val_loss = comp_loss(val_predicts, val_labels)
        val_acc   = pred_acc(val_predicts, val_labels)

        if (epoch == 0):
            print("\nepoch,tr_loss,tr_acc,val_loss,val_acc")                   

        mean_loss = np.mean(ep_loss)
        print("\n" + str(epoch) + "," + str(mean_loss) + "," + str(np.mean(ep_acc)) + \
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


        if (epoch % 2 == 0):
            th.save(gnn.state_dict(), EXP_NAME + "/_ep" + str(epoch) + "_tr" + str("{0:0.2f}".format(mean_loss)) + \
                                                                  "_val"+ str("{0:0.2f}".format(val_loss.item())))  

        #FIXME output val idxs
        if (epoch == 0):
            val_writer = open(EXP_NAME + "/val_idxs", 'w+')    
            for ex_idx in val_idxs:
                val_writer.write(str(ex_idx) + "\n") 
            val_writer.close()           



    return gnn 



######################################################################
# FIXME FIXME FIXME FIXME FIXME FIXME make sure not to tune constants!
######################################################################


#
def mpgnn_test_eval(gnn, g_edges, feats, labels, unary_masks, g_idxs, shuff_idxs, BASE_PATH):
        
    print("\n**********test phase*************")
 
    example_count = len(feats)
    BAT_COUNT = int((example_count * TR_DS_PROP) / BAT_SZ) 
    VALID_SZ  = int(example_count * VAL_DS_PROP)  


    tst_idxs = []
    if not(MOD_PATH == None):
        print("loading tst idxs from path")


        mod_dev = get_dev() if USE_GPU else th.device('cpu')
        state_dict = th.load(MOD_PATH, map_location=mod_dev)

        gnn.to(get_dev())
        gnn.load_hier_state(state_dict)
        gnn.to('cpu')

        idxs_f = open(TST_IDXS_PATH, 'r')
        
        for ex_idx in idxs_f:
            tst_idxs.append(int(ex_idx)) 
        idxs_f.close()

    else:
        print("writing tst idxs")
        tst_idxs = [shuff_idxs[i] for i in range(BAT_COUNT*BAT_SZ + VALID_SZ, len(feats))]

        tst_writer = open(EXP_NAME + "/tst_idxs", 'w+')    
        for ex_idx in tst_idxs:
            tst_writer.write(str(ex_idx) + "\n") 
        tst_writer.close()           




    # difference in prec % after tuning
    prec_freq_deltas = {32:[], 64:[], 80:[]}
    all_count = all_prop_count =  0

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


    sp_err_props   = []
    og_err_props   = []
    tune_err_props = []

    for tst_idx in range(len(tst_idxs)):         

        if (all_count >= MAX_TST_PROGS):
            break 

        if (tst_idx % 25 == 0):#100 == 0):
            print("tst_idx " + str(tst_idx) + ", examps: " + str(all_count))
        
        ex_idx   = tst_idxs[tst_idx] 
        g_idx    = g_idxs[ex_idx]
        ex_feats = feats[ex_idx] # tuples of [opcode, prec]

        ex_graph, ex_label = batch_graphs_from_idxs([ex_idx], g_edges, unary_masks, g_idxs, 
                                                     feats, use_gpu=False, labels=labels)        
        
        predicts, top_order = gnn(ex_graph, False)

        sm = nn.Softmax(dim=-1)
        predicts = sm(th.sigmoid(predicts))

        # print predicts         
        if (tst_idx < 8):
            print("")
            for p_idx in range(len(predicts)):
                if not(ex_feats[p_idx][0] == 0):
                    print(str(predicts.detach().numpy()[p_idx]) + ", true: " + str(ex_label.detach().numpy()[p_idx])) 

        exec_list = [None for i in range(len(ex_feats))]
        otc_orig  = [None for i in range(len(ex_feats))]
        otc_tuned  = [None for i in range(len(ex_feats))]      

        for step in top_order:
            for n in step:       
                rec = int(np.argmax(predicts[n].detach()))

                parents = [int(v) for v in ex_graph.in_edges(n)[0]]

                if (len(parents) < 2):
                    if (len(parents) < 1): 
                        parents.append(None) 
                    parents.append(None) 
 
                if (SINGLE_GRAPH_TARG):
                    exec_list[n] = [int(n), ex_feats[n], parents[0], parents[1]]

                    if not(ex_feats[n] == ops['CONST']):
                        otc_tuned[n] = precs_inv[rec]
                else:
                    exec_list[n] = ([int(n), ex_feats[n][0], parents[0], parents[1]])
                    otc_orig[n]  = precs_inv[ex_feats[n][1]]              

                    if not(ex_feats[n] == ops['CONST']):
                        if (COARSE_TUNE):
                            if (SP_TARGET):
                                otc_tuned[n] = 32 if (rec==1 and predicts.detach().numpy()[n][rec] >= PRED_THRESH) \
                                                  else precs_inv[ex_feats[n][1]]                            
                            else: 
                                otc_tuned[n] = precs_inv[max(0, ex_feats[n][1]-1)] if (rec==1 and predicts.detach().numpy()[n][rec] >= PRED_THRESH)\
                                                                                   else precs_inv[ex_feats[n][1]] 
                        else:
                            otc_tuned[n] = precs_inv[tune_prec(ex_feats[n][1], rec)]
                                          
        #inputs = gen_stratified_inputs(exec_list, input_samp_sz, inputs_mag) 
        inputs = load_inputs(BASE_PATH + "/inputs_" + str(ex_idx) +".csv") 
        inputs = [pad_inputs(exec_list, ins) for ins in inputs]

        viable, sp_prop, og_prop, sp_errs, og_errs = check_init_sols(exec_list, inputs, otc_orig, return_err_props=True, return_errs=True)

        # skip example if either trivial sol works, shadow throws except, orig sol throws except       
        if not(viable):
             continue        

        prec, rec = prec_recall(predicts, ex_label)
        tst_precs.append(prec)
        tst_recs.append(rec)

        ex_errs = []
        gt_otc = gen_spec_otc(exec_list, precs_inv[2])

        for ins in inputs:
            result      = sim_prog(exec_list, ins, otc_tuned)
            shad_result = sim_prog(exec_list, ins, gt_otc) 
            ex_errs.append(relative_error(result, shad_result))
        all_count += 1

        accept, tune_gt_prop = accept_err(ex_errs)
        if (accept):
            all_prop_count += 1

            for pred in predicts:
                tune_counts[np.argmax(pred.detach().numpy())] += 1
            if not(SINGLE_GRAPH_TARG):
                update_freq_delta(prec_freq_deltas, otc_tuned, otc_orig)   

        sp_err_props.append(sp_prop)
        og_err_props.append(og_prop)
        tune_err_props.append(tune_gt_prop)   

    if (all_count > 0):
        print("\n**test prog count: " + str(all_count) + "\n")     
        print("**proportion within thresh, for proportion of inputs " + str(float(all_prop_count) / float(all_count)))
        print("\n**avg sp_err_prop: " + str(np.mean(sp_err_props)) + "**\navg tune_err_prop: " + str(np.mean(tune_err_props)) + "\n**avg OG_err_prop: " + str(np.mean(og_err_props)))


        print("\nfreq deltas after tuning")
        for key in prec_freq_deltas.keys():
            print(str(key) + " " + str(np.mean(prec_freq_deltas[key])))

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


        #FIXME
        #print("\n\nsp_err_props,og_err_props,tune_err_props")
        #for i in range(len(sp_err_props)):
        #    print(str(sp_err_props[i]) + "," + str(og_err_props[i]) +"," + str(tune_err_props[i]))



    #
    #
    # eval on fpb
    #
    #

    print("\n**eval on fpbench\n")


    # jet
    edges, n_ops, unary_masks, consts = jet_app()
    inputs = gen_jet_inputs(consts, input_samp_sz) 

    # create exec list            
    exec_list = []    
    for n in range(len(n_ops)): 

        #NOTE seems like this is a better way to maintain commutativity since edges are kept in the order they're added
        parents = [e[0] for e in edges if e[1] == n] 

        if (len(parents) < 2):                       
            if (len(parents) < 1): 
                parents.append(None) 
            parents.append(None) 
        exec_list.append([int(n), n_ops[n], parents[0], parents[1]])

    # make prediction
    dp_graph = batch_graphs_from_idxs([0], [edges], [unary_masks], [0], 
                                      [ [[op,1] for op in n_ops] ], use_gpu=False)            

    predicts, _ = gnn(dp_graph, False)
    sm = nn.Softmax(dim=-1)
    predicts = sm(th.sigmoid(predicts))

    tuned_otc = [64 for i in range(len(predicts))]

    for p_idx in range(len(predicts)):
       if not(n_ops[p_idx] == ops['CONST']):
           rec = int(np.argmax(predicts[p_idx].detach()))  
           if (predicts.detach().numpy()[p_idx][rec] >= PRED_THRESH and rec==1):
               tuned_otc[p_idx] = 32


    for p in predicts.detach().numpy():
        print(p)

    print("\njet tuned otc: " + str(tuned_otc))
              
    gt_otc = gen_spec_otc(exec_list, precs_inv[2])    
    init_otc = gen_spec_otc(exec_list, precs_inv[1])
    sp_otc = gen_spec_otc(exec_list, precs_inv[0])

    ex_errs = []
    init_errs = []
    sp_errs  = []   

    init_good = True

 
    for ins in inputs:
        result      = sim_prog(exec_list, ins, tuned_otc)
        shad_result = sim_prog(exec_list, ins, gt_otc) 
        init_result = sim_prog(exec_list, ins, init_otc)
        sp_result   = sim_prog(exec_list, ins, sp_otc)

        if (shad_result == None or init_result == None):
            init_good = False

        ex_errs.append(relative_error(result, shad_result))
        init_errs.append(relative_error(init_result, shad_result))
        sp_errs.append(relative_error(sp_result, shad_result))

    thresh_jet = 0.00000001
    accept_init, og_prop = accept_err(init_errs, or_thresh=thresh_jet)
    accept_sp, sp_prop   = accept_err(sp_errs, or_thresh=thresh_jet)

    if not(init_good):
        print("\t****initial sol or GT threw exception!")

    if not(accept_init):
        print("\t**initial sol (all-DP) for jet wasn't accepted!")
    elif (accept_sp):
        print("\t**trivial sol (all-SP) for jet was accepted!")

    accept, prop_gt_thresh = accept_err(ex_errs, or_thresh=thresh_jet)    
    print("**accept model-tuned jet: " + str(accept) + ", prop < thresh: " + str(1.0-prop_gt_thresh) + ", max_err: " + str(np.amax(ex_errs)) + "\n")
    print("\tsp prop > thresh: " + str(sp_prop))
    print("\tog prop > thresh: " + str(og_prop))
    print("\tavg err: " + str(np.mean(ex_errs)))

    print("\n**jet ops and final OTC")
    for op_idx in range(len(tuned_otc)):
        print('op_' + str(op_idx) + ", " + str(n_ops[op_idx]) + ", " + str(tuned_otc[op_idx]))
  
    #print_for_gviz(dp_graph.to_networkx(), exec_list, gt_otc, [True if tuned_otc[op_idx]==32 else False for op_idx in range(len(tuned_otc))])





    #########
    # kepler1
    #########

    edges, n_ops, unary_masks, consts = kep1_app()
    inputs = gen_kep1_inputs(consts, input_samp_sz) 

    # create exec list            
    exec_list = []    
    for n in range(len(n_ops)): 
        parents = [e[0] for e in edges if e[1] == n]
        if (len(parents) < 2):                       
            if (len(parents) < 1): 
                parents.append(None) 
            parents.append(None) 
        exec_list.append([int(n), n_ops[n], parents[0], parents[1]])

    # make prediction
    dp_graph = batch_graphs_from_idxs([0], [edges], [unary_masks], [0], 
                                      [ [[op,1] for op in n_ops] ], use_gpu=False)            

    predicts, _ = gnn(dp_graph, False)
    sm = nn.Softmax(dim=-1)
    predicts = sm(th.sigmoid(predicts))

    tuned_otc = [64 for i in range(len(predicts))]

    for p_idx in range(len(predicts)):
       if not(n_ops[p_idx] == ops['CONST']):
           rec = int(np.argmax(predicts[p_idx].detach()))  

           if (predicts.detach().numpy()[p_idx][rec] >= PRED_THRESH and rec==1):
               tuned_otc[p_idx] = 32

    for p in predicts.detach().numpy():
        print(p)

    print("\nkep1 tuned otc: " + str(tuned_otc))
              
    gt_otc = gen_spec_otc(exec_list, precs_inv[2])    
    init_otc = gen_spec_otc(exec_list, precs_inv[1])
    sp_otc = gen_spec_otc(exec_list, precs_inv[0])

    ex_errs = []
    init_errs = []
    sp_errs  = []   

    init_good = True
 
    for ins in inputs:
        result      = sim_prog(exec_list, ins, tuned_otc)
        shad_result = sim_prog(exec_list, ins, gt_otc) 
        init_result = sim_prog(exec_list, ins, init_otc)
        sp_result   = sim_prog(exec_list, ins, sp_otc)

        if (shad_result == None or init_result == None):
            init_good = False 
            continue

        ex_errs.append(relative_error(result, shad_result))    
        init_errs.append(relative_error(init_result, shad_result))
        sp_errs.append(relative_error(sp_result, shad_result))

    thresh_kep1 = 0.0000001 #0.000000001
    accept_init, og_prop = accept_err(init_errs, or_thresh=thresh_kep1)
    accept_sp, sp_prop   = accept_err(sp_errs, or_thresh=thresh_kep1)

    if not(init_good):
        print("\t****initial sol or GT threw exception!")

    if not(accept_init):
        print("\t**initial sol (all-DP) for kep1 wasn't accepted!")
    elif (accept_sp):
        print("\t**trivial sol (all-SP) for kep1 was accepted with prop " + str(1.0-sp_prop) + " < thresh and max_err: " + str(np.amax(sp_errs)))

    accept, prop_gt_thresh = accept_err(ex_errs, or_thresh=thresh_kep1)    
    print("**accept model-tuned kep1: " + str(accept) + " prop<thresh: " + str(1.0-prop_gt_thresh) + ", max_err: " + str(np.amax(ex_errs)) + "\n")
    print("\tsp prop > thresh: " + str(sp_prop))
    print("\tog prop > thresh: " + str(og_prop))

    print("\tavg err: " + str(np.mean(ex_errs)))

    print("\n**kep1 ops and final OTC")
    for op_idx in range(len(tuned_otc)):
        print('op_' + str(op_idx) + ", " + str(n_ops[op_idx]) + ", " + str(tuned_otc[op_idx]))
    #print_for_gviz(dp_graph.to_networkx(), exec_list, gt_otc, [True if tuned_otc[op_idx]==32 else False for op_idx in range(len(tuned_otc))])



    #########
    # kepler2
    #########

    edges, n_ops, unary_masks, consts = kep2_app()
    inputs = gen_kep2_inputs(consts, input_samp_sz) 

    # create exec list            
    exec_list = []    
    for n in range(len(n_ops)): 
        parents = [e[0] for e in edges if e[1] == n]
        if (len(parents) < 2):                       
            if (len(parents) < 1): 
                parents.append(None) 
            parents.append(None) 
        exec_list.append([int(n), n_ops[n], parents[0], parents[1]])

    # make prediction
    dp_graph = batch_graphs_from_idxs([0], [edges], [unary_masks], [0], 
                                      [ [[op,1] for op in n_ops] ], use_gpu=False)            

    predicts, _ = gnn(dp_graph, False)
    sm = nn.Softmax(dim=-1)
    predicts = sm(th.sigmoid(predicts))

    tuned_otc = [64 for i in range(len(predicts))]

    for p_idx in range(len(predicts)):
       if not(n_ops[p_idx] == ops['CONST']):
           rec = int(np.argmax(predicts[p_idx].detach()))  

           if (predicts.detach().numpy()[p_idx][rec] >= PRED_THRESH and rec==1):
               tuned_otc[p_idx] = 32

    for p in predicts.detach().numpy():
        print(p)

    print("\nkep2 tuned otc: " + str(tuned_otc))
              
    gt_otc = gen_spec_otc(exec_list, precs_inv[2])    
    init_otc = gen_spec_otc(exec_list, precs_inv[1])
    sp_otc = gen_spec_otc(exec_list, precs_inv[0])

    ex_errs = []
    init_errs = []
    sp_errs  = []   

    init_good = True
 
    for ins in inputs:
        result      = sim_prog(exec_list, ins, tuned_otc)
        shad_result = sim_prog(exec_list, ins, gt_otc) 
        init_result = sim_prog(exec_list, ins, init_otc)
        sp_result   = sim_prog(exec_list, ins, sp_otc)

        if (shad_result == None or init_result == None):
            init_good = False 
            continue

        ex_errs.append(relative_error(result, shad_result))    
        init_errs.append(relative_error(init_result, shad_result))
        sp_errs.append(relative_error(sp_result, shad_result))

    thresh_kep2 = 0.000000001 #0.000000001 

    accept_init, prop_og = accept_err(init_errs, or_thresh=thresh_kep2 )
    accept_sp, prop_sp   = accept_err(sp_errs,  or_thresh=thresh_kep2)

    if not(init_good):
        print("\t****initial sol or GT threw exception!")

    if not(accept_init):
        print("\t**initial sol (all-DP) for kep2 wasn't accepted!")
    elif (accept_sp):
        print("\t**trivial sol (all-SP) for kep2 was accepted with prop " + str(prop_sp) + " > thresh and max_err: " + str(np.amax(sp_errs)))

    accept, prop_gt_thresh = accept_err(ex_errs, or_thresh=thresh_kep2)    
    print("**accept model-tuned kep2: " + str(accept) + " prop<thresh: " + str(1.0-prop_gt_thresh) + ", max_err: " + str(np.amax(ex_errs)) + "\n")
    print("\tsp prop < thresh: " + str(1.0-prop_sp))
    print("\tog prop < thresh: " + str(1.0-prop_og))

    print("\n**k2p ops and final OTC")
    for op_idx in range(len(tuned_otc)):
        print('op_' + str(op_idx) + ", " + str(n_ops[op_idx]) + ", " + str(tuned_otc[op_idx]))
    #print_for_gviz(dp_graph.to_networkx(), exec_list, gt_otc, [True if tuned_otc[op_idx]==32 else False for op_idx in range(len(tuned_otc))])


    print("\tavg err: " + str(np.mean(ex_errs)))


    #
    # AL
    #
    edges, n_ops, unary_masks, consts = al_bbfun()

    dp_graph = batch_graphs_from_idxs([0], [edges], [unary_masks], [0], 
                                      [ [[op,1] for op in n_ops] ], use_gpu=False)            

    predicts, _ = gnn(dp_graph, False)
    sm = nn.Softmax(dim=-1)
    predicts = sm(th.sigmoid(predicts))

    for p in predicts.detach().numpy():
        print(p)

    tuned_otc = [64 for i in range(len(predicts))]

    for p_idx in range(len(predicts)):
       if not(n_ops[p_idx] == ops['CONST']):
           rec = int(np.argmax(predicts[p_idx].detach()))  

           if (predicts.detach().numpy()[p_idx][rec] >= PRED_THRESH and rec==1):
               tuned_otc[p_idx] = 32

    print("\nal tuned otc")
    for op_idx in range(len(tuned_otc)):
        print('op_' + str(op_idx) + ", " + str(n_ops[op_idx]) + ", " + str(tuned_otc[op_idx]))



    #
    # simpsons
    #
    edges, n_ops, unary_masks, consts = simps_bb()

    dp_graph = batch_graphs_from_idxs([0], [edges], [unary_masks], [0], 
                                      [ [[op,1] for op in n_ops] ], use_gpu=False)            

    predicts, _ = gnn(dp_graph, False)
    sm = nn.Softmax(dim=-1)
    predicts = sm(th.sigmoid(predicts))

    for p in predicts.detach().numpy():
        print(p)

    tuned_otc = [64 for i in range(len(predicts))]

    for p_idx in range(len(predicts)):
       if not(n_ops[p_idx] == ops['CONST']):
           rec = int(np.argmax(predicts[p_idx].detach()))  

           if (predicts.detach().numpy()[p_idx][rec] >= PRED_THRESH and rec==1):
               tuned_otc[p_idx] = 32

    print("\nsimpson's tuned otc")
    for op_idx in range(len(tuned_otc)):
        print('op_' + str(op_idx) + ", " + str(n_ops[op_idx]) + ", " + str(tuned_otc[op_idx]))


              

