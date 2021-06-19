from dgl import DGLGraph, batch
import torch as th

import csv
import sys

from params import *
from bignn import bignn

from eval_metrics import *
from otc import *

from prog_sim import *



# 
def create_dgl_graph(edges, feats, is_unary_op, use_gpu):
    if (use_gpu):
        edge_src    = th.tensor([e[0] for e in edges]).to(get_dev())
        edge_target = th.tensor([e[1] for e in edges]).to(get_dev())    
    else:
        edge_src    = th.tensor([e[0] for e in edges])
        edge_target = th.tensor([e[1] for e in edges])    
   
    graph = DGLGraph((edge_src, edge_target))

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
    labels_bat  = []
 
    for ex_idx in idxs:         
        g_idx    = g_idxs[ex_idx]
        edges    = g_edges[g_idx] 

        ex_feats = [OP_ENC[feat[0]][feat[1]] for feat in feats[ex_idx]]
        ex_graph = create_dgl_graph(g_edges[g_idx],\
                                    ex_feats,\
                                    unary_masks[g_idx],use_gpu)         
        if (use_gpu):
            ex_graph = ex_graph.to(get_dev())                          
        graphs_list.append(ex_graph)

        if not(labels == None):
            labels_bat += labels[ex_idx]                     
    graphs_bat = batch(graphs_list) 
    labels_bat = th.tensor(labels_bat) if not(labels == None) else None

    if not(labels == None):
        return graphs_bat, labels_bat
    return graphs_bat


#
def rev_graph_batch(graphs):
    rev_bat = batch([g.reverse(share_ndata=True, share_edata=True) for g in graphs])
    return rev_bat
    

#
def get_class_weights(labels):
    counts  = Counter(labels)
    weights = []
    total   = 0

    for k in counts.keys():
        if not (k == IGNORE_CLASS):
            total += counts[k]        

    for i in range(CLASSES):
        if (i not in counts.keys()):
            weights.append(1.0)
        else:
            weights.append(1.0 / counts[i])

    norm_const = np.sum(weights)
    weights    = [float(w/norm_const) for w in weights]
    return th.tensor(weights)

#
def get_dev():
    if not(th.cuda.is_available()):
        raise RuntimeError('CUDA unavailable')
    dev = "cuda:0"
    return th.device(dev)




#
def pad_inputs(exec_list, ins):
    cidx = 0
    flush_ins = []
    for line in exec_list:
        if is_const(line[1]):
            flush_ins.append(ins[cidx])
            cidx += 1
        else:
            flush_ins.append(None)
    return flush_ins

#
def load_inputs(path):
    f_hand = open(path, 'r')  
    reader = csv.reader(f_hand, delimiter=',')

    ins = []

    mp.prec = 65
    for row in reader:        
        ins.append([mp.mpf(val) for val in row])
    f_hand.close()

    return ins



# copied form Ophoff on github, pytorch issue #8741
def optimizer_to(optim, device):
    for param in optim.state.values():
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
def train_bignn(gnn, ds):
    example_count = len(ds['feats'])
    bat_count     = int((example_count * TR_DS_PROP) / BAT_SZ) 
    valid_sz      = int(example_count * VAL_DS_PROP)  

    feat_dim  = OP_ENC_DIM
    optimizer = th.optim.Adagrad(gnn.parameters()) 

    if (USE_GPU):
        gnn.to(get_dev())
        optimizer_to(optimizer, get_dev())
  
    for epoch in range(EPOCHS):
        ep_loss = []
        ep_prec = []
        ep_rec  = []

        comp_loss = None

        # graphs in each batch combined into single graph 
        for bat_idx in range(bat_count):            
            optimizer.zero_grad()

            # gather batch
            bat_idxs                = ds['shuff_idxs'][bat_idx*BAT_SZ : (bat_idx+1)*BAT_SZ] 
            graphs_bat, labels_bat  = batch_graphs_from_idxs(bat_idxs, ds['g_edges'], ds['unary_masks'], ds['g_idxs'], 
                                                             ds['feats'], USE_GPU, ds['labels'])        
            if (USE_GPU):
                graphs_bat.to(get_dev()) 

            # instantiate loss functor 
            comp_loss = None
            if (USE_CL_BAL):
                class_weights = get_class_weights(labels_bat.detach().numpy()) 
                comp_loss = th.nn.CrossEntropyLoss(weight=class_weights,ignore_index=IGNORE_CLASS, size_average=True) 
            else:
                comp_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS, size_average=True) 
            if (USE_GPU):
                class_weights.to(get_dev())
                comp_loss = comp_loss.to(get_dev()) 

            # compute loss, precision, recall
            predicts, _ = gnn(graphs_bat, USE_GPU)
            loss        = None

            if (USE_GPU):                                        
                loss     = comp_loss(predicts, labels_bat.to(get_dev()))
                predicts = predicts.to("cpu")
            else:
                loss = comp_loss(predicts, labels_bat)

            prec, rec = prec_recall(predicts, labels_bat)
            ep_prec.append(prec)
            ep_rec.append(rec)                     
            ep_loss.append(loss.cpu().item())

            loss.backward() 
            optimizer.step()

            if (USE_GPU):
                th.cuda.empty_cache()
                th.cuda.synchronize()

        # eval on validation set
        val_idxs                   = ds['shuff_idxs'][bat_count*BAT_SZ : bat_count*BAT_SZ + valid_sz]        
        val_graphs_bat, val_labels = batch_graphs_from_idxs(val_idxs, ds['g_edges'], ds['unary_masks'],\
                                                            ds['g_idxs'], ds['feats'], USE_GPU, ds['labels'])            
        val_predicts, _ = gnn(val_graphs_bat, USE_GPU)

        if (USE_GPU):
            val_graphs_bat = val_graphs_bat.to(get_dev())
        val_loss = None

        if (USE_GPU):
            val_loss     = comp_loss(val_predicts, val_labels.to(get_dev()))
            val_predicts = val_predicts.to("cpu")
        else:
            val_loss = comp_loss(val_predicts, val_labels)

        # avg prec-rec
        avg_prec = {}
        avg_rec  = {}
        for i in range(bat_count):
            for c in range(CLASSES):            
                try:
                    avg_prec[c].append(ep_prec[i][c])
                except:
                    avg_prec[c] = [ep_prec[i][c]]
                try:     
                    avg_rec[c].append(ep_rec[i][c]) 
                except:
                    avg_rec[c] = [ep_rec[i][c]] 

        mean_loss = np.mean(ep_loss)
        print('loss: ' + str(mean_loss))

        for c in range(CLASSES):            
            print("\t" + str(c) + ": " + str(np.mean(avg_prec[c])) + ", " + str(np.mean(avg_rec[c])))

        # save model
        if (epoch % 2 == 0):
            th.save(gnn.state_dict(), EXP_NAME + "/_ep" + str(epoch) + "_tr" + str("{0:0.2f}".format(mean_loss)) + \
                                                 "_val"+ str("{0:0.2f}".format(val_loss.item())))  
        # output validation idxs
        if (epoch == 0):
            val_writer = open(EXP_NAME + "/val_idxs", 'w+')    
            for ex_idx in val_idxs:
                val_writer.write(str(ex_idx) + "\n") 
            val_writer.close()           
    return gnn 


#
def test_bignn(ds, base_path, gnn):         
    example_count = len(ds['feats'])
    tr_bat_count  = int((example_count * TR_DS_PROP) / BAT_SZ) 
    valid_sz      = int(example_count * VAL_DS_PROP)  

    # load test_idxs if loading pretrained from disk
    tst_idxs = []
    if not(MOD_PATH == None):      
        if not (TST_IDXS_PATH is None):
            with open(TST_IDXS_PATH, 'r') as idxs_f:        
                for ex_idx in idxs_f:
                    tst_idxs.append(int(ex_idx)) 
                idxs_f.close()
        else:
            tst_idxs = list(np.arange(len(ds['feats'])))
    else:
        tst_idxs   = [ds['shuff_idxs'][i] for i in range(tr_bat_count*BAT_SZ + valid_sz, len(ds['feats']))]
        tst_writer = open(EXP_NAME + "/tst_idxs", 'w+')    

        for ex_idx in tst_idxs:
            tst_writer.write(str(ex_idx) + "\n") 
        tst_writer.close()           

    # eval metrics 
    prec_freq_deltas = {32:[], 64:[], 80:[]}
    accepted_count   =  0
    tune_counts      = {0:0, 1:0}
    
    tst_precs = []
    tst_recs  = []

    sp_err_props   = []
    og_err_props   = []
    tune_err_props = []

    # infer on single test example at a time, since running simulator for each result
    for tst_idx in range(MAX_TST_PROGS):         
        if (tst_idx % 25 == 0):
            print("tst_idx " + str(tst_idx))
        
        ex_idx    = tst_idxs[tst_idx] 
        ex_feats  = ds['feats'][ex_idx]
        num_nodes = len(ex_feats)

        ex_graph, ex_label = batch_graphs_from_idxs([ex_idx], ds['g_edges'], ds['unary_masks'], ds['g_idxs'], 
                                                     ds['feats'], use_gpu=False, labels=ds['labels'])                
        predicts, top_order = gnn(ex_graph, False)

        sm       = th.nn.Softmax(dim=-1)
        predicts = sm(th.sigmoid(predicts))

        otc_orig  = [None for i in range(num_nodes)]
        otc_tuned = [None for i in range(num_nodes)]      
        exec_list = ds['exec_lists'][ex_idx] 

        # construct OTCs for simulator
        for step in top_order:
            for n in step:       
                otc_orig[n] = precs_inv[ex_feats[n][1]]              
                rec         = int(np.argmax(predicts[n].detach()))

                #construct tuned otc
                if not(ex_feats[n] == ops['CONST']):
                    otc_tuned[n] = 32 if (rec==1 and predicts.detach().numpy()[n][rec] >= PRED_THRESH and not(ex_label.detach().numpy()[n]==IGNORE_CLASS)) \
                                      else 64                            
                else:
                    otc_tuned[n] = precs_inv[ex_feats[n][1]]

        # write_result mask is used to denote intermediate variables, since 
        # their precision must be handled differently than those in registers        
        parent_counts = [0 for i in range(len(exec_list))]
        for e in exec_list:
            if not(e[2] == None):
                parent_counts[e[2]] += 1
            if not(e[3] == None):
                parent_counts[e[3]] += 1
        write_result = []
        for i in range(len(exec_list)):
            if (is_func(exec_list[i][1]) or (parent_counts[i] > 1 and not(is_const(exec_list[i][1])))):
                write_result.append(True)
            else:
                write_result.append(False) 

        # load same input sets used for producing labels                                           
        inputs = load_inputs(base_path + "/inputs_" + str(ex_idx) +".csv") 
        inputs = [pad_inputs(exec_list, ins) for ins in inputs]

        # simulate test programs with tuning 
        prec, rec = prec_recall(predicts, ex_label)
        tst_precs.append(prec)
        tst_recs.append(rec)

        ex_errs = []
        gt_otc  = gen_spec_otc(exec_list, precs_inv[2])

        for ins in inputs:
            result      = sim_prog(exec_list, write_result, ins, otc_tuned)
            shad_result = sim_prog(exec_list, write_result, ins, gt_otc) 
            ex_errs.append(relative_error(result, shad_result))

        accept, prop_exceeding_thresh = accept_err(ex_errs)
        if (accept):
            accepted_count += 1
            for pred in predicts:
                tune_counts[np.argmax(pred.detach().numpy())] += 1
            update_freq_delta(prec_freq_deltas, otc_tuned, otc_orig)   
        tune_err_props.append(prop_exceeding_thresh)   

    print("**proportion of inputs accepted " + str(float(accepted_count) / float(MAX_TST_PROGS)))
    print("**avg tune_err_prop: " + str(np.mean(tune_err_props)))
    print("\n**freq deltas after tuning")
    for key in prec_freq_deltas.keys():
        print(str(key) + " " + str(np.mean(prec_freq_deltas[key])))

    print("\n**avg prec-recall across tests graphs")
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





