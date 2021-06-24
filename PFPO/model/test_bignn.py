import numpy as np
from common.graph_helper import batch_graphs_from_idxs  
from common.otc import * 
from common.metrics import *
from sim.input_helper import load_inputs, pad_inputs 
from sim.prog_sim import sim_prog
import torch as th


#
def test_bignn(ds, inputs_dir, gnn, config):         
    example_count = len(ds['feats'])
    bat_sz        = config['training']['batch_sz']
    tr_bat_count  = int((example_count * config['training']['train_ds_proportion']) / bat_sz) 
    valid_sz      = int(example_count * config['training']['validation_ds_proportion'])  

    # load test_idxs if loading pretrained from disk
    tst_idxs = []
    if not(config['pretrained']['model_path'] == None):      
        tst_idxs_path = config['pretrained']['tst_idxs_path']
        if not (tst_idxs_path is None):
            with open(tst_idxs_path, 'r') as idxs_f:        
                for ex_idx in idxs_f:
                    tst_idxs.append(int(ex_idx)) 
                idxs_f.close()
        else:
            tst_idxs = list(np.arange(len(ds['feats'])))
    else:
        tst_idxs   = [ds['shuff_idxs'][i] for i in range(tr_bat_count*BAT_SZ + valid_sz, len(ds['feats']))]
        tst_writer = open(config['experiment_path'] + "/tst_idxs", 'w+')    

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
    for tst_idx in range(config['testing']['max_test_count']):         
        if (tst_idx % 25 == 0):
            print("tst_idx " + str(tst_idx))
        
        ex_idx    = tst_idxs[tst_idx] 
        ex_feats  = ds['feats'][ex_idx]
        num_nodes = len(ex_feats)

        ex_graph, ex_label = batch_graphs_from_idxs([ex_idx], ds['g_edges'], ds['unary_masks'], ds['g_idxs'], 
                                                     ds['feats'], use_gpu=config['use_gpu'], labels=ds['labels'])                
        predicts, top_order = gnn(ex_graph, False)

        sm       = th.nn.Softmax(dim=-1)
        predicts = sm(th.sigmoid(predicts))

        otc_orig  = [None for i in range(num_nodes)]
        otc_tuned = [None for i in range(num_nodes)]      
        exec_list = ds['exec_lists'][ex_idx] 

        # construct OTCs for simulator
        for step in top_order:
            for n in step:       
                otc_orig[n] = PRECS_INV[ex_feats[n][1]]              
                rec         = int(np.argmax(predicts[n].detach()))

                #construct tuned otc
                if not(ex_feats[n] == OPS['CONST']):
                    otc_tuned[n] = 32 if (rec==1 and predicts.detach().numpy()[n][rec] >= config['model']['prediction_thresh'] and \
                                          not(ex_label.detach().numpy()[n]==config['model']['ignore_class'])) \
                                      else 64                            
                else:
                    otc_tuned[n] = otc.precs_inv[ex_feats[n][1]]

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
        inputs = load_inputs(inputs_dir + "/inputs_" + str(ex_idx) +".csv") 
        inputs = [pad_inputs(exec_list, ins) for ins in inputs]

        # simulate test programs with tuning 
        prec, rec = prec_recall(predicts, ex_label, config['model']['classes'],\
                                config['model']['ignore_class'], config['model']['prediction_thresh'])
        tst_precs.append(prec)
        tst_recs.append(rec)

        ex_errs = []
        gt_otc  = gen_spec_otc(exec_list, PRECS_INV[2])

        for ins in inputs:
            result      = sim_prog(exec_list, write_result, ins, otc_tuned)
            shad_result = sim_prog(exec_list, write_result, ins, gt_otc) 
            ex_errs.append(relative_error(result, shad_result))

        accept, prop_exceeding_thresh = accept_err(ex_errs, config['testing']['error_thresh'],\
                                                    config['testing']['accept_proportion'])
        if (accept):
            accepted_count += 1
            for pred in predicts:
                tune_counts[np.argmax(pred.detach().numpy())] += 1
            update_freq_delta(prec_freq_deltas, otc_tuned, otc_orig)   
        tune_err_props.append(prop_exceeding_thresh)   

    print("**proportion of inputs accepted " + str(float(accepted_count) / float(config['testing']['max_test_count'])))
    print("**avg tune_err_prop: " + str(np.mean(tune_err_props)))
    print("\n**freq deltas after tuning")
    for key in prec_freq_deltas.keys():
        print(str(key) + " " + str(np.mean(prec_freq_deltas[key])))

    print("\n**avg prec-recall across tests graphs")
    avg_prec = {}
    avg_rec  = {}
    classes = config['model']['classes']
    for c in range(classes):
        avg_prec[c] = []
        avg_rec[c]  = [] 
    for i in range(len(tst_precs)):
        for c in range(classes):            
            avg_prec[c].append(tst_precs[i][c])
            avg_rec[c].append(tst_recs[i][c]) 
    for c in range(classes):            
        print("\t" + str(c) + ": " + str(np.mean(avg_prec[c])) + ", " + str(np.mean(avg_rec[c])))
