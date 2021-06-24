from dgl import DGLGraph, batch
import torch as th

import csv
import sys

import numpy as np
from collections import Counter

from model.bignn import bignn
from common.metrics import prec_recall
from common.pfpo_utils import get_dev
from common.graph_helper import batch_graphs_from_idxs 
import common.otc as otc

#
def get_class_weights(labels, classes, ignore_class):
    counts  = Counter(labels)
    weights = []
    total   = 0

    for k in counts.keys():
        if not (k == ignore_class):
            total += counts[k]        

    for i in range(classes):
        if (i not in counts.keys()):
            weights.append(1.0)
        else:
            weights.append(1.0 / counts[i])

    norm_const = np.sum(weights)
    weights    = [float(w/norm_const) for w in weights]
    return th.tensor(weights)


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
def train_bignn(gnn, ds, config):
    example_count = len(ds['feats'])
    bat_sz        = config['training']['batch_sz']
    bat_count     = int((example_count * config['training']['train_ds_proportion']) / bat_sz) 
    valid_sz      = int(example_count * config['training']['validation_ds_proportion'])  

    feat_dim  = otc.OP_ENC_DIM
    optimizer = th.optim.Adagrad(gnn.parameters()) 

    if (config['use_gpu']):
        gnn.to(get_dev())
        optimizer_to(optimizer, get_dev())
  
    for epoch in range(config['training']['epochs']):
        ep_loss = []
        ep_prec = []
        ep_rec  = []

        comp_loss = None

        # graphs in each batch combined into single graph 
        for bat_idx in range(bat_count):            
            optimizer.zero_grad()

            # gather batch
            bat_idxs                = ds['shuff_idxs'][bat_idx*bat_sz : (bat_idx+1)*bat_sz] 
            graphs_bat, labels_bat  = batch_graphs_from_idxs(bat_idxs, ds['g_edges'], ds['unary_masks'], ds['g_idxs'], 
                                                             ds['feats'], config['use_gpu'], ds['labels'])        
            if (config['use_gpu']):
                graphs_bat.to(get_dev()) 

            # instantiate loss functor 
            comp_loss = None
            if (config['training']['use_class_bal']):
                class_weights = get_class_weights(labels_bat.detach().numpy(), config['model']['classes'], config['model']['ignore_class']) 
                comp_loss = th.nn.CrossEntropyLoss(weight=class_weights,ignore_index=config['model']['ignore_class'], size_average=True) 
            else:
                comp_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS, size_average=True) 
            if (config['use_gpu']):
                class_weights.to(get_dev())
                comp_loss = comp_loss.to(get_dev()) 

            # compute loss, precision, recall
            predicts, _ = gnn(graphs_bat, config['use_gpu'])
            loss        = None

            if (config['use_gpu']):                                        
                loss     = comp_loss(predicts, labels_bat.to(get_dev()))
                predicts = predicts.to("cpu")
            else:
                loss = comp_loss(predicts, labels_bat)

            prec, rec = prec_recall(predicts, labels_bat, config['model']['classes'], \
                                    config['model']['ignore_class'], config['model']['prediction_thresh'])
            ep_prec.append(prec)
            ep_rec.append(rec)                     
            ep_loss.append(loss.cpu().item())

            loss.backward() 
            optimizer.step()

            if (config['use_gpu']):
                th.cuda.empty_cache()
                th.cuda.synchronize()

        # eval on validation set
        val_idxs                   = ds['shuff_idxs'][bat_count*bat_sz : bat_count*bat_sz + valid_sz]        
        val_graphs_bat, val_labels = batch_graphs_from_idxs(val_idxs, ds['g_edges'], ds['unary_masks'],\
                                                            ds['g_idxs'], ds['feats'], config['use_gpu'], ds['labels'])            
        val_predicts, _ = gnn(val_graphs_bat, config['use_gpu'])

        val_loss = None
        if (config['use_gpu']):
            val_graphs_bat = val_graphs_bat.to(get_dev())
            val_loss       = comp_loss(val_predicts, val_labels.to(get_dev()))
            val_predicts   = val_predicts.to("cpu")
        else:
            val_loss = comp_loss(val_predicts, val_labels)

        # avg prec-rec
        avg_prec = {}
        avg_rec  = {}
        for i in range(bat_count):
            for c in range(config['model']['classes']):            
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

        for c in range(config['model']['classes']):            
            print("\t" + str(c) + ": " + str(np.mean(avg_prec[c])) + ", " + str(np.mean(avg_rec[c])))

        # save model
        if (epoch % 2 == 0):
            th.save(gnn.state_dict(), config['experiment_path'] + "/_ep" + str(epoch) + "_tr" + str("{0:0.2f}".format(mean_loss)) + \
                                      "_val"+ str("{0:0.2f}".format(val_loss.item())))  
        # output validation idxs
        if (epoch == 0):
            val_writer = open(config['experiment_path'] + "/val_idxs", 'w+')    
            for ex_idx in val_idxs:
                val_writer.write(str(ex_idx) + "\n") 
            val_writer.close()           
    return gnn 



