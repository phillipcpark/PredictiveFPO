from dgl import DGLGraph, batch
from config.params import *
from common.session import get_dev

from torch import tensor

# 
def create_dgl_graph(edges, feats, is_unary_op, use_gpu):
    if (use_gpu):
        edge_src    = tensor([e[0] for e in edges]).to(get_dev())
        edge_target = tensor([e[1] for e in edges]).to(get_dev())    
    else:
        edge_src    = tensor([e[0] for e in edges])
        edge_target = tensor([e[1] for e in edges])    
   
    graph = DGLGraph((edge_src, edge_target))

    if (use_gpu):
        graph.ndata['node_feats']  = tensor(feats).to(get_dev())
        graph.ndata['is_unary_op'] = tensor(is_unary_op).to(get_dev())
    else:
        graph.ndata['node_feats']  = tensor(feats)
        graph.ndata['is_unary_op'] = tensor(is_unary_op) 
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
    labels_bat = tensor(labels_bat) if not(labels == None) else None

    if not(labels == None):
        return graphs_bat, labels_bat
    return graphs_bat


#
def rev_graph_batch(graphs):
    rev_bat = batch([g.reverse(share_ndata=True, share_edata=True) for g in graphs])
    return rev_bat
