import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from params import *

class fptc_gnn(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(fptc_gnn, self).__init__() 
        self.hidden_dim = hidden_dim
        self.embed_op   = nn.Linear(feat_dim, hidden_dim) 

        #FIXME input dimensions for embed_node and mp_binary don't work with feedforward!!!
        self.embed_node = [nn.Linear(2*hidden_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for l in range(LAYERS-1)]
        self.mp_unary   = [nn.Linear(hidden_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for l in range(LAYERS-1)]
        self.mp_binary  = [nn.Linear(2*hidden_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for l in range(LAYERS-1)]     
        self.predict    = nn.Linear(hidden_dim, CLASSES)     

    #
    def apply_node_func(self, nodes):                              
        node_count  = len(nodes.nodes())
        node_embeds = th.tanh(self.embed_op(nodes.data['node_feats']))

        if ('msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((node_embeds, nodes.data['msgs_redux'].reshape((node_count, self.hidden_dim))), -1)

            node_embeds  = th.tanh(self.embed_node[0](concat_feats))
            #node_embeds  = th.relu(self.embed_node[0](concat_feats))

            for l in range(1, LAYERS):
                node_embeds = th.tanh(self.embed_node[l](node_embeds))
                #node_embeds = th.relu(self.embed_node[l](node_embeds))


        return {'node_embeds': node_embeds}

    #
    def message_func(self, edges):                              
        return {'msgs': edges.src['node_embeds']}
 
    #
    def reduce_func(self, nodes):       
        is_unary   = nodes.data['is_unary_op'][0]
        node_count = len(nodes)
        msgs_redux = None

        if (is_unary==True):
            msgs_redux = th.tanh(self.mp_unary[0](nodes.mailbox['msgs']))                  
            for l in range(1, LAYERS):
                #msgs_redux = th.tanh(self.mp_unary[l](msgs_redux))
                msgs_redux = th.relu(self.mp_unary[l](msgs_redux))

        else:
            msgs_concat = nodes.mailbox['msgs'].reshape((node_count, 1, 2*self.hidden_dim))

            msgs_redux  = th.tanh(self.mp_binary[0](msgs_concat))
            #msgs_redux  = th.relu(self.mp_binary[0](msgs_concat))


            for l in range(1, LAYERS):
                msgs_redux = th.tanh(self.mp_unary[l](msgs_redux))
                #msgs_redux = th.relu(self.mp_unary[l](msgs_redux))



        return {'msgs_redux': msgs_redux}

    #
    def forward(self, graph):
        #graph.register_apply_node_func(self.apply_node_func)    
        #graph.register_message_func(self.message_func)
        #graph.register_reduce_func(self.reduce_func)

        topo_order = dgl.topological_nodes_generator(graph)
        graph.prop_nodes(topo_order, self.message_func, self.reduce_func, self.apply_node_func) 
        act     = nn.Softmax(dim=-1)
        predict = act(th.sigmoid(self.predict(graph.ndata['node_embeds'])))
                  
        return predict, topo_order


