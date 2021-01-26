import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from params import *
import sys

#
# 
#
class resnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None):
        super(resnet, self).__init__()
        self.lay1 = nn.Linear(in_dim, hidden_dim)
        self.lay2 = nn.Linear(hidden_dim, hidden_dim)
        self.lay3 = nn.Linear(hidden_dim, hidden_dim) if (out_dim==None) else nn.Linear(hidden_dim, out_dim)

        # FIXME relu?
        self.act = nn.Tanh()

    def forward(self, block_in):
        h_acts1 = self.act(self.lay1(block_in))
        h_acts2 = self.act(self.lay2(h_acts1))
        block_out = self.lay3(th.add(h_acts2, block_in)) 
        return block_out 

#
#
#
class fwd_mpgnn(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(fwd_mpgnn, self).__init__() 
        self.hidden_dim = hidden_dim
        self.act = nn.Tanh()

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        self.comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.comb_embed2 = resnet(hidden_dim, hidden_dim)

        self.node_embeds = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.node_embeds2 = resnet(hidden_dim, hidden_dim)

        self.mp_unary = resnet(hidden_dim, hidden_dim)
        self.mp_unary1 = resnet(hidden_dim, hidden_dim)
        self.mp_unary2 = resnet(hidden_dim, hidden_dim)

        self.mp_binary  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.mp_binary1 = resnet(hidden_dim, hidden_dim)
        self.mp_binary2 = resnet(hidden_dim, hidden_dim)

    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = self.act(self.embed_op(nodes.data['node_feats']))

        # subsequent steps of MP will entire this branch
        if ('fwd_node_embeds' in nodes.data.keys()):
            embeds = self.act(self.comb_embed(th.cat((embeds, nodes.data['fwd_node_embeds']), axis=-1))) 
            embeds = self.act(self.comb_embed1(embeds)) 
            embeds = self.act(self.comb_embed2(embeds)) 

        # constant nodes will not enter this branch
        if ('fwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['fwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)
            embeds       = self.act(self.node_embeds(concat_feats))
            embeds       = self.act(self.node_embeds1(embeds))  
            embeds       = self.act(self.node_embeds2(embeds))  
        return {'fwd_node_embeds': embeds}

    #
    def msg_func(self, edges):                              
        return {'fwd_msgs': edges.src['fwd_node_embeds']}
 
    #
    def redux_func(self, nodes):       
        is_unary   = nodes.data['is_unary_op'][0]

        node_count = len(nodes)
        msgs_redux = None
      
        if (is_unary==True):
            msgs_redux = self.act(self.mp_unary(nodes.mailbox['fwd_msgs']))
            msgs_redux  = self.act(self.mp_unary1(msgs_redux))
            msgs_redux  = self.act(self.mp_unary2(msgs_redux))
        else:
            msgs_concat = nodes.mailbox['fwd_msgs'].reshape((node_count, 1, 2*self.hidden_dim))
            msgs_redux  = self.act(self.mp_binary(msgs_concat)) 
            msgs_redux  = self.act(self.mp_binary1(msgs_redux))
            msgs_redux  = self.act(self.mp_binary2(msgs_redux))
        return {'fwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph, fwd_topo_order = None):
        if (fwd_topo_order == None):   
            fwd_topo_order = dgl.topological_nodes_generator(graph)  
        graph.prop_nodes(fwd_topo_order, self.msg_func, self.redux_func, self.apply_func) 

        return graph.ndata['fwd_node_embeds'], fwd_topo_order

#
#
#
class bwd_mpgnn(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(bwd_mpgnn, self).__init__() 

        self.hidden_dim = hidden_dim
        self.act = nn.Tanh()

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        self.mp_bwd  = resnet(hidden_dim, hidden_dim)
        self.mp_bwd1 = resnet(hidden_dim, hidden_dim)
        self.mp_bwd2 = resnet(hidden_dim, hidden_dim)

        self.node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.node_embeds2 = resnet(hidden_dim, hidden_dim)

        self.comb_embed = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.comb_embed2 = resnet(hidden_dim, hidden_dim)


    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = self.act(self.embed_op(nodes.data['node_feats'])) 

        if ('bwd_node_embeds' in nodes.data.keys()):
            embeds = self.act(self.comb_embed(th.cat((embeds, nodes.data['bwd_node_embeds']), axis=-1)))
            embeds = self.act(self.comb_embed1(embeds)) 
            embeds = self.act(self.comb_embed2(embeds)) 

        if ('bwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['bwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)
            embeds       = self.act(self.node_embeds(concat_feats))
            embeds       = self.act(self.node_embeds1(embeds))
            embeds       = self.act(self.node_embeds2(embeds))
        return {'bwd_node_embeds': embeds}

    #
    def msg_func(self, edges):                              
        return {'bwd_msgs': edges.src['bwd_node_embeds']}

    # sum all messages (bound on cardinality of in set is not known)
    def redux_func(self, nodes):       
        n_count  = len(nodes)
        msgs_sum = th.sum(nodes.mailbox['bwd_msgs'], dim=1).reshape((n_count, 1, self.hidden_dim))

        msgs_redux = self.act(self.mp_bwd(msgs_sum))
        msgs_redux = self.act(self.mp_bwd1(msgs_redux))
        msgs_redux = self.act(self.mp_bwd2(msgs_redux))
        return {'bwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph, bwd_topo_order=None):
        if (bwd_topo_order == None):
            bwd_topo_order = dgl.topological_nodes_generator(graph)
        graph.prop_nodes(bwd_topo_order, self.msg_func, self.redux_func, self.apply_func)        

        return graph.ndata['bwd_node_embeds'], bwd_topo_order


#
#
#
class bid_mpgnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(bid_mpgnn, self).__init__()

        self.act = nn.Tanh()

        self.fwd_nets = nn.ModuleList([fwd_mpgnn(in_dim, hidden_dim)])
        self.bwd_nets = nn.ModuleList([bwd_mpgnn(in_dim, hidden_dim)])
        self.comb_embeds = nn.ModuleList([resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)])

        if not(TIE_MP_PARAMS):        
            for mp_step in range(MP_STEPS-1):
                self.fwd_nets.append(fwd_mpgnn(in_dim, hidden_dim))
                self.bwd_nets.append(bwd_mpgnn(in_dim, hidden_dim))         
                self.comb_embeds.append(resnet(2*hidden_dim, 2*hidden_dim, hidden_dim))

        self.predict  = resnet(hidden_dim, hidden_dim, out_dim)

 
    def forward(self, graph):
        rev_graph = graph.reverse(share_ndata=True, share_edata=True) 

        # init step
        fwd_embed, fwd_topo_order = self.fwd_nets[0](graph)
        bwd_embed, bwd_topo_order = self.bwd_nets[0](rev_graph)
        comb                      = self.comb_embeds[0](th.cat((fwd_embed, bwd_embed), axis=-1))
        
        for mp_step in range(MP_STEPS-1):
            graph.ndata['fwd_node_embeds']     = comb
            rev_graph.ndata['bwd_node_embeds'] = comb

            step_net_idx = 0 if TIE_MP_PARAMS else mp_step

            fwd_embed, fwd_topo_order = self.fwd_nets[step_net_idx](graph, fwd_topo_order)
            bwd_embed, bwd_topo_order = self.bwd_nets[step_net_idx](rev_graph, bwd_topo_order)
            comb                      = self.act(self.comb_embeds[step_net_idx](th.cat((fwd_embed, bwd_embed), axis=-1)))

        pred = self.predict(comb)
        return pred, fwd_topo_order





