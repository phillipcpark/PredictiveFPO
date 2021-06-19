import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from copy import deepcopy

from collections import OrderedDict

from params import *
import sys

#from inference  import *


#
# 
#
class resnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None):
        super(resnet, self).__init__()
        self.lay1 = nn.Linear(in_dim, hidden_dim)
        self.lay2 = nn.Linear(hidden_dim, hidden_dim)
        self.lay3 = nn.Linear(hidden_dim, hidden_dim) if (out_dim==None) else nn.Linear(hidden_dim, out_dim)
        self.act  = nn.Tanh()

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

        # NOTE: having problem tracking gradients when using ModuleList here; model hierarchy too deep?
        #       Explicitly instantiating each residual block, for now...

        # embedding aggregate across each depth partition
        self.d1_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.d2_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.d3_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_comb_embed1 = resnet(hidden_dim, hidden_dim)

        # node embeddings 
        self.d1_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.d2_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.d3_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_node_embeds1 = resnet(hidden_dim, hidden_dim)

        # transcendnetal message-passing nets
        self.d1_mp_unary  = resnet(hidden_dim, hidden_dim)
        self.d1_mp_unary1 = resnet(hidden_dim, hidden_dim)
        self.d2_mp_unary  = resnet(hidden_dim, hidden_dim)
        self.d2_mp_unary1 = resnet(hidden_dim, hidden_dim)
        self.d3_mp_unary  = resnet(hidden_dim, hidden_dim)
        self.d3_mp_unary1 = resnet(hidden_dim, hidden_dim)

        # binary op message-passing nets
        self.d1_mp_binary  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_mp_binary1 = resnet(hidden_dim, hidden_dim)
        self.d2_mp_binary  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_mp_binary1 = resnet(hidden_dim, hidden_dim)
        self.d3_mp_binary  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_mp_binary1 = resnet(hidden_dim, hidden_dim)
         
    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        op_embeds  = self.act(self.embed_op(nodes.data['node_feats']))
        embeds     = op_embeds

        # message-passing steps >1 will enter this branch
        if ('fwd_node_embeds' in nodes.data.keys()):
            embeds = nodes.data['fwd_node_embeds']

        # constant nodes will not enter this branch
        if ('fwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['fwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)

            # compute new embeddings on previous message-passing step's 
            if (nodes.data['fwd_depth'][0] == 0):
                embeds  = self.act(self.d1_node_embeds(concat_feats))
                embeds  = self.act(self.d1_node_embeds1(embeds))  
            elif (nodes.data['fwd_depth'][0] == 1):
                embeds  = self.act(self.d2_node_embeds(concat_feats))
                embeds  = self.act(self.d2_node_embeds1(embeds))  
            else:
                embeds  = self.act(self.d3_node_embeds(concat_feats))
                embeds  = self.act(self.d3_node_embeds1(embeds))  
            
            if ('fwd_node_embeds' in nodes.data.keys()):           
                if (nodes.data['fwd_depth'][0] == 0):
                    embeds = self.act(self.d1_comb_embed(th.cat((embeds, op_embeds), axis=-1)))
                    embeds = self.act(self.d1_comb_embed1(embeds)) 
                elif (nodes.data['fwd_depth'][0] == 1):
                    embeds = self.act(self.d2_comb_embed(th.cat((embeds, op_embeds), axis=-1)))
                    embeds = self.act(self.d2_comb_embed1(embeds)) 
                else: 
                    embeds = self.act(self.d2_comb_embed(th.cat((embeds, op_embeds), axis=-1)))
                    embeds = self.act(self.d2_comb_embed1(embeds)) 
 
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
            if (nodes.data['fwd_depth'][0] == 0):
                msgs_redux = self.act(self.d1_mp_unary(nodes.mailbox['fwd_msgs']))
                msgs_redux  = self.act(self.d1_mp_unary1(msgs_redux))
            elif (nodes.data['fwd_depth'][0] == 1):
                msgs_redux = self.act(self.d2_mp_unary(nodes.mailbox['fwd_msgs']))
                msgs_redux  = self.act(self.d2_mp_unary1(msgs_redux))
            else:
                msgs_redux = self.act(self.d3_mp_unary(nodes.mailbox['fwd_msgs']))
                msgs_redux  = self.act(self.d3_mp_unary1(msgs_redux))
        else:
            msgs_concat = nodes.mailbox['fwd_msgs'].reshape((node_count, 1, 2*self.hidden_dim))

            if (nodes.data['fwd_depth'][0] == 0):
                msgs_redux  = self.act(self.d1_mp_binary(msgs_concat)) 
                msgs_redux  = self.act(self.d1_mp_binary1(msgs_redux))
            elif (nodes.data['fwd_depth'][0] == 1):
                msgs_redux  = self.act(self.d2_mp_binary(msgs_concat)) 
                msgs_redux  = self.act(self.d2_mp_binary1(msgs_redux))
            else:
                msgs_redux  = self.act(self.d2_mp_binary(msgs_concat)) 
                msgs_redux  = self.act(self.d2_mp_binary1(msgs_redux))
        return {'fwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph, use_gpu):
        if (use_gpu):
            with th.cuda.device(0):
                fwd_topo_order = dgl.topological_nodes_generator(graph)  
                fwd_topo_order = [node_front.to(th.device("cuda:0")) for node_front in fwd_topo_order]     
                graph.prop_nodes(fwd_topo_order, self.msg_func, self.redux_func, self.apply_func) 
        else:
            fwd_topo_order = dgl.topological_nodes_generator(graph)  
            graph.prop_nodes(fwd_topo_order, self.msg_func, self.redux_func, self.apply_func) 
        return graph.ndata['fwd_node_embeds'] 

#
#
#
class bwd_mpgnn(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(bwd_mpgnn, self).__init__() 

        self.hidden_dim = hidden_dim
        self.act = nn.Tanh()

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        # depthwise message-passing residual blocks
        self.d1_mp_bwd  = resnet(hidden_dim, hidden_dim)
        self.d1_mp_bwd1 = resnet(hidden_dim, hidden_dim)
        self.d2_mp_bwd  = resnet(hidden_dim, hidden_dim)
        self.d2_mp_bwd1 = resnet(hidden_dim, hidden_dim)
        self.d3_mp_bwd  = resnet(hidden_dim, hidden_dim)
        self.d3_mp_bwd1 = resnet(hidden_dim, hidden_dim)

        # depthwise node embeddings
        self.d1_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.d2_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.d3_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_node_embeds1 = resnet(hidden_dim, hidden_dim)

        # depthwise embedding aggregation
        self.d1_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.d2_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.d3_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_comb_embed1 = resnet(hidden_dim, hidden_dim)

    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = self.act(self.embed_op(nodes.data['node_feats'])) 

        # message-passing steps > 1 will enter; pass previous embeddings through subsequent layers
        if ('bwd_node_embeds' in nodes.data.keys()):
            if (nodes.data['bwd_depth'][0] == 0):
                embeds = self.act(self.d1_comb_embed(th.cat((embeds, nodes.data['bwd_node_embeds']), axis=-1)))
                embeds = self.act(self.d1_comb_embed1(embeds)) 
            elif (nodes.data['bwd_depth'][0] == 1):
                embeds = self.act(self.d2_comb_embed(th.cat((embeds, nodes.data['bwd_node_embeds']), axis=-1)))
                embeds = self.act(self.d2_comb_embed1(embeds))  
            else: 
                embeds = self.act(self.d3_comb_embed(th.cat((embeds, nodes.data['bwd_node_embeds']), axis=-1)))
                embeds = self.act(self.d3_comb_embed1(embeds)) 
 
        if ('bwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['bwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)

            if (nodes.data['bwd_depth'][0] == 0):
                embeds = self.act(self.d1_node_embeds(concat_feats))
                embeds = self.act(self.d1_node_embeds1(embeds))
            elif (nodes.data['bwd_depth'][0] == 1):
                embeds = self.act(self.d2_node_embeds(concat_feats))
                embeds = self.act(self.d2_node_embeds1(embeds))
            else:
                embeds = self.act(self.d3_node_embeds(concat_feats))
                embeds = self.act(self.d3_node_embeds1(embeds))
        return {'bwd_node_embeds': embeds}

    #
    def msg_func(self, edges):                              
        return {'bwd_msgs': edges.src['bwd_node_embeds']}

    # sum all messages (bound on in set cardinality is not known a priori)
    def redux_func(self, nodes):       
        n_count  = len(nodes)
        msgs_sum = th.sum(nodes.mailbox['bwd_msgs'], dim=1).reshape((n_count, 1, self.hidden_dim))

        # depthwise message reduction 
        if (nodes.data['bwd_depth'][0] == 0):
            msgs_redux = self.act(self.d1_mp_bwd(msgs_sum))
            msgs_redux = self.act(self.d1_mp_bwd1(msgs_redux))
        elif (nodes.data['bwd_depth'][0] == 1):
            msgs_redux = self.act(self.d2_mp_bwd(msgs_sum))
            msgs_redux = self.act(self.d2_mp_bwd1(msgs_redux))
        else:
            msgs_redux = self.act(self.d3_mp_bwd(msgs_sum))
            msgs_redux = self.act(self.d3_mp_bwd1(msgs_redux))
        return {'bwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph, use_gpu): 
        if (use_gpu):
            with th.cuda.device(0):
                bwd_topo_order = dgl.topological_nodes_generator(graph)
                bwd_topo_order = [node_front.to(th.device("cuda:0")) for node_front in bwd_topo_order]
                graph.prop_nodes(bwd_topo_order, self.msg_func, self.redux_func, self.apply_func)        

        else:
            bwd_topo_order = dgl.topological_nodes_generator(graph)
            graph.prop_nodes(bwd_topo_order, self.msg_func, self.redux_func, self.apply_func)        

        return graph.ndata['bwd_node_embeds'] 




#
#
#
class bignn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(bignn, self).__init__()

        self.act = nn.Tanh()

        self.fwd_nets    = nn.ModuleList([fwd_mpgnn(in_dim, hidden_dim)])
        self.bwd_nets    = nn.ModuleList([bwd_mpgnn(in_dim, hidden_dim)])
        self.comb_embeds = nn.ModuleList([resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)])

        if not(TIE_MP_PARAMS):        
            for mp_step in range(MP_STEPS-1):
                self.fwd_nets.append(fwd_mpgnn(in_dim, hidden_dim))
                self.bwd_nets.append(bwd_mpgnn(in_dim, hidden_dim))         
                self.comb_embeds.append(resnet(2*hidden_dim, 2*hidden_dim, hidden_dim))
        self.predict  = resnet(hidden_dim, hidden_dim, out_dim)

    # 
    def load_hier_state(self, flat_dict):
        if not(MP_STEPS < 10):
            raise ValueError('MP_STEPS must be < 10 to load model; TODO: update parser to handle arbitrary # of digits')

        fwd_sd  = OrderedDict() 
        bwd_sd  = OrderedDict()
        comb_sd = OrderedDict()
        pred_sd = OrderedDict()

        for k, v in flat_dict.items():
            net_id = k.split('.')[0]
            if net_id == 'fwd_nets':
                fwd_sd[k[9:]] = v
            elif(net_id == 'bwd_nets'):
                bwd_sd[k[9:]] = v
            elif(net_id == 'comb_embeds'):
                comb_sd[k[12:]] = v
            elif(net_id == 'predict'):
                pred_sd[k[8:]] = v
            else:
                raise ValueError('Unrecognized key encountered when loading model')

        for fwd_idx in range(MP_STEPS):             
            step_sd = OrderedDict()

            for k, v in fwd_sd.items():
                step_id = int(k.split('.')[0])
               
                if (step_id == fwd_idx):
                    # requires MP_STEPS < 10
                    if (fwd_idx < 10):
                        step_sd[k[2:]] = v
                    else:
                        step_sd[k[3:]] = v

            self.fwd_nets[fwd_idx].load_state_dict(step_sd, strict=True)

        for bwd_idx in range(MP_STEPS):             
            step_sd = OrderedDict()

            for k, v in bwd_sd.items():
                step_id = int(k.split('.')[0])
               
                if (step_id == bwd_idx):
                    # requires MP_STEPS < 10
                    if (bwd_idx < 10):
                        step_sd[k[2:]] = v
                    else:
                        step_sd[k[3:]] = v
            self.bwd_nets[bwd_idx].load_state_dict(step_sd, strict=True)

        pred_net_sd = OrderedDict()

        for comb_idx in range(MP_STEPS):
            step_sd = OrderedDict()
            for k, v in comb_sd.items():
                step_id = int(k.split('.')[0])                            
                if (step_id == comb_idx):
                    # requires MP_STEPS < 10
                    if (comb_idx < 10):
                        step_sd[k[2:]] = v   
                    else:
                        step_sd[k[3:]] = v
            self.comb_embeds[comb_idx].load_state_dict(step_sd, strict=True)        
        self.predict.load_state_dict(pred_sd, strict=True)


    # 
    def forward(self, graph, use_gpu):
        rev_graph      = graph.reverse(share_ndata=True, share_edata=True) 
        fwd_topo_order = dgl.topological_nodes_generator(graph)
        rev_topo_order = dgl.topological_nodes_generator(rev_graph)

        # vertices are routed to different bignns, based on depth 
        curr_fwd_depth = curr_bwd_depth = 0
        fwd_depth_map  = [None for i in range(len(graph.nodes()))]
        bwd_depth_map  = [None for i in range(len(graph.nodes()))]

        for front in fwd_topo_order:
            for node in front:
                depth_bin = 0 if curr_fwd_depth < 10 else 1
                if curr_fwd_depth > 20:
                    depth_bin = 2
                fwd_depth_map[node] = depth_bin 
            curr_fwd_depth += 1

        for front in rev_topo_order:
            for node in front:
                depth_bin = 0 if curr_bwd_depth < 10 else 1
                if curr_bwd_depth > 20:
                    depth_bin = 2

                bwd_depth_map[node] = depth_bin 
            curr_bwd_depth += 1

        fwd_depth_map = th.tensor(fwd_depth_map)
        bwd_depth_map = th.tensor(bwd_depth_map)

        if (use_gpu):
            fwd_depth_map = fwd_depth_map.to('cuda:0')
            bwd_depth_map = bwd_depth_map.to('cuda:0')

        graph.ndata['fwd_depth']     = fwd_depth_map
        rev_graph.ndata['bwd_depth'] = bwd_depth_map

        # init step
        fwd_embed = self.fwd_nets[0](graph, use_gpu)
        bwd_embed = self.bwd_nets[0](rev_graph, use_gpu)
        comb      = self.comb_embeds[0](th.cat((fwd_embed, bwd_embed), axis=-1))
 
        for mp_step in range(MP_STEPS-1):
            graph.ndata['fwd_node_embeds']     = comb
            rev_graph.ndata['bwd_node_embeds'] = comb

            step_net_idx = 0 if TIE_MP_PARAMS else mp_step

            fwd_embed = self.fwd_nets[step_net_idx](graph, use_gpu)
            bwd_embed = self.bwd_nets[step_net_idx](rev_graph, use_gpu)
            comb      = self.act(self.comb_embeds[step_net_idx](th.cat((fwd_embed, bwd_embed), axis=-1)))
        pred = self.predict(comb)
        return pred, fwd_topo_order






