import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from copy import deepcopy

from params import *
import sys
from tr_eval_model import *



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

        #
        self.d1_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.d2_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.d3_comb_embed  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_comb_embed1 = resnet(hidden_dim, hidden_dim)

        #
        self.d1_node_embeds = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.d2_node_embeds = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.d3_node_embeds = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_node_embeds1 = resnet(hidden_dim, hidden_dim)

        #
        self.d1_mp_unary = resnet(hidden_dim, hidden_dim)
        self.d1_mp_unary1 = resnet(hidden_dim, hidden_dim)
        self.d2_mp_unary = resnet(hidden_dim, hidden_dim)
        self.d2_mp_unary1 = resnet(hidden_dim, hidden_dim)
        self.d3_mp_unary = resnet(hidden_dim, hidden_dim)
        self.d3_mp_unary1 = resnet(hidden_dim, hidden_dim)

        #
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

        # subsequent steps of MP will entire this branch
        if ('fwd_node_embeds' in nodes.data.keys()):
            embeds = nodes.data['fwd_node_embeds']


        # constant nodes will not enter this branch
        if ('fwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['fwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)

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
    def forward(self, graph, use_gpu): #, fwd_topo_order = None):

        #init_fwd_topo_order = deepcopy(fwd_topo_order)

        if (use_gpu):
            with th.cuda.device(0):
                #if (fwd_topo_order == None):   
                fwd_topo_order = dgl.topological_nodes_generator(graph)  
                fwd_topo_order = [node_front.to(th.device("cuda:0")) for node_front in fwd_topo_order]     
                graph.prop_nodes(fwd_topo_order, self.msg_func, self.redux_func, self.apply_func) 

        else:
            fwd_topo_order = dgl.topological_nodes_generator(graph)  
            graph.prop_nodes(fwd_topo_order, self.msg_func, self.redux_func, self.apply_func) 

        return graph.ndata['fwd_node_embeds'] #, init_fwd_topo_order

#
#
#
class bwd_mpgnn(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(bwd_mpgnn, self).__init__() 

        self.hidden_dim = hidden_dim
        self.act = nn.Tanh()

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        #
        self.d1_mp_bwd  = resnet(hidden_dim, hidden_dim)
        self.d1_mp_bwd1 = resnet(hidden_dim, hidden_dim)
        self.d2_mp_bwd  = resnet(hidden_dim, hidden_dim)
        self.d2_mp_bwd1 = resnet(hidden_dim, hidden_dim)
        self.d3_mp_bwd  = resnet(hidden_dim, hidden_dim)
        self.d3_mp_bwd1 = resnet(hidden_dim, hidden_dim)

        #
        self.d1_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.d2_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_node_embeds1 = resnet(hidden_dim, hidden_dim)
        self.d3_node_embeds  = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_node_embeds1 = resnet(hidden_dim, hidden_dim)

        #
        self.d1_comb_embed = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d1_comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.d2_comb_embed = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d2_comb_embed1 = resnet(hidden_dim, hidden_dim)
        self.d3_comb_embed = resnet(2*hidden_dim, 2*hidden_dim, hidden_dim)
        self.d3_comb_embed1 = resnet(hidden_dim, hidden_dim)


    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = self.act(self.embed_op(nodes.data['node_feats'])) 


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
                embeds       = self.act(self.d1_node_embeds(concat_feats))
                embeds       = self.act(self.d1_node_embeds1(embeds))
            elif (nodes.data['bwd_depth'][0] == 1):
                embeds       = self.act(self.d2_node_embeds(concat_feats))
                embeds       = self.act(self.d2_node_embeds1(embeds))
            else:
                embeds       = self.act(self.d3_node_embeds(concat_feats))
                embeds       = self.act(self.d3_node_embeds1(embeds))

        return {'bwd_node_embeds': embeds}

    #
    def msg_func(self, edges):                              
        return {'bwd_msgs': edges.src['bwd_node_embeds']}

    # sum all messages (bound on cardinality of in set is not known)
    def redux_func(self, nodes):       
        n_count  = len(nodes)
        msgs_sum = th.sum(nodes.mailbox['bwd_msgs'], dim=1).reshape((n_count, 1, self.hidden_dim))

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
    def forward(self, graph, use_gpu): #, bwd_topo_order=None):

        if (use_gpu):
            with th.cuda.device(0):
                bwd_topo_order = dgl.topological_nodes_generator(graph)
                bwd_topo_order = [node_front.to(th.device("cuda:0")) for node_front in bwd_topo_order]
                graph.prop_nodes(bwd_topo_order, self.msg_func, self.redux_func, self.apply_func)        

        else:
                #if (bwd_topo_order == None):
                bwd_topo_order = dgl.topological_nodes_generator(graph)

                # because topo order is a generator
                #init_bwd_topo_order = deepcopy(bwd_topo_order)

                graph.prop_nodes(bwd_topo_order, self.msg_func, self.redux_func, self.apply_func)        

        return graph.ndata['bwd_node_embeds'] #, init_bwd_topo_order


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

 
    def forward(self, graph, use_gpu):

        rev_graph = graph.reverse(share_ndata=True, share_edata=True) 

        fwd_topo_order = dgl.topological_nodes_generator(graph)
        rev_topo_order = dgl.topological_nodes_generator(rev_graph)

         
        # FIXME depthwise model
        depth_delim    = 7       
        curr_fwd_depth = curr_bwd_depth = 0
        fwd_depth_map  = [None for i in range(len(graph.nodes()))]
        bwd_depth_map  = [None for i in range(len(graph.nodes()))]

        for front in fwd_topo_order:
            for node in front:
                depth_bin = 0 if curr_fwd_depth < depth_delim else 1
                depth_bin = 2 if curr_fwd_depth > depth_delim*2 else depth_bin
                fwd_depth_map[node] = depth_bin 
            curr_fwd_depth += 1

        for front in rev_topo_order:
            for node in front:
                depth_bin           = 0 if curr_bwd_depth < depth_delim else 1
                depth_bin           = 2 if curr_bwd_depth > depth_delim*2 else depth_bin
                bwd_depth_map[node] = depth_bin 
            curr_bwd_depth += 1

        fwd_depth_map = th.tensor(fwd_depth_map)
        bwd_depth_map = th.tensor(bwd_depth_map)

        if (use_gpu):
            fwd_depth_map = fwd_depth_map.to('cuda:0')
            bwd_depth_map = bwd_depth_map.to('cuda:0')

        graph.ndata['fwd_depth'] = fwd_depth_map
        rev_graph.ndata['bwd_depth'] = bwd_depth_map
        #####



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


#FIXME FIXME sc
#mod = bid_mpgnn(32, 32, 32)

#mod.load_state_dict(th.load('0_MOD_TST_PATH'), strict=False)
#th.save(mod.state_dict(), '0_MOD_TST_PATH')



