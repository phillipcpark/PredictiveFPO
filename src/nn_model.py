import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from params import *
import sys

# NOTE: these GNNs don't seem to be working well...maybe a problem with how ModuleList is used?
class fwd_gnn(nn.Module):
    def __init__(self, feat_dim, hidden_dim, layers, lay_conn = 'residual'):
        super(fwd_gnn, self).__init__() 
        self.hidden_dim = hidden_dim
        self.layers     = layers
        self.resid_conn = True if (lay_conn == 'residual' and layers>2) else False
        self.dense_conn = True if lay_conn == 'dense' else False

        mult = 1
        if (self.dense_conn):
            mult = 2

        # layer defs
        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        self.node_embeds = nn.ModuleList([nn.Linear(2*hidden_dim, hidden_dim)])
        self.node_embeds.append(nn.Linear(hidden_dim, hidden_dim))
        self.node_embeds.extend([nn.Linear(mult*hidden_dim, hidden_dim) for lay in range(layers-1)])

        self.mp_unary = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.mp_unary.append(nn.Linear(hidden_dim, hidden_dim))
        self.mp_unary.extend([nn.Linear(mult*hidden_dim, hidden_dim) for lay in range(layers-1)])

        self.mp_binary = nn.ModuleList([nn.Linear(2*hidden_dim, hidden_dim)])
        self.mp_binary.append(nn.Linear(hidden_dim, hidden_dim))
        self.mp_binary.extend([nn.Linear(mult*hidden_dim, hidden_dim) for lay in range(layers-1)])

        self.act = nn.ReLU()

    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = self.act(self.embed_op(nodes.data['node_feats']))

        # constant nodes will not enter this branch
        if ('fwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['fwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)
            embeds       = self.act(self.node_embeds[0](concat_feats))
                        
            if not(self.resid_conn or self.dense_conn):
                for lay in self.node_embeds[1:]:
                    embeds = self.act(lay(embeds))                                            

            else:
                lay_acts = [embeds]
                lay_acts.append(self.act(self.node_embeds[1](embeds)))

                for lay in self.node_embeds[2:]:
                    if (self.resid_conn):
                        embeds = self.act(lay(th.add(lay_acts[-2], lay_acts[-1])))                                           
                    else:
                        embeds = self.act(lay(th.cat((lay_acts[-2], lay_acts[-1]), -1)))
                    lay_acts.append(embeds)

        return {'fwd_node_embeds': embeds}

    #
    def msg_func(self, edges):                              
        return {'fwd_msgs': edges.src['fwd_node_embeds']}
 
    #
    def redux_func(self, nodes):       

        # FIXME all grouped nodes, conveniently, share this flag value -why?
        is_unary   = nodes.data['is_unary_op'][0]

        node_count = len(nodes)
        msgs_redux = None
      
        if (is_unary==True):
            msgs_redux = self.act(self.mp_unary[0](nodes.mailbox['fwd_msgs']))                  

            if not(self.resid_conn or self.dense_conn):            
                for lay in self.mp_unary[1:]:
                    msgs_redux = self.act(lay(msgs_redux))

            else: 
                lay_acts = [msgs_redux]
                lay_acts.append(self.act(self.mp_unary[1](msgs_redux)))

                for lay in self.mp_unary[2:]:
                    if (self.resid_conn):
                        msgs_redux = self.act(lay(th.add(lay_acts[-2], lay_acts[-1])))                                           
                    else:
                        msgs_redux = self.act(lay(th.cat((lay_acts[-2], lay_acts[-1]), -1)))
                    lay_acts.append(msgs_redux)

        else:
            # FIXME is L-R ordering guaranteed through ordering of edges added to graph? 
            msgs_concat = nodes.mailbox['fwd_msgs'].reshape((node_count, 1, 2*self.hidden_dim))
            msgs_redux  = self.act(self.mp_binary[0](msgs_concat))                  

            if not(self.resid_conn or self.dense_conn):            
                for lay in self.mp_binary[1:]:
                    msgs_redux = self.act(lay(msgs_redux))
            else:
                lay_acts = [msgs_redux]
                lay_acts.append(self.act(self.mp_binary[1](msgs_redux)))

                for lay in self.mp_binary[2:]:
                    if (self.resid_conn):
                        msgs_redux = self.act(lay(th.add(lay_acts[-2], lay_acts[-1])))                                           
                    else:
                        msgs_redux = self.act(lay(th.cat((lay_acts[-2], lay_acts[-1]), -1)))
                    lay_acts.append(msgs_redux)

        return {'fwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph):
        fwd_topo_order = dgl.topological_nodes_generator(graph)
        graph.prop_nodes(fwd_topo_order, self.msg_func, self.redux_func, self.apply_func) 
        return graph.ndata['fwd_node_embeds'], fwd_topo_order
 
#                         
#
#
class bwd_gnn(nn.Module):
    def __init__(self, feat_dim, hidden_dim, layers, lay_conn='residual'):
        super(bwd_gnn, self).__init__() 
        self.hidden_dim = hidden_dim
        self.resid_conn = True if (lay_conn == 'residual' and layers>2) else False
        self.dense_conn = True if lay_conn == 'dense' else False

        mult = 1
        if (self.dense_conn):
            mult = 2 

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        self.mp_bwd  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.mp_bwd.append(nn.Linear(hidden_dim, hidden_dim))
        self.mp_bwd.extend([nn.Linear(mult*hidden_dim, hidden_dim) for lay in range(layers-1)])

        self.node_embeds = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim)])
        self.node_embeds.append(nn.Linear(hidden_dim, hidden_dim))
        self.node_embeds.extend([nn.Linear(mult*hidden_dim, hidden_dim) for lay in range(layers-1)])
             
        self.act = nn.ReLU()

    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = th.tanh(self.embed_op(nodes.data['node_feats'])) 

        if ('bwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['bwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)
            embeds       = self.act(self.node_embeds[0](concat_feats))

            if not (self.resid_conn or self.dense_conn):
                for lay in self.node_embeds[1:]:               
                    embeds = self.act(lay(embeds))
            else:
                lay_acts = [embeds]
                lay_acts.append(self.act(self.node_embeds[1](embeds)))

                for lay in self.node_embeds[2:]:
                    if (self.resid_conn):
                        embeds = self.act(lay(th.add(lay_acts[-2], lay_acts[-1])))                                           
                    else:                        
                        embeds = self.act(lay(th.cat((lay_acts[-2], lay_acts[-1]), -1)))                                            
                    lay_acts.append(embeds)                                          
        return {'bwd_node_embeds': embeds}

    #
    def msg_func(self, edges):                              
        return {'bwd_msgs': edges.src['bwd_node_embeds']}

    # sum all messages (bound on cardinality of in set is not known)
    def redux_func(self, nodes):       
        n_count  = len(nodes)
        msgs_sum = th.sum(nodes.mailbox['bwd_msgs'], dim=1).reshape((n_count, 1, self.hidden_dim))

        msgs_redux  = self.act(self.mp_bwd[0](msgs_sum))

        if not(self.resid_conn or self.dense_conn):
            for lay in self.mp_bwd[1:]:
                msgs_redux = self.act(lay(msgs_redux))

        else:
            lay_acts = [msgs_redux]
            lay_acts.append(self.act(self.mp_bwd[1](msgs_redux)))

            for lay in self.mp_bwd[2:]:
                if (self.resid_conn):
                    msgs_redux = self.act(lay(th.add(lay_acts[-2], lay_acts[-1])))                                           
                else:
                    msgs_redux = self.act(lay(th.cat((lay_acts[-2], lay_acts[-1]), -1)))
                lay_acts.append(msgs_redux)
   
        return {'bwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph):
        bwd_topo_order = dgl.topological_nodes_generator(graph)
        graph.prop_nodes(bwd_topo_order, self.msg_func, self.redux_func, self.apply_func)        

        return graph.ndata['bwd_node_embeds'], bwd_topo_order

#
#
#
class comb_state(nn.Module):
    def __init__(self, feat_dim, hidden_dim, layers, classes, lay_conn='residual'):
        super(comb_state, self).__init__()
        self.comb_embed = nn.Linear(2 * hidden_dim, hidden_dim)
        self.resid_conn = True if (lay_conn == 'residual' and layers>2) else False
        self.dense_conn = True if lay_conn == 'dense' else False

        mult = 1
        if (self.dense_conn):
            mult = 2

        self.predict  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.predict.append(nn.Linear(hidden_dim, hidden_dim))
        self.predict.extend([nn.Linear(mult*hidden_dim, hidden_dim) for lay in range(layers-1)])
        self.predict.append(nn.Linear(mult*hidden_dim, classes))
  
        self.act = nn.ReLU()  
          

    def forward(self, fwd_states, bwd_states):
        concat_states = th.cat((fwd_states, bwd_states), -1)
        comb_embed    = th.tanh(self.comb_embed(concat_states))       

        pred = self.act(self.predict[0](comb_embed))

        if not(self.resid_conn or self.dense_conn):
            for lay in self.predict[1:-1]:
                predict = self.act(lay(predict)) 

        else:
            lay_acts = [pred]
            lay_acts.append(self.act(self.predict[1](pred)))

            for lay in self.predict[2:-1]:
                if (self.resid_conn):
                    pred = self.act(lay(th.add(lay_acts[-2], lay_acts[-1])))                                           
                else:
                    pred = self.act(lay(th.cat((lay_acts[-2], lay_acts[-1]), -1)))
                lay_acts.append(pred)

            if (self.resid_conn):                    
                pred = th.sigmoid(self.predict[-1](th.add(lay_acts[-2], lay_acts[-1])))  
            else:
                pred = th.sigmoid(self.predict[-1](th.cat((lay_acts[-2], lay_acts[-1]), -1)))

        return pred







#
#
#
#
#
#
#
#
class fwd_gnn_dense(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(fwd_gnn_dense, self).__init__() 
        self.hidden_dim = hidden_dim

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        self.node_embeds  = nn.Linear(2*hidden_dim, hidden_dim)
        self.node_embeds1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.node_embeds2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.node_embeds3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.node_embeds4 = nn.Linear(2*hidden_dim, hidden_dim)

        self.mp_unary  = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary1 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_unary3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_unary4 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_unary5 = nn.Linear(2*hidden_dim, hidden_dim)

        self.mp_binary  = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_binary1 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_binary3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_binary4 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_binary5 = nn.Linear(2*hidden_dim, hidden_dim)

        self.act = nn.Tanh()

    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = self.act(self.embed_op(nodes.data['node_feats']))

        # constant nodes will not enter this branch
        if ('fwd_msgs_redux' in nodes.data.keys()):             
            embeds0 = embeds

            concat_feats = th.cat((embeds0, nodes.data['fwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)
            embeds       = self.act(self.node_embeds(concat_feats))
            embeds       = self.act(self.node_embeds1(th.cat((embeds, embeds0),-1)))  
            embeds       = self.act(self.node_embeds2(th.cat((embeds, embeds0),-1)))                        
            embeds       = self.act(self.node_embeds3(th.cat((embeds, embeds0),-1)))  
            embeds       = self.act(self.node_embeds4(th.cat((embeds, embeds0),-1)))                        

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
            msgs_redux0 = self.act(self.mp_unary(nodes.mailbox['fwd_msgs'])) 
            msgs_redux  = self.act(self.mp_unary1(msgs_redux0))
            msgs_redux  = self.act(self.mp_unary2(th.cat((msgs_redux, msgs_redux0),-1)))
            msgs_redux  = self.act(self.mp_unary3(th.cat((msgs_redux, msgs_redux0),-1)))
            msgs_redux  = self.act(self.mp_unary4(th.cat((msgs_redux, msgs_redux0),-1)))
            msgs_redux  = self.act(self.mp_unary5(th.cat((msgs_redux, msgs_redux0),-1)))

        else:
            msgs_concat = nodes.mailbox['fwd_msgs'].reshape((node_count, 1, 2*self.hidden_dim))
            msgs_redux0 = self.act(self.mp_binary(msgs_concat)) 
            msgs_redux  = self.act(self.mp_binary1(msgs_redux0))
            msgs_redux  = self.act(self.mp_binary2(th.cat((msgs_redux, msgs_redux0),-1)))
            msgs_redux  = self.act(self.mp_binary3(th.cat((msgs_redux, msgs_redux0),-1)))
            msgs_redux  = self.act(self.mp_binary4(th.cat((msgs_redux, msgs_redux0),-1)))
            msgs_redux  = self.act(self.mp_binary5(th.cat((msgs_redux, msgs_redux0),-1)))

        return {'fwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph):
        fwd_topo_order = dgl.topological_nodes_generator(graph)
        graph.prop_nodes(fwd_topo_order, self.msg_func, self.redux_func, self.apply_func) 
        return graph.ndata['fwd_node_embeds'], fwd_topo_order
 
#                         
#
#
class bwd_gnn_dense(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(bwd_gnn_dense, self).__init__() 
        self.hidden_dim = hidden_dim

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        self.mp_bwd  = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd1 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_bwd3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_bwd4 = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_bwd5 = nn.Linear(2*hidden_dim, hidden_dim)

        self.node_embeds  = nn.Linear(2 * hidden_dim, hidden_dim)
        self.node_embeds1 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.node_embeds3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.node_embeds4 = nn.Linear(2*hidden_dim, hidden_dim)
        self.node_embeds5 = nn.Linear(2*hidden_dim, hidden_dim)             

        self.act = nn.Tanh()


    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = th.tanh(self.embed_op(nodes.data['node_feats'])) 

        if ('bwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['bwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)
            embeds0      = self.act(self.node_embeds(concat_feats))
            embeds       = self.act(self.node_embeds1(embeds0))
            embeds       = self.act(self.node_embeds2(th.cat((embeds, embeds0),-1)))
            embeds       = self.act(self.node_embeds3(th.cat((embeds, embeds0),-1)))
            embeds       = self.act(self.node_embeds4(th.cat((embeds, embeds0),-1)))
            embeds       = self.act(self.node_embeds5(th.cat((embeds, embeds0),-1)))

        return {'bwd_node_embeds': embeds}

    #
    def msg_func(self, edges):                              
        return {'bwd_msgs': edges.src['bwd_node_embeds']}

    # sum all messages (bound on cardinality of in set is not known)
    def redux_func(self, nodes):       
        n_count  = len(nodes)
        msgs_sum = th.sum(nodes.mailbox['bwd_msgs'], dim=1).reshape((n_count, 1, self.hidden_dim))

        msgs_redux0  = self.act(self.mp_bwd(msgs_sum))
        msgs_redux   = self.act(self.mp_bwd1(msgs_redux0))
        msgs_redux   = self.act(self.mp_bwd2(th.cat((msgs_redux, msgs_redux0),-1)))
        msgs_redux   = self.act(self.mp_bwd3(th.cat((msgs_redux, msgs_redux0),-1)))
        msgs_redux   = self.act(self.mp_bwd4(th.cat((msgs_redux, msgs_redux0),-1)))
        msgs_redux   = self.act(self.mp_bwd5(th.cat((msgs_redux, msgs_redux0),-1)))

        return {'bwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph):
        bwd_topo_order = dgl.topological_nodes_generator(graph)
        graph.prop_nodes(bwd_topo_order, self.msg_func, self.redux_func, self.apply_func)        

        return graph.ndata['bwd_node_embeds'], bwd_topo_order

#
#
#
class comb_state_dense(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(comb_state_dense, self).__init__()
        self.comb_embed = nn.Linear(2 * hidden_dim, hidden_dim)

        self.predict  = nn.Linear(hidden_dim, hidden_dim)
        self.predict1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.predict2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.predict3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.predict4 = nn.Linear(2*hidden_dim, CLASSES)
  
        self.act = nn.Tanh()  

        
    def forward(self, fwd_states, bwd_states):
        concat_states = th.cat((fwd_states, bwd_states), -1)
        comb_embed    = th.tanh(self.comb_embed(concat_states))       

        pred = self.act(self.predict(comb_embed))
        pred = self.act(self.predict1(th.cat((pred, comb_embed),-1)))
        pred = self.act(self.predict2(th.cat((pred, comb_embed),-1)))
        pred = self.act(self.predict3(th.cat((pred, comb_embed),-1)))

        pred = th.sigmoid(self.predict4(th.cat((pred, comb_embed),-1))) 

        return pred




#
#
#
#
#
#
#
#
class fwd_gnn_resid(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(fwd_gnn_resid, self).__init__() 
        self.hidden_dim = hidden_dim

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        self.node_embeds  = nn.Linear(2*hidden_dim, hidden_dim)
        self.node_embeds1 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds2 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds3 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds4 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds5 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds6 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds7 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds8 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds9 = nn.Linear(hidden_dim, hidden_dim)

        self.mp_unary  = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary1 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary2 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary3 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary4 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary5 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary6 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary7 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary8 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_unary9 = nn.Linear(hidden_dim, hidden_dim)


        self.mp_binary  = nn.Linear(2*hidden_dim, hidden_dim)
        self.mp_binary1 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary2 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary3 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary4 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary5 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary6 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary7 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary8 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_binary9 = nn.Linear(hidden_dim, hidden_dim)

        self.act = nn.Tanh()

    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = self.act(self.embed_op(nodes.data['node_feats']))

        # constant nodes will not enter this branch
        if ('fwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['fwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)
            embeds1       = self.act(self.node_embeds(concat_feats))
            embeds2       = self.act(self.node_embeds1(embeds1))  
            embeds3       = self.act(self.node_embeds2(embeds2))                        
            embeds4       = self.act(self.node_embeds3(th.add(embeds3, embeds1)))  
            embeds5       = self.act(self.node_embeds4(embeds4))                        
            embeds6       = self.act(self.node_embeds5(embeds5))  
            embeds7       = self.act(self.node_embeds6(th.add(embeds6, embeds4))) 
            embeds8       = self.act(self.node_embeds7(embeds7))                        
            embeds9       = self.act(self.node_embeds8(embeds8))  
            embeds       = self.act(self.node_embeds9(th.add(embeds9, embeds7)))


            #embeds2       = self.act(self.node_embeds1(embeds1))  
            #embeds3       = self.act(self.node_embeds2(embeds2))                        
            #embeds4       = self.act(self.node_embeds3(th.add(embeds3, embeds1)))  
            #embeds5       = self.act(self.node_embeds4(th.add(embeds4, embeds1)))                        
            #embeds       = self.act(self.node_embeds5(th.add(embeds5, embeds1)))  

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
            msgs_redux0 = self.act(self.mp_unary(nodes.mailbox['fwd_msgs']))
            msgs_redux1  = self.act(self.mp_unary1(msgs_redux0))
            msgs_redux2  = self.act(self.mp_unary2(msgs_redux1))
            msgs_redux3  = self.act(self.mp_unary3(th.add(msgs_redux2, msgs_redux0)))
            msgs_redux4  = self.act(self.mp_unary4(msgs_redux3))
            msgs_redux5  = self.act(self.mp_unary5(msgs_redux4))
            msgs_redux6  = self.act(self.mp_unary6(th.add(msgs_redux5, msgs_redux3)))
            msgs_redux7  = self.act(self.mp_unary7(msgs_redux6))
            msgs_redux8  = self.act(self.mp_unary8(msgs_redux7))
            msgs_redux  = self.act(self.mp_unary9(th.add(msgs_redux8, msgs_redux6)))


            #msgs_redux0 = self.act(self.mp_unary(nodes.mailbox['fwd_msgs']))
            #msgs_redux1  = self.act(self.mp_unary1(msgs_redux0))
            #msgs_redux2  = self.act(self.mp_unary2(msgs_redux1))
            #msgs_redux3  = self.act(self.mp_unary3(th.add(msgs_redux2, msgs_redux0)))
            #msgs_redux4  = self.act(self.mp_unary4(th.add(msgs_redux3, msgs_redux0)))
            #msgs_redux  = self.act(self.mp_unary5(th.add(msgs_redux4, msgs_redux0)))

        else:
            msgs_concat = nodes.mailbox['fwd_msgs'].reshape((node_count, 1, 2*self.hidden_dim))
            msgs_redux0 = self.act(self.mp_binary(msgs_concat)) 
            msgs_redux1  = self.act(self.mp_binary1(msgs_redux0))
            msgs_redux2  = self.act(self.mp_binary2(msgs_redux1))
            msgs_redux3  = self.act(self.mp_binary3(th.add(msgs_redux2, msgs_redux0)))
            msgs_redux4  = self.act(self.mp_binary4(msgs_redux3))
            msgs_redux5  = self.act(self.mp_binary5(msgs_redux4))
            msgs_redux6  = self.act(self.mp_binary6(th.add(msgs_redux5, msgs_redux3)))
            msgs_redux7  = self.act(self.mp_binary7(msgs_redux6))
            msgs_redux8  = self.act(self.mp_binary8(msgs_redux7))
            msgs_redux  = self.act(self.mp_binary9(th.add(msgs_redux8, msgs_redux6)))

            #msgs_redux0 = self.act(self.mp_binary(msgs_concat)) 
            #msgs_redux1  = self.act(self.mp_binary1(msgs_redux0))
            #msgs_redux2  = self.act(self.mp_binary2(msgs_redux1))
            #msgs_redux3  = self.act(self.mp_binary3(th.add(msgs_redux2, msgs_redux0)))
            #msgs_redux4  = self.act(self.mp_binary4(th.add(msgs_redux3, msgs_redux0)))
            #msgs_redux   = self.act(self.mp_binary5(th.add(msgs_redux4, msgs_redux0)))

        return {'fwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph):
        fwd_topo_order = dgl.topological_nodes_generator(graph)
        graph.prop_nodes(fwd_topo_order, self.msg_func, self.redux_func, self.apply_func) 
        return graph.ndata['fwd_node_embeds'], fwd_topo_order
 
#                         
#
#
class bwd_gnn_resid(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(bwd_gnn_resid, self).__init__() 
        self.hidden_dim = hidden_dim

        self.embed_op  = nn.Linear(feat_dim, hidden_dim) 

        self.mp_bwd  = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd1 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd2 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd3 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd4 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd5 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd6 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd7 = nn.Linear(hidden_dim, hidden_dim)
        self.mp_bwd8 = nn.Linear(hidden_dim, hidden_dim)

        self.node_embeds  = nn.Linear(2 * hidden_dim, hidden_dim)
        self.node_embeds1 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds2 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds3 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds4 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds5 = nn.Linear(hidden_dim, hidden_dim)             
        self.node_embeds6 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds7 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embeds8 = nn.Linear(hidden_dim, hidden_dim)             
        self.node_embeds9 = nn.Linear(hidden_dim, hidden_dim)

        self.act = nn.Tanh()


    #
    def apply_func(self, nodes):                              
        node_count = len(nodes.nodes())
        embeds     = th.tanh(self.embed_op(nodes.data['node_feats'])) 

        if ('bwd_msgs_redux' in nodes.data.keys()):             
            concat_feats = th.cat((embeds, nodes.data['bwd_msgs_redux'].reshape((node_count, self.hidden_dim))), -1)
            embeds0      = self.act(self.node_embeds(concat_feats))
            embeds1       = self.act(self.node_embeds1(embeds0))
            embeds2       = self.act(self.node_embeds2(embeds1))
            embeds3       = self.act(self.node_embeds3(th.add(embeds2, embeds0)))
            embeds4       = self.act(self.node_embeds4(embeds3))
            embeds5       = self.act(self.node_embeds5(embeds4))
            embeds6       = self.act(self.node_embeds6(th.add(embeds5, embeds3)))
            embeds7       = self.act(self.node_embeds7(embeds6))
            embeds8       = self.act(self.node_embeds8(embeds7))
            embeds       = self.act(self.node_embeds9(th.add(embeds8, embeds6)))


            #embeds1       = self.act(self.node_embeds1(embeds0))
            #embeds2       = self.act(self.node_embeds2(embeds1))
            #embeds3       = self.act(self.node_embeds3(th.add(embeds2, embeds0)))
            #embeds4       = self.act(self.node_embeds4(th.add(embeds3, embeds0)))
            #embeds       = self.act(self.node_embeds5(th.add(embeds4, embeds0)))

        return {'bwd_node_embeds': embeds}

    #
    def msg_func(self, edges):                              
        return {'bwd_msgs': edges.src['bwd_node_embeds']}

    # sum all messages (bound on cardinality of in set is not known)
    def redux_func(self, nodes):       
        n_count  = len(nodes)
        msgs_sum = th.sum(nodes.mailbox['bwd_msgs'], dim=1).reshape((n_count, 1, self.hidden_dim))

        msgs_redux0  = self.act(self.mp_bwd(msgs_sum))
        msgs_redux1   = self.act(self.mp_bwd1(msgs_redux0))
        msgs_redux2   = self.act(self.mp_bwd2(th.add(msgs_redux1, msgs_sum))) #FIXME maybe use ReLU block input as layer input, instead of msg sum
        msgs_redux3   = self.act(self.mp_bwd3(msgs_redux2))
        msgs_redux4   = self.act(self.mp_bwd4(msgs_redux3))
        msgs_redux5   = self.act(self.mp_bwd5(th.add(msgs_redux4, msgs_redux2)))
        msgs_redux6   = self.act(self.mp_bwd6(msgs_redux5))
        msgs_redux7   = self.act(self.mp_bwd7(msgs_redux6))
        msgs_redux   = self.act(self.mp_bwd8(th.add(msgs_redux7, msgs_redux5)))

        #msgs_redux0  = self.act(self.mp_bwd(msgs_sum))
        #msgs_redux1   = self.act(self.mp_bwd1(msgs_redux0))
        #msgs_redux2   = self.act(self.mp_bwd2(th.add(msgs_redux1, msgs_sum)))
        #msgs_redux3   = self.act(self.mp_bwd3(th.add(msgs_redux2, msgs_sum)))
        #msgs_redux   = self.act(self.mp_bwd4(th.add(msgs_redux3, msgs_sum)))

        return {'bwd_msgs_redux': msgs_redux}

    #
    def forward(self, graph):
        bwd_topo_order = dgl.topological_nodes_generator(graph)
        graph.prop_nodes(bwd_topo_order, self.msg_func, self.redux_func, self.apply_func)        

        return graph.ndata['bwd_node_embeds'], bwd_topo_order

#
#
#
class comb_state_resid(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(comb_state_resid, self).__init__()
        self.comb_embed = nn.Linear(2 * hidden_dim, hidden_dim)

        self.predict  = nn.Linear(hidden_dim, hidden_dim)
        self.predict1 = nn.Linear(hidden_dim, hidden_dim)
        self.predict2 = nn.Linear(hidden_dim, hidden_dim)
        self.predict3 = nn.Linear(hidden_dim, hidden_dim)
        self.predict4 = nn.Linear(hidden_dim, hidden_dim)
        self.predict5 = nn.Linear(hidden_dim, hidden_dim)
        self.predict6 = nn.Linear(hidden_dim, hidden_dim)
        self.predict7 = nn.Linear(hidden_dim, hidden_dim)
        self.predict8 = nn.Linear(hidden_dim, CLASSES)
  
        self.act = nn.Tanh()  

        
    def forward(self, fwd_states, bwd_states):
        concat_states = th.cat((fwd_states, bwd_states), -1)
        comb_embed    = self.act(self.comb_embed(concat_states)) 

        pred0 = self.act(self.predict(comb_embed))
        pred1 = self.act(self.predict1(pred0))
        pred2 = self.act(self.predict2(th.add(pred1, comb_embed)))
        pred3 = self.act(self.predict3(pred2))
        pred4 = self.act(self.predict4(pred3))
        pred5 = self.predict5(th.add(pred4, pred2)) 
        pred6 = self.act(self.predict6(pred5))
        pred7 = self.act(self.predict7(pred6))
        pred = self.predict8(th.add(pred7, pred5))

        #pred0 = self.act(self.predict(comb_embed))
        #pred1 = self.act(self.predict1(pred0))
        #pred2 = self.act(self.predict2(th.add(pred1, comb_embed)))
        #pred3 = self.act(self.predict3(th.add(pred2, comb_embed)))
        #pred = self.predict4(th.add(pred3, comb_embed))

        return pred












