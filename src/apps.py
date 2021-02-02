from nn_model import *
from params import *
from prog_inputs import *
from otc import *
from eval_metrics import *
from ds_gen import *

from mpmath import mp

import random as rand

# <= -1.57079632679 x 1.57079632679))    
#x - (((x * x) * x) / 6.0) 
#+ (((((x * x) * x) * x) * x) / 120.0) 
#- (((((((x * x) * x) * x) * x) * x) * x) / 5040.0);

# 0,  x
# 1,  6.0
# 2,  120.0
# 3,  5040.0

# 4, MUL, 0, 0
# 5, MUL, 4, 0
# 6, DIV, 5, 1
# 7, SUB, 0, 6

# 8, MUL, 0, 0
# 9, MUL, 8, 0
# 10, MUL, 9, 0
# 11, MUL, 10, 0
# 12, DIV, 11, 2
# 13, ADD, 7, 12

# 14, MUL, 0, 0
# 15, MUL, 14, 0
# 16, MUL, 15, 0
# 17, MUL, 16, 0
# 18, MUL, 17, 0
# 19, MUL, 18, 0
# 20, DIV, 19, 3
# 21, SUB, 13, 20

#  err_thresh for SP failure: 0.0000000001

def sine_app():
    edges = [ [0, 4], [0, 4], [4, 5], [0,5], [5,6], [1,6], [0,7], [6,7],\
              [0,8], [0,8], [8,9], [0,9], [9,10], [0,10], [10,11], [0,11], [11,12], [2,12],\
              [7,13], [12,13], [0,14], [0,14], [14,15], [0,15], [15,16], [0,16], [16,17], [0,17],\
              [17,18], [0,18], [18,19], [0,19], [19,20], [3,20], [13,21], [20,21]] 


    n_ops = [0, 0, 0, 0, 3, 3, 4, 2, 3, 3, 3, 3, 4, 1, \
             3, 3, 3, 3, 3, 3, 4, 2]

    consts = [None for i in range(len(n_ops))]
    consts[1] = 6.0
    consts[2] = 120.0
    consts[3] = 5040.0

    unary_masks = [False for op in n_ops] 
  
    return edges, n_ops, unary_masks, consts

def gen_sine_inputs(consts, samp_sz):
    inputs = []

    for samp in range(samp_sz):
        x1 = rand.uniform(-1.57079632679, 1.57079632679) 
        curr_inputs = [c for c in consts]
        curr_inputs[0] = x1
        inputs.append(curr_inputs)
    return inputs 




def jet_app():
   
    edges = [ 
             [ 0, 3 ],[ 0, 3],[ 3, 4],[ 2, 4],[ 5, 6],[ 0, 6],[ 6, 7],[ 0, 7], 
             [ 8, 9], [ 1, 9], [ 7, 10 ],[ 9, 10 ],[ 10, 11 ],[ 0, 11 ],[ 12, 14 ],
             [ 0, 14 ],[ 14, 15 ],[ 0, 15 ] ,[ 13, 16 ],[ 1, 16 ],[ 15, 17 ], [ 16, 17 ],
             [ 17, 18 ],[ 0, 18 ],[ 11, 19 ],[ 4, 19 ] ,[ 18, 20 ],[ 4, 20 ],[ 21, 22 ],
             [ 0, 22 ] ,[ 22, 23 ] ,[ 19, 23 ] ,[ 19, 25 ],[ 24, 25 ],[ 23, 26 ],[ 25, 26 ], 
             [ 27, 28 ],[ 19, 28 ],[ 28, 30 ],[ 29, 30 ],[ 0, 31 ],[ 0, 31 ],[ 31, 32 ],
             [ 30, 32 ],[ 26, 33 ],[ 32, 33 ],[ 33, 34 ],[ 4, 34 ],[ 35, 36 ],[ 0, 36 ],
             [ 36, 37 ], [ 0, 37 ],[ 37, 38 ],[ 19, 38 ],[ 34, 39 ],[ 38, 39 ],[ 0, 40 ],
             [ 0, 40 ],[ 40, 41 ],[ 0, 41 ],[ 39, 42 ],[ 41, 42 ],[ 42, 43 ],[ 0, 43 ],
             [ 44, 45 ],[ 20, 45 ],[ 43, 46 ],[ 45, 46 ],[ 0, 47 ],[ 46, 47 ] ]

    
    n_ops = [
             0,0,0,3,1,0,3,3,0,3,1,2,0,0,3,3,3,2,2,4,4,0,3,3,
             0,2,3,0,3,0,2,3,3,1,3,0,3,3,3,1,3,3,1,1,0,3,1,1 ]


    consts = [None, None, 1.0, None, None, 3.0, None, None, 2.0, None, None, None, 3.0, 2.0,\
              None, None, None, None, None, None, None, 2.0, None, None, 3.0, None, None, 4.0,\
              None, 6.0, None, None, None, None, None, 3.0, None, None, None, None, None, None, None,\
              None,  3.0, None, None, None]

    unary_masks = [False for op in n_ops] 
  
    return edges, n_ops, unary_masks, consts

# x1 in [-5,5], x2 in [-20,5]
def gen_jet_inputs(consts, samp_sz):
    inputs = []

    for samp in range(samp_sz):
        x1 = rand.uniform(-5.0, 5.0)
        x2 = rand.uniform(-20.0, 5.0)
        curr_inputs = [c for c in consts]
        curr_inputs[0] = x1
        curr_inputs[1] = x2
        inputs.append(curr_inputs)
    return inputs 
    

# kepler1
#    4 vars in [4, 159/25]
# err_thresh for SP failure 0.000000001

def kep1_app():
    edges = [[0,5], [3,5], [4,6], [0,6], [1,7], [6,7], [7,8], [2,8], [8,9], [3,9], [5,10], [9,10],\
             [0,11], [1,11], [11,12], [2,12], [12,13], [3,13], [1,14], [13,14], [10,15], [14,15],\
             [0,16], [1,16], [16,17], [2,17], [17,18], [3,18], [2,19], [18,19], [15,20], [19,20],\
             [1,21], [2,21], [21,22], [3,22], [20,23], [22,23], [0,24], [2,24], [23,25], [24,25],\
             [0,26], [1,26], [25,27], [26,27], [27,28], [3,28]]

    
    n_ops = [0, 0, 0, 0, 0, 3, 3, 1, 1, 2, 3, 2, 1, 1, 3, 1,\
             1, 2, 1, 3, 1, 3, 3, 2, 3, 2, 3, 2, 2] 

    consts = [None for i in range(len(n_ops))]
    consts[4] = -1.0
  
    unary_masks = [False for i in range(len(n_ops))]

    return edges, n_ops, unary_masks, consts 

# 
def gen_kep1_inputs(consts, samp_sz):
    inputs = []

    for samp in range(samp_sz):
        x1 = rand.uniform(4.0, 159.0/25.0)
        x2 = rand.uniform(4.0, 159.0/25.0)
        x3 = rand.uniform(4.0, 159.0/25.0)
        x4 = rand.uniform(4.0, 159.0/25.0)

        curr_inputs = [c for c in consts]
        curr_inputs[0] = x1
        curr_inputs[1] = x2
        curr_inputs[2] = x3
        curr_inputs[3] = x4

        inputs.append(curr_inputs)
    return inputs 


# kepler2
#    4 vars in [4, 159/25]
# err thresh for SP failure: 0.000000001

def kep2_app():
    edges = [ [0, 7], [3,7], [6,8], [0,8], [8,9],[1,9], [9,10], [2,10], [10, 11], [3, 11], [11, 12],\
              [4, 12], [12, 13], [5, 13], [7,14], [13, 14], [1, 15], [4, 15], [0,16], [1, 16], [16, 17], [2, 17], \
              [17, 18], [3,18], [18,19], [4,19], [19,20], [5,20], [15,21], [20,21], [14,22], [21,22], \
              [2,23], [5,23], [0,24], [1,24], [24,25], [2,25], [25,26], [3,26], [26,27], [4,27], [27, 28], [5,28], [23,29], [28,29],\
              [22,30], [29,30], [1, 31], [2, 31], [31, 32], [3, 32], [30, 33], [32, 33], [0,34], [2,34], [34,35], [4,35], [33,36], [35,36], \
              [0, 37], [1, 37], [37, 38], [5, 38], [36, 39], [38,39], [3, 40], [4, 40], [40, 41], [5, 41], [39, 42], [41, 42]] 

    n_ops = [0, 0, 0, 0, 0, 0, 0, \
             3, 3, 1, 1, 2, 1, 1, 3, \
             3, 2, 1, 1, 2, 1, 3, 1, \
             3, 1, 2, 1, 1, 2, 3, 1, \
             3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2] 

    consts = [None for i in range(len(n_ops))]
    consts[6] = -1.0
  
    unary_masks = [False for i in range(len(n_ops))]

    return edges, n_ops, unary_masks, consts 

# 
def gen_kep2_inputs(consts, samp_sz):
    inputs = []

    for samp in range(samp_sz):
        x1 = rand.uniform(4.0, 159.0/25.0)
        x2 = rand.uniform(4.0, 159.0/25.0)
        x3 = rand.uniform(4.0, 159.0/25.0)
        x4 = rand.uniform(4.0, 159.0/25.0)
        x5 = rand.uniform(4.0, 159.0/25.0)
        x6 = rand.uniform(4.0, 159.0/25.0)

        curr_inputs = [c for c in consts]
        curr_inputs[0] = x1
        curr_inputs[1] = x2
        curr_inputs[2] = x3
        curr_inputs[3] = x4
        curr_inputs[4] = x3
        curr_inputs[5] = x4

        inputs.append(curr_inputs)
    return inputs 



# b2 = b * b;
# b4 = b2 * b2;
# b6 = b4 * b2;
# b8 = b4 * b4;
# a2 = a * a;

# firstexpr = (((11.0 * a2) * b2) - (121.0 * b4)) - 2.0;

# (333.75 - a2) * b6 
# + (a2 * firstexpr) 
# + (5.5 * b8) 
# + (a / (2.0 * b));


#
def rump_app():

    return None






# FIXME 
# quick tb

#feat_dim = OP_ENC_NOPREC_DIM if SINGLE_GRAPH_TARG else OP_ENC_DIM
#gnn = bid_mpgnn(feat_dim, H_DIM, CLASSES)
#
#edges, feats, unary_masks, consts = jet_app()
#ex_graph = batch_graphs_from_idxs([0], [edges], [unary_masks], [0], [feats], use_gpu=False)        
#
#_, top_order = gnn(ex_graph, False)
#_exec_list    = []
#
#for n in range(len(feats)): 
#    parents = [e[0] for e in edges if e[1] == n]
#    if (len(parents) < 2):
#        if (len(parents) < 1): 
#            parents.append(None) 
#        parents.append(None) 
#    _exec_list.append([int(n), feats[n], parents[0], parents[1]])
#
#exec_list = [None for i in range(len(_exec_list))]
#for insn in _exec_list:
#    exec_list[insn[0]] = insn
#
#
#inputs = gen_jet_inputs(consts, 100000) 
#
#gt_otc = gen_spec_otc(exec_list, precs_inv[2])
#sp_otc = gen_spec_otc(exec_list, precs_inv[0])
#
#
#
##FIXME
#print_for_gviz(ex_graph.to_networkx(), exec_list, gt_otc)
#sys.exit(0)
#
#
#ex_errs = []
#
#for ins in inputs:
#    result      = sim_prog(exec_list, ins, sp_otc)
#    shad_result = sim_prog(exec_list, ins, gt_otc) 
#    err = relative_error(result, shad_result)
#  
#    mp.prec = 80
#    err = mp.mpf( abs((result - shad_result) / shad_result ))
#    ex_errs.append(err)
#
#accept, gt_thresh_prop = accept_err(ex_errs)
#
#print("\naccept, prop>thresh: " + str(accept) + " " + str(gt_thresh_prop))
#print("max_err: " + str(np.amax(ex_errs)))




