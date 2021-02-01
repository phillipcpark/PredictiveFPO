from nn_model import *
from params import *
from prog_inputs import *
from otc import *
from eval_metrics import *
from ds_gen import *

from mpmath import mp

import random as rand

# 0, 0,,
# 1, 0,,
# 2, 0,,      (1)
# 3, 3, 0, 0
# 4, 1, 3, 2   
# 5, 0,,      (3) 
# 6, 3, 5, 0
# 7, 3, 6, 0
# 8, 0,,      (2)
# 9, 3, 8, 1
# 10, 1, 7, 9
# 11, 2, 10, 0   
# 12, 0,,      (3)
# 13, 0,,      (2)
# 14, 3, 12, 0
# 15, 3, 14, 0
# 16, 3, 13, 1
# 17, 2, 15, 16
# 18, 2, 17, 0   
# 19, 4, 11, 4 
# 20, 4, 18, 4 
# 21, 0,       (2)
# 22, 3, 21, 0
# 23, 3, 22, 19
# 24, 0,        (3)
# 25, 2, 19, 24
# 26, 3, 23, 25
# 27, 0,        (4)
# 28, 3, 27, 19
# 29, 0,        (6)
# 30, 2, 28, 29
# 31, 3, 0,  0
# 32, 3, 31, 30
# 33, 1, 26, 32
# 34, 3, 33, 4
# 35, 0,        (3)
# 36, 3, 35, 0
# 37, 3, 36, 0
# 38, 3, 37, 19
# 39, 1, 34, 38
# 40, 3, 0,  0
# 41, 3, 40, 0
# 42, 1, 39, 41
# 43, 1, 42, 0
# 44, 0,        (3)
# 45, 3, 44, 20
# 46, 1, 43, 45
# 47, 1, 0,  46

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

#    x1 * x4 
#  * (((-x1 + x2) + x3) - x4) 
#  + (x2 * (((x1 - x2) + x3) + x4))
#  + (x3 * (((x1 + x2) - x3) + x4)) 
#  - ((x2 * x3) * x4) 
#  - (x1 * x3)
#  - (x1 * x2)
#  - x4;

#0, 0        #x1      
#1, 0        #x2      
#2, 0        #x3
#3, 0        #x4      
#4, 0        # -1.0
#
#5, MUL, 0, 3
#
#6, MUL, 4, 0     *
#7, ADD, 6, 1     *
#8, ADD, 7, 2     *
#9, SUB, 8, 3   
#10, MUL, 5, 9
#
#11, SUB, 0, 1    *
#12, ADD, 11, 2  
#13, ADD, 12, 3  
#14, MUL, 1, 13
#15, ADD, 10, 14  
#
#16, ADD, 0, 1    *
#17, SUB, 16, 2  
#18, ADD, 17, 3  
#19, MUL, 2, 18
#20, ADD, 15, 19
#
#21, MUL, 1, 2    *
#22, MUL, 21, 3
#23, SUB, 20, 22  *
#
#24, MUL, 0, 2    *
#25, SUB, 23, 24  
#
#26, MUL, 0, 1    *
#27, SUB, 25, 26
#
#28, SUB, 27, 3 


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
#edges, feats, unary_masks, consts = kep1_app()
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
#inputs = gen_kep1_inputs(consts, input_samp_sz) 
#
#gt_otc = gen_spec_otc(exec_list, precs_inv[2])
#sp_otc = gen_spec_otc(exec_list, precs_inv[0]) #precs_inv[0])
#
#
#
##FIXME
##print_for_gviz(ex_graph.to_networkx(), exec_list, gt_otc)
##sys.exit(0)
#
#
#ex_errs = []
#
#for ins in inputs:
#    result      = sim_prog(exec_list, ins, sp_otc)
#    shad_result = sim_prog(exec_list, ins, gt_otc) 
#    err = relative_error(result, shad_result)
#
#    print(err)
#   
#    mp.prec = 80
#    err = mp.mpf( abs((result - shad_result) / shad_result ))
#    ex_errs.append(err)
#
#    print(err)
#    sys.exit(0)
#
#
#accept, gt_thresh_prop = accept_err(ex_errs)
#
#print("\naccept, prop>thresh: " + str(accept) + " " + str(gt_thresh_prop))
#print("max_err: " + str(np.amax(ex_errs)))
#
#
#
#
