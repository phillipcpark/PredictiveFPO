from nn_model import *
from params import *
from prog_inputs import *
from otc import *
from eval_metrics import *

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

# TODO
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


    consts = [1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 6.0, 3.0, 3.0]

    unary_masks = [False for op in n_ops] 
  
    return edges, n_ops, unary_masks, consts


# FIXME 
# checking if jet works in SP
#feat_dim = OP_ENC_NOPREC_DIM if SINGLE_GRAPH_TARG else OP_ENC_DIM
#gnn = bid_mpgnn(feat_dim, H_DIM, CLASSES)
#
#edges, feats, unary_masks, consts = jet_app()
#ex_graph = batch_graphs_from_idxs([0], [edges], [unary_masks], [0], [feats])        
#
#_, top_order = gnn(ex_graph)
#exec_list    = []
#
#for step in top_order:
#    for n in step:
#        parents = [int(v) for v in ex_graph.in_edges(n)[0]]
#        if (len(parents) < 2):
#            if (len(parents) < 1): 
#                parents.append(None) 
#            parents.append(None) 
#        exec_list.append([int(n), feats[n], parents[0], parents[1]])
#
#inputs = gen_stratified_inputs(exec_list, input_samp_sz, inputs_mag) 
#
#gt_otc = gen_spec_otc(exec_list, precs_inv[2])
#sp_otc = gen_spec_otc(exec_list, precs_inv[1])
#
#ex_errs = []
#
#for ins in inputs:
#    result      = sim_prog(exec_list, ins, sp_otc)
#    shad_result = sim_prog(exec_list, ins, gt_otc) 
#    ex_errs.append(relative_error(result, shad_result))
#
#accept, gt_thresh_prop = accept_err(ex_errs)
#
#print("\naccept: " + str(accept))
#print("max_err: " + str(np.amax(ex_errs)))




