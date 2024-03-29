from numpy import random as rand, argsort
from copy import deepcopy

# parameters and encodings of operations and precision
CONST_PREC = 64
PRECS      = {32:0, 64:1, 80:2}
PRECS_INV  = {0:32, 1:64, 2:80}

OPS = {'CONST':0,
       'ADD':1, 
       'SUB':2, 
       'MUL':3, 
       'DIV':4, 
       'SIN':5, 
       'COS':6,
       'TAN':7,
       'ASIN':8,
       'ACOS':9,
       'ATAN':10,
       'SQRT':11,
       'POW':12 } 

# 1-hot vector encoding for model 
OP_ENC_DIM = len(OPS.keys())*len(PRECS.keys()) 
OP_ENC     = {} 

curr_enc = [1.0] + [0.0 for i in range(OP_ENC_DIM-1)]
for op in OPS.keys():
    curr_op_encs = {} 

    for prec in PRECS.keys():
        curr_op_encs[PRECS[prec]] = deepcopy(curr_enc) 
        curr_enc.insert(0, 0.0)
        curr_enc.pop()
    OP_ENC[OPS[op]] = curr_op_encs


#
# helpers for checking, comparing, and assessing operation type configurations,
# as well as their corresponding operations
#

#
def is_const(opcode):
    if (opcode == OPS['CONST']):
        return True
    return False

#
def is_unary(opcode):
    if (opcode >= OPS['SIN']):
        return True
    return False

#
def is_func(opcode):
    if (opcode >= OPS['SIN']):
        return True
    else:
        return False 

#
def tune_prec(orig, tune_rec):
    rec = None

    if (tune_rec == 1):
        rec = -2  
    else:
        rec = 0
    if rec < 1:
        return max(0, orig+rec)
    else:
        return min(2, orig+rec) 

#
def gen_spec_otc(prog, prec):
    otc = [prec for insn in prog]
    return otc

#
def gen_rand_otc(prog, precs, p_precs):
    otc = [rand.choice(precs, p=p_precs) for insn in prog] 
    return otc

# scoring is based on number of 32-bit operations
def sort_otcs_by_score(otcs, exec_trace, write_result):
    scores = []

    for otc in otcs:
        score = 0
        for nidx in range(len(exec_trace)):
            if (exec_trace[nidx][1]==0 or write_result[nidx]):
                score += otc[nidx]
        scores.append(score) 

    sort_idxs = argsort(scores)
    sorted_otcs = [otcs[idx] for idx in sort_idxs]
    
    return sorted_otcs

#
def are_same_otcs(otc1, otc2):
    if (otc1 is None or otc2 is None):
        return False
   
    otc1_sz = len(otc1)
    otc2_sz = len(otc2)
    if not(otc1_sz == otc2_sz):
        return False                

    for i in range(otc1_sz):
        if not(otc1[i] == otc2[i]):
            return False
    return True 

# of types between solution and initial
def otc_dist(sol_otc, init_otc, shift=0):
    dists = []
    for insn_idx in range(len(init_otc)):
        dists.append(sol_otc[insn_idx] - init_otc[insn_idx] + shift)
    return dists



