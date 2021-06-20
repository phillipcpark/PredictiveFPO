from config.params import *
from numpy import random as rand
import numpy as np

#
# helpers for checking, comparing, and assessing operation type configurations,
# as well as their corresponding operations
#

#
def is_const(opcode):
    if (opcode == ops['CONST']):
        return True
    return False

#
def is_unary(opcode):
    if (opcode >= ops['SIN']):
        return True
    return False

#
def is_func(opcode):
    if (opcode >= ops['SIN']):
        return True
    else:
        return False 

#
def tune_prec(orig, tune_rec):
    rec = None

    if (COARSE_TUNE):
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

    sort_idxs = np.argsort(scores)
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



