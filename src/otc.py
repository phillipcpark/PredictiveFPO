from params import *

#
def is_const(opcode):
    if (opcode == ops['CONST']):
        return True
    return False

#
def is_unary(opcode):
    if (opcode == ops['SIN'] or opcode == ops['COS']):
        return True
    return False

#
def tune_prec(orig, tune_rec):
    rec = tune_rec - int((CLASSES-1)/2)

    if rec < 1:
        return max(0, orig+rec)
    else:
        return min(2, orig+rec) 

#
def gen_spec_otc(prog, prec):
    otc = [prec for insn in prog]
    return otc

#
def gen_rand_otc(prog):
    otc = [rand.choice(precisions, p=p_precisions) for insn in prog] 
    return otc

#
def sort_otcs_by_score(otcs):
    scores = [np.sum(otc) for otc in otcs]
    sort_idxs = np.argsort(scores)
    sorted_otcs = [otcs[idx] for idx in sort_idxs]
    
    return sorted_otcs

#
def are_same_otcs(otc1, otc2):

    # FIXME should have better handling, so None is never passed
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

# gets the # of types between solution and initial
def otc_dist(sol_otc, init_otc, shift=0):
    dists = []
    for insn_idx in range(len(init_otc)):
        dists.append(sol_otc[insn_idx] - init_otc[insn_idx] + shift)
    return dists



