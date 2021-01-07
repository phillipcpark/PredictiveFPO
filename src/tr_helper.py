from random import shuffle
from random import choice
import numpy as np
from numpy import random as rand
import networkx as netwx
import sys
import itertools as it
import re
import copy
import time
import math
import csv

from mpmath import mp
from fp_funcs import *

# samples seperately at each magnitude
def gen_stratified_inputs(prog, samp_sz, max_mag):
   inputs  = []

   samps_per_mag = int(samp_sz / (max_mag * 2))

   for mag in range(-max_mag + 1, max_mag):

       for samp in range(samps_per_mag):
           samp_inputs = []

           for insn in prog:                            
               is_const = True if insn[1] == 0 else False 
               if (is_const):
                   samp_inputs.append(rand.uniform(-1.0 * (10.0**mag), 10.0**mag))
               else:
                   samp_inputs.append(0.0)
           inputs.append(samp_inputs)

   return inputs   
        
   
#
def gen_spec_otc(prog, prec):
    otc = [prec for insn in prog]
    return otc


# 
def sim_prog(insns, inputs, otc):
    results = [None for i in insns]
    for insn_idx in range(len(insns)):    
        insn      = insns[insn_idx]
        node_id   = insn[0]
        func_type = insn[1]
        is_const  = True if func_type == 0 else False      
        precision = otc[insn_idx]

        if (is_const):
            result = p_functions[func_type](inputs[insn_idx], precision)           
            results[node_id] = result
        else:
            l_operand = results[insn[2]]
            r_operand = results[insn[3]] if not(insn[3] is None) else None               
                              
            result = p_functions[func_type](l_operand, r_operand, precision)          

            if result is None:
                return None
            else:
                results[node_id] = result

    # program result is in graph drain
    return results[-1]

# 
def relative_error(val_num, val_denom):       
    err = None
    try:
        err = abs(val_num - val_denom) / abs(val_denom)
    except:
        err = 0.0
    return err


#
def sort_otcs_by_score(otcs):
    scores = [np.sum(otc) for otc in otcs]
    sort_idxs = np.argsort(scores)
    sorted_otcs = [otcs[idx] for idx in sort_idxs]
    
    return sorted_otcs




                      




















