from sim.fp_funcs import * 
from common.otc import *

#
# Simulates statically defined program and returns scalar result
#   -instructions have format [id: Int, opcode: Int, left_src_id, right_src_id]
#   -write_result is a list of flags, indicating whether or not idx-aligned instruction is intermediate variable 
#   -inputs is list of same length as instructions, where non-constants have 'None' value 
#   -OTC is list of precisions index-aligned with instructions, to be used on the corresponding instruction
def sim_prog(insns, write_result, inputs, otc):

    results = [None for i in insns]
    for insn_idx in range(len(insns)):    
        insn      = insns[insn_idx]
        result_id = insn[0]
        func_type = insn[1]
        is_const  = True if func_type == 0 else False      
        precision = otc[insn_idx]

        if (is_const):
            result = p_functions[func_type](inputs[insn_idx], precision)           
            results[result_id] = {'val':result,
                                  'prec':precision}
        else:
            l_operand = results[insn[2]]['val']
            precision = results[insn[2]]['prec']
            r_operand = None 

            if not(insn[3] is None):              
                r_operand = results[insn[3]]['val']          
                precision = max(precision, results[insn[3]]['prec']) 

            #transcendental
            else:
                precision = otc[insn_idx]                   
            result = p_functions[func_type](l_operand, r_operand, precision)          

            #set precision if variable
            if (write_result[insn_idx]):
                precision = otc[insn_idx]
            if result is None:
                return None
            else:
                results[result_id] = {'val': result,
                                      'prec': precision} 
            
    # program result is in graph drain
    return results[-1]['val']

