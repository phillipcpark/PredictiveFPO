from fp_funcs import * 

# 
def sim_prog(insns, inputs, otc):

    results = [None for i in insns]
    for insn_idx in range(len(insns)):    
        insn      = insns[insn_idx]
        result_id = insn[0]
        func_type = insn[1]
        is_const  = True if func_type == 0 else False      
        precision = otc[insn_idx]

        if (is_const):
            result = p_functions[func_type](inputs[insn_idx], precision)           
            results[result_id] = result
        else:
            l_operand = results[insn[2]]
            r_operand = results[insn[3]] if not(insn[3] is None) else None               
                              
            result = p_functions[func_type](l_operand, r_operand, precision)          

            if result is None:
                return None
            else:
                results[result_id] = result

    # program result is in graph drain
    return float(results[-1])
