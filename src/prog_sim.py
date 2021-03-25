from fp_funcs import * 
from params import *

from otc import *

# 
def sim_prog(insns, write_result, inputs, otc):

    results = [None for i in insns]
    for insn_idx in range(len(insns)):    
        insn      = insns[insn_idx]
        result_id = insn[0]
        func_type = insn[1]
        is_const  = True if func_type == 0 else False      
        precision = otc[insn_idx]

        if (is_const):
            #print("const defined in " + str(precision))

            result = p_functions[func_type](inputs[insn_idx], precision)           
            results[result_id] = {'val':result,\
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

            #print("\texec in " + str(precision))
                   
            result = p_functions[func_type](l_operand, r_operand, precision)          

            #set precision if variable
            if (write_result[insn_idx]):
                precision = otc[insn_idx]
                #print("writing in " + str(precision))

            if result is None:
                return None
            else:
                results[result_id] = {'val': result, \
                                      'prec': precision} 
            
    # program result is in graph drain
    return float(results[-1]['val'])


#exec_list = [
#[0,0,None,None], #32  
#[1,0,None,None], #64
#[2,0,None,None], #64
#[3,1, 2,1],      #64, 32 
#[4,3, 3,0],      #32, 32
#[5,4, 3,4],      #32, 64
#[6,1, 4,5],      #64
#[7,2, 5,6],      #64
#[8,1, 4,7],      #64
#[9,1, 5,8]       #64
#]
#
#
#counts = [0 for i in range(len(exec_list))]
#for e in exec_list:
#    if not(e[2] == None):
#        counts[e[2]] += 1
#    if not(e[3] == None):
#        counts[e[3]] += 1
#
#write_result = []
#for i in range(len(exec_list)):
#    if (counts[i] > 1):
#        write_result.append(True)
#    else:
#        write_result.append(False) 
#
#otc1 = [32,32,32,32,64,64, -1, -1, -1, -1]
#otc2 = [64,64,64,32,32,32, -1, -1, -1, -1]
#
#_sorted = sort_otcs_by_score([otc1, otc2], exec_list, write_result)
#print(_sorted)



#result_qp = sim_prog(exec_list, write_result, [9483094039493.23, 0.1394083493821723, 0.4833940823423], [80 for i in range(len(exec_list))]) 
#result = sim_prog(exec_list, write_result, [9483094039493.23, 0.1394083493821723, 0.4833940823423], [32,64,64,32,32,64, 80, 80, 80, 80])
#mp.prec=64
#print("\n" + str(abs((result_qp - result)/result_qp)))




