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

from fp_funcs import *

search_steps = 100
input_samp_sz = 10000    
otc_samp_sz = 1000

inputs_bound_l = -1e6
inputs_bound_h = 1e6 

precisions = [32, 64, 80] 
p_precisions = [0.6, 0.3, 0.1]
p_tune_max = 0.5
p_tune_min = 0.5

err_thresh = 0.001 #0.01

is_binary = {'ADD': True,
             'SUB': True,
             'MUL': True,
             'DIV': True,
             'SIN': False,
             'COS': False}

soft_constraints = {'max_edges':30, 'max_out_degree':3, 'max_consts':4}

edge_types  = ['op_new', 'op_exist', 'const_new', 'const_exist']
op_types    = ['ADD', 'SUB', 'MUL', 'DIV', 'SIN', 'COS']

dir_p_ops   = [10.0, 10.0, 10.0, 10.0, 2.0, 2.0]     

dir_p_edges = [12.0, 6.0, 1.0, 0.5] 

# NOTE good distribution!
#dir_p_edges = [12.0, 6.0, 1.0, 0.5] 


#NOTE by the backward generation process, the maximum path between any two nodes sharing an edge is small...

# categorical sampler
# sampling returns (sample_class, sample_id)
class cat_sampler:    
    def __init__(self, probs, classes):
        self.p = probs
        self.c = classes
        self.c_counts = {}
        for _c in classes:
            self.c_counts[_c] = 0        

    def __call__(self):
        samp = rand.choice(self.c, p = self.p)
        self.c_counts[samp] += 1
        return samp, samp + str(self.c_counts[samp])

# generates unique const ids
class const_generator:
    def __init__(self):
        self.c_counts = 0

    def __call__(self):
        const_id       = 'CONST'+str(self.c_counts) 
        self.c_counts += 1
        return 'CONST', const_id 

    def delete_const(self):
        self.c_counts -= 1

#
def update_parents_flag(targ, prog_g):
    parent_count = len(prog_g.in_edges(targ))

    if (parent_count==2 or (not(is_binary[prog_g.nodes[targ]['node_type']]) and parent_count==1)):
        prog_g.nodes()[targ]['has_parents'] = True                     
  

# sample new op and connect it to target
def conn_new_op(targ, g_sink_id, prog_g, nodes_by_attr, samplers, soft_constraints):
    src_type, src_id  = samplers['op']()
    nodes_by_attr['ops'].append(src_id)

    prog_g.add_node(src_id, node_type=src_type, has_parents=False)    
    prog_g.add_edge(src_id, targ)
    update_parents_flag(targ, prog_g)
    return True
   
#
def conn_new_const(targ, g_sink_id, prog_g, nodes_by_attr, samplers, soft_constraints):
    src_type, src_id  = samplers['const']()
    nodes_by_attr['consts'].append(src_id)

    prog_g.add_node(src_id, node_type=src_type)    
    prog_g.add_edge(src_id, targ)
    update_parents_flag(targ, prog_g)
    return True

 
# 
def conn_exist_op(targ, g_sink_id, prog_g, nodes_by_attr, samplers, soft_constraints):
    # candidates may not be on path from target->sink  
    targ_sink_paths = []

    # NOTE this path-finding routine is bottleneck...
    #  -decay certain edge probabilities with each gen step, so that this routine is not called as frequently
    #  -cache intermediate results?
    for path in netwx.all_simple_paths(prog_g, targ, g_sink_id):
        targ_sink_paths.append(path)          
    ts_paths_flat = []

    for path in targ_sink_paths:
        ts_paths_flat += path

    g_ops = nodes_by_attr['ops']   
    if (len(g_ops) < 1):
        return False
    cands = [op for op in g_ops if (not(op in ts_paths_flat) and not(op == targ))]
  
    # candidates may not have path to target (creates cycle)
    cands_filt = []
    for c in cands:
        cand_targ_paths = []
        for path in netwx.all_simple_paths(prog_g, c, targ):
            cand_targ_paths.append(path)
        if (len(cand_targ_paths) < 1 and \
            not(prog_g.has_edge(targ, c)) and \
            len([child for child in prog_g.successors(c)]) <= soft_constraints['max_out_degree']):
            cands_filt.append(c)

    if (len(cands_filt) < 1):
        return False 

    src_id   = rand.choice(cands_filt)
    prog_g.add_edge(src_id, targ)
    update_parents_flag(targ, prog_g)

    return True
          
# 
def conn_exist_const(targ, g_sink_id, prog_g, nodes_by_attr, samplers, soft_constraints): 
    g_consts = nodes_by_attr['consts']

    cands = [cand for cand in nodes_by_attr['consts'] ]

    #FIXME FIXME will loop indefinitely when all consts have max out_degree, and there are 'leftover' ops without full parents
    #cands = [cand for cand in nodes_by_attr['consts'] \
    #         if len([child for child in prog_g.successors(cand)]) <= soft_constraints['max_out_degree']]
   
    if (len(cands) < 1):
        return False
 
    src_id   = rand.choice(cands)
    prog_g.add_edge(src_id, targ)
    update_parents_flag(targ, prog_g)
    return True



# 
def conn_src(targ, g_sink_id, prog_g, nodes_by_attr, samplers, soft_constraints):
    while (True):
        src_type, _ = samplers['edge']()
        succ        = gen_src[src_type](targ, g_sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)      
        if (succ):
            return   

#
def print_for_gviz(edges):
    for e in edges:
        print(str(e[0]) + "->" + str(e[1]) + ";")


opcodes = {'CONST':0,
           'ADD':1,
           'SUB':2,
           'MUL':3,
           'DIV':4,
           'SIN':5,
           'COS':6 } 
#
def emit_binary_op(op_id, op, srcs):     
    if not (len(srcs) == 2):
        raise RuntimeError('binary op did not have two parents at emission') from None  

    operation = [op_id,opcodes[op],srcs[0],srcs[1]]
    return operation

#
def emit_unary_op(op_id, op, srcs):     
    if not(len(srcs) == 1):
        raise RuntimeError('unary op did not have one parent at emission') from None

    operation = [op_id, opcodes[op], srcs[0],None]
    return operation

#
def emit_const(op_id, op, srcs):
    operation = [op_id,opcodes[op],None,None]
    return operation

emit_op = {'CONST': emit_const,
           'ADD': emit_binary_op,
           'SUB': emit_binary_op,
           'MUL': emit_binary_op,
           'DIV': emit_binary_op,
           'SIN': emit_unary_op,
           'COS': emit_unary_op }

gen_src = {'op_new': conn_new_op,
           'op_exist': conn_exist_op,
           'const_new': conn_new_const,
           'const_exist': conn_exist_const}



#
def gen_prog(samplers, soft_constraints):
    prog_g        = netwx.DiGraph()
    nodes_by_attr = {'ops':[], 'consts':[]} 
 
    # gen graph sink
    sink_type, sink_id = samplers['op']()
    prog_g.add_node(sink_id, node_type=sink_type, has_parents=False)
    nodes_by_attr['ops'].append(sink_id)       
 
    # gen sink srcs  
    conn_src(sink_id, sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)         
    if is_binary[sink_type]:        
        conn_src(sink_id, sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)      
 
    prog_g.nodes[sink_id]['has_parents'] = True

    # 
    # gen rest of graph
    #
    for i in range(soft_constraints['max_edges']):
        #print("**gen step " + str(i))

        # select target        
        n_attr  = netwx.get_node_attributes(prog_g, 'has_parents') 
        cands   = [op_id for op_id in nodes_by_attr['ops'] if not(n_attr[op_id])]             

        # if prematurely ran out of candidates, generate one; remove const from existing op to create target       
        if (len(cands) < 1):
            targ_cands = []

            # idx-aligned with targ_cands; associated src const
            src_cands  = []

            for c in nodes_by_attr['consts']:
                c_children = prog_g.successors(c)
                for child in c_children:
                    targ_cands.append(child)
                    src_cands.append(c)

            # choose an existing op, to diconnect from const src
            targ_idx    = choice(np.arange(len(targ_cands)))                       
            remove_src  = src_cands[targ_idx]
            remove_targ = targ_cands[targ_idx]
            prog_g.remove_edge(remove_src, remove_targ)

            #FIXME FIXME need to update all existing const ids which were generated after deleted src                           
            succs = [child for child in prog_g.successors(remove_src)]
            #if (len(succs) < 1):
            #    samplers['const'].delete_const()

            # a candidate target now exists for connecting a new op                       
            conn_new_op(remove_targ, sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)           
            continue            

        targ_id = choice(cands) 

        # gen src
        conn_src(targ_id, sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)         

        #if is_binary[prog_g.nodes()[targ_id]['node_type']]:
        #    conn_src(targ_id, sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)        
        #prog_g.nodes[targ_id]['has_parents'] = True

    #
    # connect 'leftover' ops (without full set of parents) 
    #
    n_attrs = netwx.get_node_attributes(prog_g, 'has_parents')
    needs_parents = [op for op in nodes_by_attr['ops'] if not(n_attrs[op])] 
 
    max_consts = soft_constraints['max_consts']     
    for op in needs_parents:
        if len(nodes_by_attr['consts']) < max_consts:
            while not(prog_g.nodes[op]['has_parents']): 
                conn_new_const(op, sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)               
        else:
            while not(prog_g.nodes[op]['has_parents']):
                if not(conn_exist_op(op, sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)):                         
                    conn_exist_const(op, sink_id, prog_g, nodes_by_attr, samplers, soft_constraints)
                 
    # FIXME
    #print_for_gviz(prog_g.edges())

    # gen exec order from traversal
    visit_order = [node for node in netwx.topological_sort(prog_g)]    

    # change node_ids to visit_idxs for trace
    counter   = 0
    visit_idx = {}
    for node in visit_order:
        visit_idx[node] = counter
        counter       += 1   
 
    exec_list = []
   
    for node_idx in range(len(visit_order)):
        curr_node = visit_order[node_idx]
        node_type = prog_g.nodes()[curr_node]['node_type']

        parents     = [parent for parent in prog_g.predecessors(curr_node)]
        parent_idxs = [visit_idx[parent] for parent in parents]

        operation = emit_op[node_type](node_idx, node_type, parent_idxs) 
        exec_list.append(operation)           
  
    return exec_list



# should all inputs be kept in memory? If I want to share inputs for every OTC...
def gen_inputs(prog, samp_sz, bound_low, bound_high):
   inputs = []
   for samp in range(samp_sz):
       samp_inputs = []
       for insn in prog:
           is_const    = True if insn[1] == 0 else False 
           if (is_const):
               samp_inputs.append(rand.uniform(bound_low, bound_high))
           else:
               samp_inputs.append(0.0)
       inputs.append(samp_inputs)
   return inputs

# samples seperately at each magnitude
def gen_stratified_inputs(prog, samp_sz, bound_low, bound_high):
    inputs  = []
    max_mag = math.ceil(math.log10(bound_high)) 
    samps_per_mag = samp_sz / max_mag

    #FIXME FIXME FIXME
        

    


#
def gen_spec_otc(prog, prec):
    otc = [prec for insn in prog]
    return otc

#
def gen_rand_otc(prog):
    otc = [rand.choice(precisions, p=p_precisions) for insn in prog] 
    return otc


# 
def sim_prog(insns, inputs, otc):
    results = [None for i in insns]
    for insn_idx in range(len(insns)):    
        insn      = insns[insn_idx]
        func_type = insn[1]
        is_const  = True if func_type == 0 else False      
        precision = otc[insn_idx]

        if (is_const):
            result = p_functions[func_type](inputs[insn_idx], precision)           
            results[insn_idx] = result
        else:
            l_operand = results[insn[2]]
            r_operand = results[insn[3]] if not(insn[3] is None) else None               
                              
            result = p_functions[func_type](l_operand, r_operand, precision)          

            if result is None:
                return None
            else:
                results[insn_idx] = result




    # program result is in graph drain
    return results[-1]

# 
def relative_error(val_num, val_denom):       
    err = None
    try:
        err = abs(val_num - val_denom) / val_denom
    except:
        err = 0.0
    return err


# 
def tune_up(prec):
    tune_max = rand.choice([True, False], p=[p_tune_max, 1.0-p_tune_max])    

    if (tune_max or prec == precisions[1]):
        return precisions[-1]
    else:
        return precisions[1]

# 
def tune_down(prec):
    tune_min = rand.choice([True, False], p=[p_tune_min, 1.0-p_tune_min])    

    if (tune_min or prec == precisions[1]):
        return precisions[0]
    else:
        return precisions[1]



# for each candidate, expand by randomly up-tuning a random new node
def expand_otcs_up(otcs, gen_rate):
    exp_otcs = []

    for otc in otcs:
        tuneable_idxs = []

        for prec_idx in range(len(otc)):
            if not (otc[prec_idx] == precisions[-1]):  
                tuneable_idxs.append(prec_idx)
         
        for _idx in tuneable_idxs:
            if rand.choice([True, False], p=[gen_rate, 1.0-gen_rate]):
                _otc = list(copy.deepcopy(otc))
                _otc[_idx] = tune_up(_otc[_idx]) 
                exp_otcs.append(_otc)          
    expanded = list(np.unique(exp_otcs, axis=0))
    return expanded
 

#
def expand_otcs_down(otcs, gen_rate):
    exp_otcs = []

    for otc in otcs:
        tuneable_idxs = []

        for prec_idx in range(len(otc)):
            if not (otc[prec_idx] == precisions[0]):  
                tuneable_idxs.append(prec_idx)
         
        for _idx in tuneable_idxs:
            if rand.choice([True, False], p=[gen_rate, 1.0-gen_rate]):

                _otc = copy.deepcopy(otc)
                _otc[_idx] = tune_down(_otc[_idx]) 
                exp_otcs.append(_otc)          
    expanded = np.unique(exp_otcs, axis=0)
    return expanded


#
def sort_otcs_by_score(otcs):
    scores = [np.sum(otc) for otc in otcs]
    sort_idxs = np.argsort(scores)
    sorted_otcs = [otcs[idx] for idx in sort_idxs]
    
    return sorted_otcs

#
def progress_tuneup_strat(trace, inputs, max_prec_otc, steps):
    cand_otcs   = [ gen_spec_otc(exec_trace,precisions[0]) ] 

    shad_results = [sim_prog(exec_trace, _input, max_prec_otc) for _input in inputs]
    for result in shad_results:
        if (result == None):
            print("\n\n****shadow caused FP exception; no solution for program")
            return False

    optimal_otc= None
    best_err   = 1.0

    step = 0
    while (optimal_otc is None and step < steps):     
        print("\n**eval " + str(len(cand_otcs)) + " candidates")        
        for otc in cand_otcs: 
            invalid = False     
            errors  = 0.0

            for input_idx in range(len(inputs)): 
                result_cand = sim_prog(exec_trace, inputs[input_idx], otc) 

                if result_cand == None:
                    print("\t\tcandidate cause FP exception")
                    invalid = True
                    break
                error = abs(relative_error(result_cand, shad_results[input_idx]))
                errors += error

                if error > err_thresh:                 
                    invalid = True

                    if (not input_idx == 0):
                        avg_err = errors / float(input_idx)

                        if avg_err < best_err:
                            best_err = avg_err
                            print("best err " + str(best_err))

                    break
            if (invalid):
                continue  

            # valid, is optimal otc
            optimal_otc = otc
            print("\toptimal total, avg err " + str(errors) + ", " + str(errors/ float(len(inputs))))
            break

        if (optimal_otc is None and steps > 1):
            print("\t**expanding candidates")
       
            gen_rate = 1.0 / (step+1.01)**2
            print("\tgen_rate = " + str(gen_rate))  
 
            cand_otcs = expand_otcs_up(cand_otcs, gen_rate)

            if len(cand_otcs) == 0:
                print("\t**no candidates after expansion")
                return None
        step += 1
    return optimal_otc

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


#
def rand_subset_strat(exec_trace, inputs, max_prec_otc, samp_sz):
    print("\tgen rand cand otcs")
    cand_otcs   = [ gen_rand_otc(exec_trace) for samp in range(samp_sz) ] 

    print("\tsorting cand otcs by scores")
    cand_otcs   = sort_otcs_by_score(cand_otcs)

    print("\tgen shadow results")
    shad_results = [sim_prog(exec_trace, _input, max_prec_otc) for _input in inputs]

    for result in shad_results:
        if (result == None):
            print("\n\n****shadow caused FP exception; no soltion for program")
            return False

    print("\tsearching over candidates")
  
    optimal_otc = None
    for otc_idx in range(len(cand_otcs)): 
        if (otc_idx % 10 == 0):
            print(otc_idx)

        otc = cand_otcs[otc_idx]

        invalid = False     
        for input_idx in range(len(inputs)): 
            result_cand = sim_prog(exec_trace, inputs[input_idx], otc) 

            if result_cand == None:
                invalid = True
                break
            error = relative_error(result_cand, shad_results[input_idx])
            if error > err_thresh:                 
                invalid = True
                break
        if (invalid):
            continue  

        # valid, is optimal otc
        optimal_otc = otc
        break


    if not (optimal_otc is None):
        print("\ninitial sol")
        print(np.sum(optimal_otc))

        # keep tuning down until no new solution
        last_sol = optimal_otc                  
        cands   = expand_otcs_down([last_sol], 1.0)    
        phase   = 0 

        #FIXME FIXME              
        while(len(cands) > 0): 
            curr_sol = None    

            cands = sort_otcs_by_score(cands) 
            print("\n\teval " + str(len(cands)) + " improved solutions")
             
            for cand_idx in range(len(cands)):
                cand = cands[cand_idx]
                viable = True 

                for input_idx in range(len(inputs)):               
                    result_cand = sim_prog(exec_trace, inputs[input_idx], cand) 
    
                    if result_cand == None:
                        viable = False
                        break

                    error = relative_error(result_cand, shad_results[input_idx])

                    if error > err_thresh:                 
                        viable = False
                        break
                if (viable):                     
                    if are_same_otcs(curr_sol, cand):
                        break
                 
                    curr_sol  = copy.deepcopy(cand)
                    cands = expand_otcs_down([curr_sol], 1.0 - (1.0 / (1 + math.e**(phase+1))))
                    break                     

            if (curr_sol is None):
                break                        

            last_sol = curr_sol
            phase += 1            

        if not(last_sol is optimal_otc):
            optimal_otc = last_sol                
            print("\n\n**optimal " + str(optimal_otc))    

    return optimal_otc

#
def hybrid_strat(exec_trace, inputs, max_prec_otc):

    # check if trivial solution
    sol = progress_tuneup_strat(exec_trace, inputs, max_prec_otc, 1)

    if not(sol is None):      
        print("\n**solution was trivial, skipping program")
        return None 

    sol = rand_subset_strat(exec_trace, inputs, max_prec_otc, otc_samp_sz)
    return sol

#
def search_opt_otc(exec_trace, samplers):     
    # generate inputs/precision and perform search 
    inputs       = gen_inputs(exec_trace, input_samp_sz, inputs_bound_l, inputs_bound_h)
    max_prec_otc = gen_spec_otc(exec_trace,precisions[-1])

    #

    #sol = rand_subset_strat(exec_trace, inputs, max_prec_otc, otc_samp_sz)
    #sol = progress_tuneup_strat(exec_trace, inputs, max_prec_otc, search_steps)
    sol = hybrid_strat(exec_trace, inputs, max_prec_otc)

    if sol is None:
        print("\n**failed to find sol")
        return None

    print("\n\t**found optimal")
    print(str(np.sum(sol)))

    return sol

             
#    
if __name__ == '__main__':

    p_edges     = rand.dirichlet(dir_p_edges)
    p_ops       = rand.dirichlet(dir_p_ops)   
        
    samp_edge = cat_sampler(p_edges, edge_types)
    samp_op   = cat_sampler(p_ops, op_types)
    const_gen = const_generator()
    samplers = {'edge':samp_edge, 'op':samp_op, 'const': const_gen}

    start_t = time.time()
    prog_count = 1

    for i in range(prog_count):
        print("\n**prog " + str(i))
        feat = None
        while (feat is None):
            exec_trace = gen_prog(samplers, soft_constraints)        
            feat = search_opt_otc(exec_trace, samplers)
        
    end_t = time.time()

    print("\n** " + str(prog_count) + " done in " + str(end_t - start_t)) 
                 


    








