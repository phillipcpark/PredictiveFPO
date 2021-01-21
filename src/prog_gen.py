from random import shuffle
from random import choice
import numpy as np
import networkx as netwx
import sys
import itertools as it
import re
import copy
import time
import math
import csv

from fp_funcs import *
from prog_inputs import *
from otc import *
from prog_sim import *
from eval_metrics import *

from numpy import random as rand

search_steps = 1 #for progressive tuning
input_samp_sz = 5000    
otc_samp_sz = 200

inputs_mag = 3 #6

precisions = [32, 64, 80] 
prec_map ={32:0,
           64:1,
           80:2}

p_precisions = [0.35, 0.55, 0.1] #[0.45, 0.45, 0.1]  #[0.6, 0.3, 0.1]
p_tune_max = 0.5
p_tune_min = 0.5

err_thresh = 0.0000001 #0.01

is_binary = {'ADD': True,
             'SUB': True,
             'MUL': True,
             'DIV': True,
             'SIN': False,
             'COS': False}

soft_constraints = {'max_edges':30, 'max_out_degree':3, 'max_consts':4}
#soft_constraints = {'max_edges':25, 'max_out_degree':3, 'max_consts':4}


op_types    = ['ADD', 'SUB', 'MUL', 'DIV', 'SIN', 'COS']
dir_p_ops   = [10.0, 10.0, 10.0, 10.0, 1.0, 1.0]     

edge_types  = ['op_new', 'op_exist', 'const_new', 'const_exist']
dir_p_edges = [10.0, 1.0, 0.1, 0.1] 

#dir_p_edges = [10.0, 1.0, 0.1, 0.1] 
#dir_p_edges = [12.0, 6.0, 1.0, 0.5] 
#dir_p_edges = [12.0, 1.0, 0.5, 0.1] 


#NOTE by the backward generation process, the maximum path between any two nodes sharing an edge is often small...

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


# FIXME FIXME FIXME are discrepancies in list ordering causing poor performance?
def gen_exec_list(prog_g):

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
                
    exec_list = gen_exec_list(prog_g)
          
    return exec_list, prog_g

          
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
def progress_tuneup_strat(exec_trace, inputs, max_prec_otc, steps):

    # start with all lowest precision
    cand_otcs   = [ gen_spec_otc(exec_trace,precisions[0]) ] 

    shad_results = [sim_prog(exec_trace, _input, max_prec_otc) for _input in inputs]
    for result in shad_results:
        if (result == None):
            print("\t****shadow caused FP exception; no solution for program")
            return False

    optimal_otc= None
    best_err   = 1.0

    step = 0
    while (optimal_otc is None and step < steps):     
        print("\n**eval " + str(len(cand_otcs)) + " candidates")        
        for otc in cand_otcs: 
            invalid = False     

            for input_idx in range(len(inputs)): 
                result_cand = sim_prog(exec_trace, inputs[input_idx], otc) 

                if result_cand == None:
                    invalid = True
                    break
                error = abs(relative_error(result_cand, shad_results[input_idx]))

                if error > err_thresh:                                    
                    invalid = True
                    break

            if (invalid):
                continue  

            # valid, is optimal otc
            optimal_otc = otc
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
def rand_subset_strat(exec_trace, inputs, max_prec_otc, samp_sz):
    print("\tgen rand cand otcs")
    cand_otcs   = [ gen_rand_otc(exec_trace) for samp in range(samp_sz) ] 
     

    print("\tsorting cand otcs by scores")
    cand_otcs   = sort_otcs_by_score(cand_otcs)

    print("\tgen shadow results")
 
    shad_results = [sim_prog(exec_trace, _input, max_prec_otc) for _input in inputs]

    for result in shad_results:
        if (result == None):
            print("\n****shadow caused FP exception; no soltion for program")
            return False

    print("\tsearching over candidates")
  
    optimal_otc = None
    for otc_idx in range(len(cand_otcs)): 
        if (otc_idx % 50 == 0):
            print("\tsearching otc " + str(otc_idx))

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

    #
    # keep tuning down until no new solution
    #
    if not (optimal_otc is None):
        print("\tinitial sol")
        print(np.sum(optimal_otc))

        last_sol = optimal_otc                  
        cands   = expand_otcs_down([last_sol], 1.0)    
        phase   = 0 

        while(len(cands) > 0): 
            curr_sol = None    

            cands = sort_otcs_by_score(cands) 
            print("\teval " + str(len(cands)) + " improved solutions")
             
            for cand_idx in range(len(cands)):
                cand = cands[cand_idx]
                viable = True 

                for input_idx in range(len(inputs)):               
                    result_cand = sim_prog(exec_trace, inputs[input_idx], cand) 
    
                    if result_cand == None:
                        viable = False
                        break

                    error = abs(relative_error(result_cand, shad_results[input_idx]))

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
            print("\t**optimal " + str(optimal_otc))    

    return optimal_otc

#
def hybrid_strat(exec_trace, inputs, max_prec_otc):

    # check if trivial solution
    sol = progress_tuneup_strat(exec_trace, inputs, max_prec_otc, 1)

    if not(sol is None):      
        print("\t**solution was trivial, skipping program")
        return None 

    sol = rand_subset_strat(exec_trace, inputs, max_prec_otc, otc_samp_sz)
    
    return sol

#
def search_opt_otc(exec_trace, samplers):     
    # generate inputs/precision and perform search 
    #inputs = gen_inputs(exec_trace, input_samp_sz, inputs_bound_l, inputs_bound_h)
    inputs = gen_stratified_inputs(exec_trace, input_samp_sz, inputs_mag)

    max_prec_otc = gen_spec_otc(exec_trace,precisions[-1])

    #sol = rand_subset_strat(exec_trace, inputs, max_prec_otc, otc_samp_sz)
    #sol = progress_tuneup_strat(exec_trace, inputs, max_prec_otc, search_steps)
    sol = hybrid_strat(exec_trace, inputs, max_prec_otc)

    if sol is None:
        print("\t\t**failed to find sol")
        return None, None

    print("\t\t**found optimal")
    print(str(np.sum(sol)))

    return sol, inputs



#
def map_precisions(otc):
    mapped = []
    for prec in otc:
        mapped.append(prec_map[prec])
    return mapped





                      
#    
if __name__ == '__main__':

    p_edges     = rand.dirichlet(dir_p_edges)
    p_ops       = rand.dirichlet(dir_p_ops)   
        
    samp_edge = cat_sampler(p_edges, edge_types)
    samp_op   = cat_sampler(p_ops, op_types)
    const_gen = const_generator()
    samplers = {'edge':samp_edge, 'op':samp_op, 'const': const_gen}

    start_t = time.time()
    prog_count = 1000

    exec_traces = []
    solutions   = []
    inputs      = []

    for i in range(prog_count):
        print("\n******\n**prog\n****** " + str(i))
        sol_otc = None
        candidates = None

        exec_trace = None
        sol_otc = None
        samp_inputs  = None

        while (sol_otc is None):
            exec_trace, prog_g  = gen_prog(samplers, soft_constraints)        
            sol_otc, samp_inputs = search_opt_otc(exec_trace, samplers)

        if not(np.sum(sol_otc) - 0.0 > np.finfo(np.float64).eps):
            err_hand = None
            try:
                err_hand = open("err_log.txt", 'a')
                err_hand.write("all-SP generator error")
            except:
                err_hand = open("err_log.txt", 'w')

            err_hand.write("all-SP generator error")                
            err_hand.close() 
            print("\t**sol otc was, SOMEHOW, all-sp")    
            continue

        #FIXME
        #print_for_gviz(prog_g, exec_trace, sol_otc)

        exec_traces.append(exec_trace)
        solutions.append(sol_otc)
        inputs.append(samp_inputs)         
 
    end_t = time.time()
            
    print("\n** " + str(prog_count) + " done in " + str(end_t - start_t)) 
                    
    #number of a priori input OTCs are generated
    feats_per_prog = 50 
    all_feats = []
    all_labels = [] 

    for prog_idx in range(len(exec_traces)):    
        print("gen feats for prog " + str(prog_idx))

        feats, labels = gen_ds(exec_traces[prog_idx], solutions[prog_idx], inputs[prog_idx], feats_per_prog) 
        all_feats.append(feats)
        all_labels.append(labels)

    path = sys.argv[1]
    emit_ds(path, exec_traces, all_feats, all_labels, feats_per_prog)




















