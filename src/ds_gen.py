from prog_gen import *
from otc import *
from random import choice

# generates sz number of input-output pairs from random a priori OTCs and tuning relative to solution   
def gen_ds(exec_trace, write_result, solution, inputs, sz):
    feats  = []
    labels = []
            
    gt_otc = gen_spec_otc(exec_trace, precisions[-1])  
    shad_results = []

    for ins in inputs:
        shad_results.append(sim_prog(exec_trace, write_result, ins, gt_otc)) 

    input_sz = len(inputs)        

    for i_sol in range(sz):
        if (i_sol % 20 == 0):
            print("\tcreating init sol feat " + str(i_sol))

        cand = None       
        valid = False

        while not(valid):
            #cand = gen_rand_otc(exec_trace, precisions, p_precisions)    
            cand = gen_spec_otc(exec_trace, precisions[1])   
 
            for i_idx in range(input_sz):
                result_cand = sim_prog(exec_trace, write_result, inputs[i_idx], cand) 
    
                if result_cand == None:
                    break
    
                error = abs(relative_error(result_cand, shad_results[i_idx]))    
                if error > err_thresh:                 
                    break
            valid = True

        feats.append(map_precisions(cand))
        labels.append(otc_dist(map_precisions(solution), map_precisions(cand), shift=len(precisions)-1))

    return feats, labels         


#
def emit_ds(path, traces, feats, labels, feats_per_prog, offset=0):
    ds_path = path + '/' + 'ds.csv' 
 
    with open(ds_path, 'w') as f_hand:
        f_writer = csv.writer(f_hand, delimiter=',')

        header = ['gid', 'nid','opcode','src_l','src_r','init_prec','tune_rec']
        f_writer.writerow(header)

        prog_count = len(traces) 

        for prog_idx in range(prog_count):             
            trace_len = len(traces[prog_idx])

            for otc_idx in range(feats_per_prog):
                for insn_idx in range(trace_len):                                       
                    f_writer.writerow([prog_idx+offset] + traces[prog_idx][insn_idx] + [feats[prog_idx][otc_idx][insn_idx]] + \
                                      [labels[prog_idx][otc_idx][insn_idx]])                
                f_writer.writerow([None for attrs in range(len(traces[prog_idx][0]) + 3)])     

#
def write_in_sets(path, in_sets):
    for in_set in range(len(in_sets)):
        ds_path = path + '/' + 'inputs_'+str(in_set) + '.csv'

        with open(ds_path, 'w') as f_hand:
            f_writer = csv.writer(f_hand, delimiter=',')
            for prog_ins in in_sets[in_set]:
                f_writer.writerow([val for val in prog_ins if not(val==None)])
        
                    

#
def print_for_gviz(exec_trace, write_result, precs):

    print("")
    for insn in exec_trace:
        print(insn)
    print("")

    for insn in exec_trace:
        if is_func(insn[1]):
            if (precs[insn[0]] == 32):
                print("\"" + str(insn[0]) + "\"" + " [style=filled,color=green];\n")
            else:
                print("\"" + str(insn[0]) + "\"" + " [style=red];\n")          

        elif (write_result[insn[0]] or insn[1]==0):
            if (precs[insn[0]] == 32):
                print("\"" + str(insn[0]) + "\"" + " [style=filled,color=blue];\n")
            else:
                print("\"" + str(insn[0]) + "\"" + " [style=filled];\n")          
        if not(insn[2] == None):
            print("\"" + str(insn[2])+ "\" -> \"" + str(insn[0]) + "\";")
        if not(insn[3] == None):
            print("\"" + str(insn[3])+ "\" -> \"" + str(insn[0]) + "\";")

                      
#    
if __name__ == '__main__':

    p_edges     = rand.dirichlet(dir_p_edges)
    p_ops       = rand.dirichlet(dir_p_ops)   
        
    samp_edge = cat_sampler(p_edges, edge_types)
    samp_op   = cat_sampler(p_ops, op_types)
    const_gen = const_generator()
    samplers = {'edge':samp_edge, 'op':samp_op, 'const': const_gen}

    start_t = time.time()
    prog_count = 1024 #1250

    exec_traces = []
    solutions   = []
    inputs      = []
    write_results = []

    for i in range(prog_count):
        print("\n******\n**prog\n****** " + str(i))
        sol_otc = None
        candidates = None

        prog_g = None
        exec_trace = None
        sol_otc = None
        samp_inputs  = None
        write_result = None

        while (sol_otc is None):          
            soft_constraints['max_edges']      = choice([25,35,45]) 
            soft_constraints['max_out_degree'] = choice([4,5,6])             
            soft_constraints['max_consts']     = choice([3,4,5,6])           
            print("\ngen graph with constraints " + str(soft_constraints))

 
            exec_trace, prog_g, write_result = gen_prog(samplers, soft_constraints)        

            #case where graph cant be constructed due to insufficient parentless candidates
            if (exec_trace is None):
                print("\n**graph construction failed because all operations had operands, before graph completion\n")
                continue

            sol_otc, samp_inputs = search_opt_otc(exec_trace, write_result, samplers)

        if (np.sum(sol_otc) == 0):
            print("\t**sol otc was, SOMEHOW, all-sp")    
            continue

        write_results.append(write_result)

        #FIXME         
        print_for_gviz(exec_trace, write_result, sol_otc)

        exec_traces.append(exec_trace)
        solutions.append(sol_otc)
        inputs.append(samp_inputs)         
 
    end_t = time.time()
            
    print("\n** " + str(prog_count) + " done in " + str(end_t - start_t)) 
                    
    #number of a priori input OTCs are generated
    feats_per_prog = 1#50 
    all_feats = []
    all_labels = [] 

    for prog_idx in range(len(exec_traces)):    
        print("gen feats for prog " + str(prog_idx))

        feats, labels = gen_ds(exec_traces[prog_idx], write_results[prog_idx], solutions[prog_idx], inputs[prog_idx], feats_per_prog) 
        all_feats.append(feats)
        all_labels.append(labels)

    path = sys.argv[1]
    emit_ds(path, exec_traces, all_feats, all_labels, feats_per_prog, offset=0)  #FIXME 
    write_in_sets(path, inputs)






