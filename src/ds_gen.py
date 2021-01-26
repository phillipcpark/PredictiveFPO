from prog_gen import *


# generates sz number of input-output pairs from random a priori OTCs and tuning relative to solution   
def gen_ds(exec_trace, solution, inputs, sz):
    feats  = []
    labels = []
            
    gt_otc = gen_spec_otc(exec_trace, precisions[-1])  
    shad_results = []

    for ins in inputs:
        shad_results.append(sim_prog(exec_trace, ins, gt_otc)) 

    input_sz = len(inputs)        

    for i_sol in range(sz):
        if (i_sol % 20 == 0):
            print("\tcreating init sol feat " + str(i_sol))

        cand = None       
        valid = False

        while not(valid):
            cand = gen_rand_otc(exec_trace)    
    
            for i_idx in range(input_sz):
                result_cand = sim_prog(exec_trace, inputs[i_idx], cand) 
    
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
def emit_ds(path, traces, feats, labels, feats_per_prog):
    ds_path = path 
 
    with open(ds_path, 'w') as f_hand:
        f_writer = csv.writer(f_hand, delimiter=',')

        header = ['gid', 'nid','opcode','src_l','src_r','init_prec','tune_rec']
        f_writer.writerow(header)

        prog_count = len(traces) 

        for prog_idx in range(prog_count):             
            trace_len = len(traces[prog_idx])

            for otc_idx in range(feats_per_prog):
                for insn_idx in range(trace_len):                                       
                    f_writer.writerow([prog_idx] + traces[prog_idx][insn_idx] + [feats[prog_idx][otc_idx][insn_idx]] + \
                                      [labels[prog_idx][otc_idx][insn_idx]])                
                f_writer.writerow([None for attrs in range(len(traces[prog_idx][0]) + 3)])     


#
def print_for_gviz(prog_g, exec_trace, precs):
    edges = prog_g.edges()

    counter = 0
    node_ids = {}
    for node in netwx.topological_sort(prog_g):    
        node_ids[node] = counter
        counter += 1              
 
    for e in edges:
        if not(exec_trace[node_ids[e[0]]][1] == 0 and not(exec_trace[node_ids[e[1]]][1] == 0)):                 
            print(str(e[0])+"_"+str(precs[node_ids[e[0]]]) + "->" + str(e[1])+ "_" + str(precs[node_ids[e[1]]]) + ";")

        elif not(exec_trace[node_ids[e[0]]][1] == 0):
                print(str(e[0]) + "_" + str(precs[node_ids[e[0]]]) + "->" + str(e[1])+ ";")
 
        elif not(exec_trace[node_ids[e[1]]][1] == 0):
                print(str(e[0]) + "->" + str(e[1])+ "_" + str(precs[node_ids[e[1]]]) + ";")

        else: 
            print(str(e[0]) + "->" + str(e[1])+ ";")


                      
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

        if not(np.sum(sol_otc) == 0):
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








