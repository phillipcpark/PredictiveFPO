
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


