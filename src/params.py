from copy import deepcopy

SINGLE_GRAPH_TARG = True

BAT_SZ     = 50 #64
EPOCHS     = 16 #1 #64 #128 #32
H_DIM      = 32
TR_DS_PROP = 0.33 #0.85 #0.85
VAL_DS_PROP = 0.05 #0.005 #0.1

LAYERS = 7

L_RATE = 0.1
USE_CL_BAL = True


CLASSES = 3 if SINGLE_GRAPH_TARG else 5
IGNORE_CLASS = 6

MP_STEPS = 8

NUM_ATTRIBUTES = 7
GRAPH_DELIM = ['' for i in range(NUM_ATTRIBUTES)]

err_thresh = 0.0000001
err_accept_prop = 0.995

input_samp_sz = 10000    
inputs_mag = 3

precs = {32:0, 64:1, 80:2}
precs_inv = {0:32, 1:64, 2:80}

ops   = {'CONST':0,\
         'ADD':1, \
         'SUB':2, \
         'MUL':3, \
         'DIV':4, \
         'SIN':5, \
         'COS':6}

OP_ENC_DIM = len(ops.keys())*len(precs.keys()) 
OP_ENC     = {} 

curr_enc = [1.0] + [0.0 for i in range(OP_ENC_DIM-1)]
for op in ops.keys():
    curr_op_encs = {} 

    for prec in precs.keys():
        curr_op_encs[precs[prec]] = deepcopy(curr_enc) 

        curr_enc.insert(0, 0.0)
        curr_enc.pop()

    OP_ENC[ops[op]] = curr_op_encs

#encoding without precision
OP_ENC_NOPREC_DIM = len(ops.keys())
OP_ENC_NOPREC     = {} 

curr_enc = [1.0] + [0.0 for i in range(OP_ENC_NOPREC_DIM-1)]
for op in ops.keys():
    curr_op_encs = deepcopy(curr_enc) 
    curr_enc.insert(0, 0.0)
    curr_enc.pop()
    OP_ENC_NOPREC[ops[op]] = curr_op_encs








