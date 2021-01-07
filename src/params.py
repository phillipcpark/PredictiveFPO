from copy import deepcopy

BAT_SZ     = 128
EPOCHS     = 32
H_DIM      = 32
TR_DS_PROP = 0.8
VAL_DS_PROP = 0.1

LAYERS = 5

L_RATE = 0.1

CLASSES = 5
IGNORE_CLASS = 6

NUM_ATTRIBUTES = 7
GRAPH_DELIM = ['' for i in range(NUM_ATTRIBUTES)]

err_thresh = 0.00001

input_samp_sz = 5000    
inputs_mag = 6

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




