from copy import deepcopy

BAT_SZ     = 128 #32
EPOCHS     = 64 #256
H_DIM      = 32
TR_DS_PROP = 0.8
L_RATE     = 0.05

CLASSES = 5
LAYERS  = 1 #5

NUM_ATTRIBUTES = 7
GRAPH_DELIM = ['' for i in range(NUM_ATTRIBUTES)]

precs = {32:0, 64:1, 80:2}

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

