from copy import deepcopy

USE_GPU = True

#doesn't use input OTCs and predicts precision directly
# don't use with COARSE_TUNE or SP_TARGET 
SINGLE_GRAPH_TARG = False 

#target is binary (single-class problem; e.g. tune to 32, or tune -1 type, etc.)
COARSE_TUNE = True 

#select whether or not to predict ops for 32, rather decision to tune -1 type
SP_TARGET = False

###############
#training specs 
###############
BAT_SZ     = 1024
EPOCHS     = 32 #16 #32 
H_DIM      = 32 #64
TR_DS_PROP = 0.9 #0.8 #0.85 #0.85
VAL_DS_PROP = 0.01 #0.005 #0.1

L_RATE     = 0.1
USE_CL_BAL = False

CLASSES = 3 if SINGLE_GRAPH_TARG else 5
if (COARSE_TUNE):
    CLASSES = 2

IGNORE_CLASS = 6

######################
#model and ds gen dims
#######################
MP_STEPS      = 8
TIE_MP_PARAMS = False #True


#FIXME FIXME
USE_PRED_THRESH = True
PRED_THRESH   = 0.6


input_samp_sz = 10000    
inputs_mag    = 3

err_thresh      = 0.0000001
err_accept_prop = 0.999

CONST_PREC = 64


#######
# other
#######
NUM_ATTRIBUTES = 7
GRAPH_DELIM = ['' for i in range(NUM_ATTRIBUTES)]

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







