from copy import deepcopy

EXP_NAME      = None

#NOTE 'lg' model uses 3 resnet blocks, whereas regular uses 2...

#MOD_PATH    = 'resources/train_32k_LG/_ep504_tr0.57_val0.62' 
#TST_IDXS_PATH = 'resources/train_32k_LG/tst_idxs'

MOD_PATH      = 'resources/train_32k/_ep174_tr0.56_val0.61'
TST_IDXS_PATH = 'resources/train_32k/tst_idxs'
 
USE_GPU       = False
MAX_TST_PROGS = 512

###############
#training specs 
###############
BAT_SZ     = 768
EPOCHS     = 256
H_DIM      = 32 
TR_DS_PROP = 0.75
VAL_DS_PROP = 0.125

L_RATE     = 0.1
USE_CL_BAL = True

CLASSES      = 2
IGNORE_CLASS = -1

######################
#model and ds gen dims
#######################
MP_STEPS      = 8
TIE_MP_PARAMS = False

USE_PRED_THRESH = True
PRED_THRESH   = 0.56

input_samp_sz = 10000    
inputs_mag    = 3

err_thresh      = 0.0000001
err_accept_prop = 0.99


#######
# other
#######
NUM_ATTRIBUTES = 7
CONST_PREC     = 64

GRAPH_DELIM = ['' for i in range(NUM_ATTRIBUTES)]

precs     = {32:0, 64:1, 80:2}
precs_inv = {0:32, 1:64, 2:80}

ops = {'CONST':0,
       'ADD':1, 
       'SUB':2, 
       'MUL':3, 
       'DIV':4, 
       'SIN':5, 
       'COS':6,
       'TAN':7,
       'ASIN':8,
       'ACOS':9,
       'ATAN':10,
       'SQRT':11,
       'POW':12 } 

ops_inv = {0: 'CONST',\
           1: '+', \
           2: '-',\
           3: 'x',\
           4: '/',\
           5: 'sin',\
           6: 'cos'}

# 1-hot vector encoding of operations
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



