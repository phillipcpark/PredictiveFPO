import csv
from mpmath import mp
from common.otc import is_const

#
def pad_inputs(exec_list, ins):
    cidx = 0
    flush_ins = []
    for line in exec_list:
        if is_const(line[1]):
            flush_ins.append(ins[cidx])
            cidx += 1
        else:
            flush_ins.append(None)
    return flush_ins

#
def load_inputs(path):
    f_hand = open(path, 'r')  
    reader = csv.reader(f_hand, delimiter=',')

    ins = []

    mp.prec = 65
    for row in reader:        
        ins.append([mp.mpf(val) for val in row])
    f_hand.close()

    return ins
