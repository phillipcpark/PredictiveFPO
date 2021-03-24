import mpmath
from mpmath import mp
import sys

def get_mant_prec(prec):
    if (prec == 32):
        return 24
    elif (prec == 64):
        return 53
    elif (prec == 80):
        return 65
    else:
        print("\ninvalid precision encountered in DPS getter")
        sys.exit(1)

# Parameterized scalar floating-point functions
#   Arguments: 
#    -src_<pos>: mp.mpf operand references
#    -prec: integer specifying precision for function's computation
#
#   Returns:
#    -mp.mpf result of function 
#
#   Notes: 
#    -prec sets global context; not thread-safe 
#    -previous context is not restored as context is always explicitly 
#     specified before each simulator step 
def p_const(val, prec):
    mp.dps = get_mant_prec(prec)
    #mp.prec = prec

    return mp.mpf(val)

def p_add(src_l, src_r, prec):
    mp.prec = get_mant_prec(prec)
    #mp.prec = prec

    return src_l+src_r
    
def p_sub(src_l, src_r, prec):
    mp.prec = get_mant_prec(prec)
    #mp.prec = prec
    return src_l-src_r


def p_mul(src_l, src_r, prec):
    mp.prec = get_mant_prec(prec)
    #mp.prec = prec

    return src_l*src_r
    #return mp.mpf(mpmath.nstr(mp.mpf(src_l*src_r), n=get_dps(prec)))

def p_div(src_l, src_r, prec):
    mp.prec = get_mant_prec(prec)

    try:
        return src_l/src_r
    except:
        return None
   
def p_sin(src_l, src_r, prec):
    mp.prec = get_mant_prec(prec)
    #mp.prec = prec
    return mp.sin(src_l)

def p_cos(src_l, src_r, prec):
    mp.prec = get_mant_prec(prec)
    #mp.prec = prec
    return mp.cos(src_l)


p_functions = \
{
    0: p_const,
    1: p_add,
    2: p_sub,
    3: p_mul,
    4: p_div,
    5: p_sin,
    6: p_cos
}

opcodes = {'CONST':0,
           'ADD':1,
           'SUB':2,
           'MUL':3,
           'DIV':4,
           'SIN':5,
           'COS':6 } 
