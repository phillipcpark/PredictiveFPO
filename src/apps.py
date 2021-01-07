import train as tr

# X1
# X2
# C0              (1)
# MUL0, X1,   X1
# ADD0, MUL0, C0  (d)

# C1               (3) 
# MUL1, C1,   X1
# MUL2, MUL1, X1
# C2               (2)
# MUL3, C2,   X2
# ADD1, MUL2, MUL3
# SUB0, ADD1, X1   (t)

# C3               (3)
# C4               (2)
# MUL4, C3,   X1
# MUL5, MUL4, X1
# MUL6, C4,   X2
# SUB1, MUL5, MUL6
# SUB2, SUB1, X1   (t*)

# DIV0, SUB0, ADD0 (s)
# DIV1, SUB2, ADD0 (s*)


# TODO
def jet_app():
   
    edges = [] 

    feats = [] # (opcode, init_prec) pairs
  
    graph = tr.create_graph(edges,\
                            [OP_ENC[feat[0]][feat[1]] for feat in ex_feats],\
                            unary_mask) 
    return graph


