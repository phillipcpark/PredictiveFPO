import random as rand


# should all inputs be kept in memory? If I want to share inputs for every OTC...
def gen_inputs(prog, samp_sz, bound_low, bound_high):
   inputs = []
   for samp in range(samp_sz):
       samp_inputs = []
       for insn in prog:
           is_const    = True if insn[1] == 0 else False 
           if (is_const):
               samp_inputs.append(rand.uniform(bound_low, bound_high))
           else:
               samp_inputs.append(0.0)
       inputs.append(samp_inputs)
   return inputs


# samples seperately at each magnitude
def gen_stratified_inputs(prog, samp_sz, max_mag):
   inputs  = []

   samps_per_mag = int(samp_sz / (max_mag * 2))

   for mag in range(-max_mag, max_mag+1):

       for samp in range(samps_per_mag):
           samp_inputs = []

           for insn in prog:                            
               is_const = True if insn[1] == 0 else False 
               if (is_const):
                   samp_inputs.append(rand.uniform(-1.0 * (10.0**mag), 10.0**mag))
               else:
                   samp_inputs.append(0.0)
           inputs.append(samp_inputs)

   return inputs   
