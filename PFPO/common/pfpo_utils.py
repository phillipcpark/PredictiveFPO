from torch import device as th_dev
from torch.cuda import is_available as cuda_avail

#
def get_dev():
    if not(cuda_avail()):
        raise RuntimeError('CUDA unavailable')
    dev = "cuda:0"
    return th_dev(dev)
