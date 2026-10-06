import math
import numpy as np

import torch
import torch.sparse
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import scipy
import scipy.sparse as sp
import ipdb


def fwd_solver(f, z_init, threshold, max_iter, mode='abs'):
    # https://github.com/liu-jc/MGNNI
    z_prev, z = z_init, f(z_init)
    nstep = 0
    while nstep < max_iter:
        z_prev, z = z, f(z)
        abs_diff = torch.norm(z_prev - z).item()
        rel_diff = abs_diff / (torch.norm(z_prev).item() + 1e-9)
        diff_dict = {'abs': abs_diff, 'rel': rel_diff}
        if diff_dict[mode] < threshold:
            # print(nstep, abs_diff)
            break
        nstep += 1
    if nstep == max_iter:
        pass
        # print(f'step {nstep}, not converged, abs_diff: {abs_diff}')
    return z, abs_diff, nstep
