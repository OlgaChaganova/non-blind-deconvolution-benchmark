"""Source code: https://github.com/ysnan/NBD_KerUnc/blob/master/data_loader/kernels.py"""

import numpy as np
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure

from deconv.neural.kerunc.imtools import fspecial, cconv_np


class PSFDistorter(object):
    def __init__(self):
        ...
    
    def __call__(self, ker: np.array, v_g: float, gaus_var: float) -> np.array:
        ''' Our kernel generation method '''
        kh, kv = np.shape(ker)
        nz = v_g * np.random.randn(kh, kv)
        if v_g == 0:
            f = fspecial('gaussian', [5, 5], gaus_var)
            nz_ker = cconv_np(ker, f)
            nz_ker = nz_ker / np.sum(nz_ker)
            return nz_ker

        ## Blurry Some Part of kernel
        W_v = (ker > 0)
        struct = generate_binary_structure(2, 1)
        W_v = binary_dilation(W_v, struct)
        nz_W_v = nz * W_v
        nz_ker = ker + nz_W_v

        ## Omit some part
        if np.random.randint(2):
            Omt = np.random.binomial(1, 0.5, size=[kh, kv])
            Omt[nz_ker == np.max(nz_ker)] = 1
            nz_ker = nz_ker * Omt

        b_s = np.random.randint(1, 5,size = 2)
        b_v = np.random.uniform(0.05,gaus_var)
        f = fspecial('gaussian',b_s, b_v)
        nz_ker = cconv_np(nz_ker, f)

        ## Add uniform noises
        W_rand = np.random.binomial(1, 0.03, size=[kh, kv])
        nz_ker += W_rand * np.random.randn(kh, kv) * 0.002
        if np.sum(nz_ker) ==0:
            print('np.sum(nz_ker)==0')
            nz_ker = ker + nz_W_v
            return nz_ker / np.sum(nz_ker)
        nz_ker[nz_ker < 1e-5] = 0
        nz_ker = nz_ker / np.sum(nz_ker)
        return nz_ker
