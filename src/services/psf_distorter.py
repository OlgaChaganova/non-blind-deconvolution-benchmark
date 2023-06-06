"""
PSF distortion module.

Source code for motion distortion: https://github.com/ysnan/NBD_KerUnc/blob/master/data_loader/kernels.py
"""

import typing as tp

import cv2
import numpy as np
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure

from deconv.neural.kerunc.imtools import fspecial, cconv_np
from data.generate.gauss_blur import generate_gauss_kernel


class PSFDistorter(object):
    def __init__(self):
        ...
    
    def __call__(self, type: tp.Literal['motion', 'gauss', 'eye'], **kwargs) -> np.ndarray:
        if type == 'motion_blur':
            return self._motion(**kwargs)
        elif type == 'gauss_blur':
            return self._gauss(**kwargs)
        elif type == 'eye_blur':
            return self._eye(**kwargs)
        else:
            raise ValueError(
                f'Available types are motion_blur, gauss_blur, eye_blur, but got {type}'
            )

    def _motion(self, psf: np.ndarray, v_g: float, gaus_var: float) -> np.ndarray:
        """Distort motion blur kernels."""

        kh, kv = np.shape(psf)
        nz = v_g * np.random.randn(kh, kv)
        if v_g == 0:
            f = fspecial('gaussian', [5, 5], gaus_var)
            nz_ker = cconv_np(psf, f)
            nz_ker = nz_ker / np.sum(nz_ker)
            return nz_ker

        ## Blurry Some Part of kernel
        W_v = (psf > 0)
        struct = generate_binary_structure(2, 1)
        W_v = binary_dilation(W_v, struct)
        nz_W_v = nz * W_v
        nz_ker = psf + nz_W_v

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
            nz_ker = psf + nz_W_v
            return nz_ker / np.sum(nz_ker)
        nz_ker[nz_ker < 1e-5] = 0
        nz_ker = nz_ker / np.sum(nz_ker)
        return nz_ker

    def _gauss(self, psf_params: tp.Dict[str, int], max_delta_sigma: int, max_delta_angle: int) -> np.ndarray:
        new_params = {}
        for param in psf_params.keys():
            if param == 'size':
                new_params['size'] = int(psf_params['size'])
            elif param == 'angle':
                delta = max_delta_angle
            elif param in ['sigmax', 'sigmay']:
                delta = min(max_delta_sigma, psf_params[param])

            delta = np.random.choice(range(0, delta, 1))
            sign = 1 if np.random.randn() > 0.5 else -1
            new_params[param] = psf_params[param] + sign * delta

        print(f'Old params: {psf_params}')
        print(f'New params: {new_params}')
        return generate_gauss_kernel(**new_params)
    
    def _eye(self, psf: np.ndarray, delta_size: int) -> np.ndarray:
        psf = cv2.resize(psf, (psf.shape[0] + delta_size, psf.shape[1] + delta_size))
        return psf / psf.sum()
