"""Taken from https://github.com/donggong1/learn-optimizer-rgdn/blob/master/data/make_kernel.py"""

import argparse
import logging

import numpy as np
from scipy.interpolate import splrep, splev
from scipy.signal import convolve2d

# globals for kernel simulating (taken from RGDN repository)
SP_SIZES = (11, 16, 21, 26, 31)
K_SIZE = 41
NUM_SPL_CTRL = (3, 4, 5, 6)


def parse():
    """CLI parser."""
    parser = argparse.ArgumentParser('Parser for motion kernels simulation')
    parser.add_argument('count', type=int, help='Count of kernels to be simulated')
    return parser.parse_args()


def kernel_sim_spline(psz: int, mxsz: int, nc: int, num: int = 1) -> np.array:
    """Spline-based blur kernel simulation."""
    k = np.zeros([mxsz, mxsz, num], dtype=np.float32)
    imp = np.zeros([mxsz, mxsz], dtype=np.float32)
    imp[(mxsz + 1) // 2, (mxsz + 1) // 2] = 1.0

    xg, yg = np.meshgrid(np.arange(0, psz), np.arange(0, psz))

    for i in range(num):
        while True:
            x = np.random.randint(0, psz, nc)
            y = np.random.randint(0, psz, nc)
            order = min(nc - 1, 3)

            spx = splrep(np.linspace(0, 1, nc), x.astype(np.float32), k=order)
            x = splev(np.linspace(0, 1, nc * 5000), spx)
            x = np.clip(x, 0, psz - 1)
            x = np.round(x).astype(np.int32)

            spy = splrep(np.linspace(0, 1, nc), y.astype(np.float32), k=order)
            y = splev(np.linspace(0, 1, nc * 5000), spy)
            y = np.clip(y, 0, psz - 1)
            y = np.round(y).astype(np.int32)

            idx = x * psz + y
            idx = np.unique(idx)

            weight = np.random.randn(np.prod(idx.shape)) * 0.5 + 1
            weight = np.clip(weight, 0, None)

            if (np.sum(weight) == 0):
                continue

            weight = weight / np.sum(weight)
            kernel = np.zeros([psz * psz])

            kernel[idx] = weight

            kernel = np.reshape(kernel, [psz, psz])

            cx = int(np.round(np.sum(kernel * xg)))
            cy = int(np.round(np.sum(kernel * yg)))

            if cx <= psz / 2:
                padding = np.zeros([psz, psz - 2 * cx + 1])
                kernel = np.concatenate([padding, kernel], axis=1)
            else:
                padding = np.zeros([psz, 2 * cx - psz - 1])
                kernel = np.concatenate([kernel, padding], axis=1)

            p2 = kernel.shape[1]

            if cy <= psz / 2:
                padding = np.zeros([psz - 2 * cy + 1, p2])
                kernel = np.concatenate([padding, kernel], axis=0)
            else:
                padding = np.zeros([2 * cy - psz - 1, p2])
                kernel = np.concatenate([kernel, padding], axis=0)
            if np.max(kernel.shape) <= mxsz:
                break
        kernel = kernel.astype(np.float32)
        ck = convolve2d(imp, kernel, 'same')
        k[:, :, i] = ck
    return k[..., 0]


def save_psf(filename: str, psf: np.array, params: dict):
    new_path = f'datasets/kernels/motion-blur/processed/synthetic/{filename}.npy'
    data = {
        'psf': psf,
        'params': params,
    }
    np.save(file=new_path, arr=data)
    logging.info(f'File {filename} was saved in {new_path}')


def main():
    args = parse()
    for i in range(args.count):
        sp_size = np.random.choice(SP_SIZES)
        num_spl_ctrl = np.random.choice(NUM_SPL_CTRL)
        params = {
            'sp_size': sp_size, 
            'k_size': K_SIZE,
            'num_spl_ctrl': num_spl_ctrl,
        }
        psf = kernel_sim_spline(psz=sp_size, mxsz=K_SIZE, nc=num_spl_ctrl)
        save_psf(filename=f'synthetic-{i}', psf=psf, params=params)
    logging.info('Motion blur dataset is fully preprocessed.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.random.seed(8)
    main()