"""Taken from https://github.com/birdievera/Anisotropic-Gaussian/blob/master/gaussian_filter.py"""

import argparse
import logging
import typing as tp
from math import exp, sqrt, pi

import numpy as np
from scipy.ndimage import rotate


# globals for kernel simulating (set mannually)
SIGMAX = np.arange(2, 10)
SIGMAY = np.arange(2, 10)
SIZES = np.arange(3, 8)
ANGLES = np.arange(-180, 181, 10)


def parse():
    """CLI parser."""
    parser = argparse.ArgumentParser('Parser for isotropic and anisotropic gauss blur kernels simulation')
    parser.add_argument('count', type=int, help='Count of kernels to be simulated')
    return parser.parse_args()


def generate_gauss_kernel(sigmax: float, sigmay: float, size: tp.Tuple[np.int64, np.int64], angle: int) -> np.array:
    """kernel size = 3*sigma for gaussian filters (make a decently sized kernel)"""
    if not isinstance(size, tuple):
        size = (size, size)
    m, n = tuple(s * sigmax + sigmay for s in size)
    kernel = np.zeros((m, n))
    for x in range(m):
        for y in range(n):
            distx, disty = x - m / 2, y - n / 2 # centre pixels
            value = (1.0 / sqrt(2 * pi) * sigmax) * exp((distx ** 2 / ( -1.0 * (sigmax ** 2 )))) *\
                (1.0 / sqrt(2 * pi) * sigmay) * exp(( disty ** 2 / (-1.0 * (sigmay ** 2))))
            kernel[x, y] = value
    kernel = kernel / np.sum(kernel)
    return rotate(kernel, angle=angle)


def save_psf(filename: str, psf: np.array, params: dict):
    new_path = f'datasets/kernels/gauss-blur/processed/synthetic/{filename}.npy'
    data = {
        'psf': psf,
        'params': params,
    }
    np.save(file=new_path, arr=data)
    logging.info(f'File {filename} was saved in {new_path}')


def main():
    args = parse()
    for i in range(args.count):
        sigmax = np.random.choice(SIGMAX)
        sigmay = np.random.choice(SIGMAY)
        size = np.random.choice(SIZES)
        angle = np.random.choice(ANGLES)
        psf = generate_gauss_kernel(sigmax, sigmay, size=(size, size), angle=angle)
        params = {
            'sigmax': sigmax, 
            'sigmay': sigmay,
            'size': size,
            'angle': angle,
        }
        save_psf(filename=f'synthetic-{i}', psf=psf, params=params)
    logging.info('Gauss blur dataset is fully preprocessed.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.random.seed(8)
    main()
