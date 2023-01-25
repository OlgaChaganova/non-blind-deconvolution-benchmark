import argparse
import logging
from pathlib import Path

import numpy as np


def parse():
    """CLI parser."""
    parser = argparse.ArgumentParser('Parser for eye psf unpacking')
    parser.add_argument('count', type=int, help='Count of kernels to be unpacked')
    return parser.parse_args()


def save_psf(data: dict, new_filename: str):
    new_path = f'datasets/kernels/eye-psf/processed/synthetic/{new_filename}.npy'
    np.save(file=new_path, arr=data)
    logging.info(f'Saved in {new_path}')


def main():
    args = parse()
    big_psfs = np.load('datasets/kernels/eye-psf/raw/synthetic/big_psf_1.npy', allow_pickle=True)
    medium_psfs = np.load('datasets/kernels/eye-psf/raw/synthetic/medium_psf_1.npy', allow_pickle=True)
    small_psfs = np.load('datasets/kernels/eye-psf/raw/synthetic/small_psf_1.npy', allow_pickle=True)

    for i, (big_psf, medium_psf, small_psf) in enumerate(zip(big_psfs, medium_psfs, small_psfs)):
        if i == args.count:
            break
        big_psf['psf'] = big_psf['psf'] / big_psf['psf'].sum()
        save_psf(big_psf, new_filename='big-psf-'+str(i))

        medium_psf['psf'] = medium_psf['psf'] / medium_psf['psf'].sum()
        save_psf(medium_psf, new_filename='medium-psf-'+str(i))

        small_psf['psf'] = small_psf['psf'] / small_psf['psf'].sum()
        save_psf(small_psf, new_filename='small-psf-'+str(i))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()