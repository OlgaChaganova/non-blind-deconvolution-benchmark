import argparse
import logging
from pathlib import Path

import numpy as np


def parse():
    """CLI parser."""
    parser = argparse.ArgumentParser('Parser for eye psf unpacking')
    parser.add_argument('count', type=int, help='Count of kernels to be unpacked')
    return parser.parse_args()


def save_psf(psf: np.array, params: dict, new_filename: str):
    data = {
        'psf': psf,
        'params': params,
    }
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
        psf = big_psf['psf'][128:-128, 128:-128]
        psf = psf / psf.sum()
        params = {key: value for (key, value) in big_psf.items() if key != 'psf'}
        save_psf(psf, params, new_filename='big-psf-'+str(i))

        psf = medium_psf['psf'][128:-128, 128:-128]
        psf = psf / psf.sum()
        params = {key: value for (key, value) in big_psf.items() if key != 'psf'}
        save_psf(psf, params, new_filename='medium-psf-'+str(i))

        psf = small_psf['psf'][128:-128, 128:-128]
        psf = psf / psf.sum()
        params = {key: value for (key, value) in big_psf.items() if key != 'psf'}
        save_psf(psf, params, new_filename='small-psf-'+str(i))
    logging.info('Eye PSFs dataset is fully preprocessed.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()