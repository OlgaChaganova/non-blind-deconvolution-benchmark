import logging
from pathlib import Path

import numpy as np
import scipy


def save_psf(filename: str, new_filename: str):
    image =  scipy.io.loadmat(filename)
    psf = image['f']
    new_path = f'datasets/kernels/motion-blur/processed/Levin/{new_filename}.npy'
    np.save(file=new_path, arr=psf)
    logging.info(f'File {filename} was saved in {new_path}')


def main():
    filenames = Path(f'datasets/kernels/motion-blur/raw/Levin/').rglob('*.mat')
    for i, filename in enumerate(filenames):
        save_psf(filename, new_filename='levin-'+str(i))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()