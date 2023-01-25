import logging
from pathlib import Path

import cv2
import numpy as np


def read_psf(img_path: str) -> np.array:
    img_path = str(img_path) if not isinstance(img_path, str) else img_path
    image = cv2.imread(img_path)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image / 255


def save_psf(filename: str, new_filename: str):
    psf = read_psf(filename)
    if not np.allclose(psf.sum(), 1, atol=0.01, rtol=0.01):
        psf /= psf.sum()
    new_path = f'datasets/kernels/motion-blur/processed/Sun/{new_filename}.npy'
    np.save(file=new_path, arr=psf)
    logging.info(f'File {filename} was saved in {new_path}')


def main():
    filenames = Path(f'datasets/kernels/motion-blur/raw/Sun').rglob('*.png')
    for i, filename in enumerate(filenames):
        save_psf(filename, new_filename='sun-'+str(i))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()