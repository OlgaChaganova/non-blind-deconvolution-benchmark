import logging
import os
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from data.convolution import convolve
from imutils import center_crop, imread, load_npy, rgb2gray
from metrics import psnr, ssim

_IMAGE_SIZE = 256


def main(benchmark_list_path: str):
    psnr_ssim_dict = {'psnr': [], 'ssim': []}
    metrics = dict()
    for blur_type in ['gauss_blur', 'motion_blur', 'small_eye_blur', 'medium_eye_blur', 'big_eye_blur']:
        metrics[blur_type] = deepcopy(psnr_ssim_dict)

    with open(benchmark_list_path, 'r+') as file:
        next(file)
        for line in tqdm(file):
            blur_type, _, kernel_path, _, image_path = line.strip().split(',')

            if blur_type == 'eye_blur':
                if 'small-psf' in kernel_path:
                    blur_type = 'small_' + blur_type
                elif 'medium-psf' in kernel_path:
                    blur_type = 'medium_' + blur_type
                elif 'big-psf' in kernel_path:
                    blur_type = 'big_' + blur_type

            image = imread(image_path)
            image = center_crop(image, _IMAGE_SIZE, _IMAGE_SIZE)
            if image.ndim == 3:
                image = rgb2gray(image)

            kernel = load_npy(kernel_path, key='psf')

            blurred = convolve(image, kernel)

            metrics[blur_type]['psnr'].append(psnr(image, blurred))
            metrics[blur_type]['ssim'].append(ssim(image, blurred))

    logging.info('Evaluating blur strength for different blur types:')
    for blur_type in metrics.keys():
        mean_psnr = np.mean(metrics[blur_type]['psnr'])
        mean_ssim = np.mean(metrics[blur_type]['ssim'])
        logging.info(f'For {blur_type}: mean PSNR: {mean_psnr}, mean SSIM: {mean_ssim}')


if __name__ == '__main__':
    logging.basicConfig(filename=os.path.join('results', 'measure_blur_strength.log'), level=logging.INFO)
    main(benchmark_list_path=os.path.join('datasets', 'full_dataset.txt'))
