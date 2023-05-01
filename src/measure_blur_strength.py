import logging
import os
import typing as tp
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from data.convertation import srgbf_to_linrgbf, uint8_to_float32
from data.convolution import convolve
from imutils import imread, load_npy, srgb2gray
from metrics import psnr, ssim


def calc_stats(values: np.array) -> tp.Tuple[float, float, float, float]:
    """Return mean, std, max and min values of array"""
    mean_value = np.mean(values)
    std_value = np.std(values)
    max_value = np.max(values)
    min_value = np.min(values)
    return mean_value, std_value, max_value, min_value


def main(benchmark_list_path: str) -> tp.Dict[str, tp.List[float]]:
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

            if image_path.endswith('.jpg'):  # sRGB 8 bit
                image = uint8_to_float32(image)  # sRGB float

            if image.ndim == 3:
                image = srgb2gray(image)  # sRGB float
            
            image = srgbf_to_linrgbf(image)  # linRGB float

            kernel = load_npy(kernel_path, key='psf')

            blurred = convolve(image, kernel).astype(np.float32)

            metrics[blur_type]['psnr'].append(psnr(image, blurred))
            metrics[blur_type]['ssim'].append(ssim(image, blurred))

    logging.info('Evaluating blur strength for different blur types:')
    for blur_type in metrics.keys():
        mean_psnr, std_psnr, max_psnr, min_psnr = calc_stats(metrics[blur_type]['psnr'])
        mean_ssim, std_ssim, max_ssim, min_ssim = calc_stats(metrics[blur_type]['ssim'])
        logging.info(
            f'For {blur_type}: PSNR: max: {max_psnr:.3f}, min: {min_psnr:.3f}, mean: {mean_psnr:.3f}, std: {std_psnr:.3f}; '
            f'SSIM: max: {max_ssim:.3f},  min: {min_ssim:.3f}, mean: {mean_ssim:.3f}, std: {std_ssim:.3f}',
        )
    return metrics


if __name__ == '__main__':
    logging.basicConfig(filename=os.path.join('results', 'measure_blur_strength.log'), level=logging.INFO)
    main(benchmark_list_path=os.path.join('datasets', 'full_dataset.txt'))
