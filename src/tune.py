"""Script for tuning Wiener parameters (balance) in case on non-blind noise scenario"""

import logging
import typing as tp
from copy import deepcopy
from time import time

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm 

from deconv.classic.wiener.wiener import wiener_gray
from data.convolution import convolve
from imutils import make_noised, imread, load_npy
from metrics import psnr, ssim


_NOISE_NAME = 'noise'
_NO_NOISE_NAME = 'no_noise'
_NOISED_TYPES = [_NOISE_NAME, _NO_NOISE_NAME]
_SSIM_NAME = 'ssim'
_PSNR_NAME = 'psnr'
_METRIC_TYPES = [_SSIM_NAME, _PSNR_NAME]


def grid_search(
    balance_values_noise: tp.List[float],
    balance_values_no_noise: tp.List[float],
    gt_images: tp.List[str],
    kernels: tp.List[str],
    mu: float,
    sigma: float,
) -> tp.Tuple[float, float, float]:
    """
    Perform simple grid search for finding optimal balance value.

    Args:
        balance_values_noise: tp.List[float]
            List with values of balance parameters considered to be a candidate to the optimal value in NOISE case.
        balance_values_no_noise: tp.List[float]
            List with values of balance parameters considered to be a candidate to the optimal value in NO NOISE case.
        gt_images: tp.List[str]
            List with paths to ground truth (sharp) images.
        kernels: tp.List[str]
            List with path to kernels.
        mu: float
            Noise parameter (mu)
        sigma: float
            Noise parameter (sigma)
    Returns:
        tp.Tuple[float, float, float]
        Optimal balance value and corresponding ssim and psnr values.
    """

    # initialize dictionaries for metrics storage
    metrics_per_noise = dict()
    metrics_per_noise[_NO_NOISE_NAME] = dict()
    metrics_per_noise[_NOISE_NAME] = dict()
    for metric_type in _METRIC_TYPES:
        metrics_per_noise[_NO_NOISE_NAME][metric_type] = dict()
        metrics_per_noise[_NO_NOISE_NAME][metric_type] = {balance_value: [] for balance_value in balance_values_no_noise}

        metrics_per_noise[_NOISE_NAME][metric_type] = dict()
        metrics_per_noise[_NOISE_NAME][metric_type] = {balance_value: [] for balance_value in balance_values_noise}

    # calculating metrics for each pair (image, kernel) for each balance value in the grid
    for gt_image, kernel in tqdm(zip(gt_images, kernels)):
        gt_image = imread(gt_image)
        kernel = load_npy(kernel, key='psf')

        # float
        blurred = convolve(gt_image, kernel)
        noised_blurred = make_noised(blurred, mu=mu, sigma=sigma)
        for balance_value in balance_values_no_noise:
            restored = wiener_gray(blurred, kernel, balance=balance_value, clip=True)
            metrics_per_noise[_NO_NOISE_NAME][_PSNR_NAME][balance_value].append(psnr(gt_image, restored))
            metrics_per_noise[_NO_NOISE_NAME][_SSIM_NAME][balance_value].append(ssim(gt_image, restored))

        for balance_value in balance_values_noise:
            restored_noised = wiener_gray(noised_blurred, kernel, balance=balance_value, clip=True)
            metrics_per_noise[_NOISE_NAME][_PSNR_NAME][balance_value].append(psnr(gt_image, restored_noised))
            metrics_per_noise[_NOISE_NAME][_SSIM_NAME][balance_value].append(ssim(gt_image, restored_noised))
        
    # finding average metrics for each balance value in the grid and the optimal values
    for noise_type in _NOISED_TYPES:
        logging.info(f'Finding optimal balance value for {noise_type} scenario')
        max_psnr, max_ssim = 0, 0
        optimal_balance = None
        balance_values = balance_values_no_noise if noise_type == _NO_NOISE_NAME else balance_values_noise
        for balance_value in balance_values:
            mean_psnr = np.mean(metrics_per_noise[noise_type][_PSNR_NAME][balance_value])
            mean_ssim = np.mean(metrics_per_noise[noise_type][_SSIM_NAME][balance_value])

            metrics_per_noise[noise_type][_PSNR_NAME][balance_value] = mean_psnr
            metrics_per_noise[noise_type][_SSIM_NAME][balance_value] = mean_ssim

            if mean_psnr >= max_psnr and mean_ssim >= max_ssim:
                max_psnr, max_ssim = mean_psnr, mean_ssim
                optimal_balance = balance_value
                print(f'Optimal balance: {optimal_balance}; psnr: {max_psnr}, ssim: {max_ssim}')
        logging.info(f'Optimal balance value for {noise_type.upper()} scenario is {optimal_balance}')
        logging.info(f'The best metrics: ssim is {max_ssim}; psnr is {max_psnr}')


def get_paths(benchmark_list_path: str) -> tp.Tuple[tp.List[str], tp.List[str]]:
    images_path = []
    kernels_path = []
    with open(benchmark_list_path, 'r+') as file:
        next(file)
        for line in tqdm(file):
            _, _, kernel_path, _, image_path = line.strip().split(',')
            if image_path.endswith('.png'):
                images_path.append(image_path)
                kernels_path.append(kernel_path)
    return images_path, kernels_path


def main():
    logging.basicConfig(filename='wiener_tuning_results.log', level=logging.INFO)

    config = OmegaConf.load('config.yml')

    benchmark_list_path = config.dataset.benchmark_list_path
    balance_values = config.wiener_tuning.balance_values
    images_path, kernels_path = get_paths(benchmark_list_path)

    mu = config.dataset.blur.mu
    sigma = config.dataset.blur.sigma

    logging.info(f'Tuning on {benchmark_list_path}')
    start_time = time()
    grid_search(
        balance_values_noise=balance_values.noise,
        balance_values_no_noise=balance_values.no_noise,
        gt_images=images_path,
        kernels=kernels_path,
        mu=mu,
        sigma=sigma,
    )
    logging.info(f'Full time: {(time() - start_time) / 60} minutes')

if __name__ == '__main__':
    np.random.seed(8)
    main()