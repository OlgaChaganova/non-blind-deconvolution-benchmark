"""Script for tuning Wiener parameters (balance) in case of non-blind noise scenario"""

import logging
import os
import typing as tp
from time import time

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm 

from deconv.classic.wiener.wiener import wiener_gray
from data.convertation import float2linrgb16bit, float2srgb8, linrrgb2srgb8bit
from data.convolution import convolve
from imutils import make_noised, imread, load_npy, rgb2gray, center_crop
from metrics import psnr, ssim


_IMAGE_SIZE = 256  # required image size 
_MAX_UINT8 = 2 ** 8 - 1
_MAX_UINT16 = 2 ** 16 - 1

_NOISE_NAME = 'noise'
_NO_NOISE_NAME = 'no_noise'
_NOISED_TYPES = [_NOISE_NAME, _NO_NOISE_NAME]

_SSIM_NAME = 'ssim'
_PSNR_NAME = 'psnr'
_METRIC_TYPES = [_SSIM_NAME, _PSNR_NAME]


def grid_search_non_blind_noise(
    balance_values_noise: tp.List[float],
    balance_values_no_noise: tp.List[float],
    gt_images: tp.List[str],
    kernels: tp.List[str],
    mu: float,
    sigma: float,
) -> tp.Tuple[float, float, float]:
    """
    Perform simple grid search for finding optimal balance value.
    NO_NOISE: float without noise
    NOISE: linRGB, sRGB (because of discretization noise), float with noise

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
    metrics = dict()
    metrics[_NO_NOISE_NAME] = dict()
    metrics[_NOISE_NAME] = dict()
    for metric_type in _METRIC_TYPES:
        metrics[_NO_NOISE_NAME][metric_type] = dict()
        metrics[_NO_NOISE_NAME][metric_type] = {balance_value: [] for balance_value in balance_values_no_noise}

        metrics[_NOISE_NAME][metric_type] = dict()
        metrics[_NOISE_NAME][metric_type] = {balance_value: [] for balance_value in balance_values_noise}

    # calculating metrics for each pair (image, kernel) for each balance value in the grid
    for image_path, kernel_path in tqdm(zip(gt_images, kernels)):
        image = imread(image_path)
        if image.ndim == 3:
            image = rgb2gray(image)
        image = center_crop(image, _IMAGE_SIZE, _IMAGE_SIZE)
        kernel = load_npy(kernel_path, key='psf')

        if image_path.endswith('png'):
            # ---- float ----
            blurred = convolve(image, kernel)
            noised_blurred = make_noised(blurred, mu=mu, sigma=sigma)
            _calc_metrics_in_nonblind_case(
                balance_values_no_noise,
                balance_values_noise,
                metrics,
                blurred,
                noised_blurred,
                kernel,
                image,
                type='float',
            )
            
            # ---- lin RGB uint16 ----
            image = float2linrgb16bit(image)
            blurred = float2linrgb16bit(blurred)
            noised_blurred = float2linrgb16bit(noised_blurred)
            _calc_metrics_in_nonblind_case(
                balance_values_no_noise,
                balance_values_noise,
                metrics,
                (blurred / _MAX_UINT16).astype(np.float32),
                (noised_blurred / _MAX_UINT16).astype(np.float32),
                kernel,
                (image / _MAX_UINT16).astype(np.float32),
                type='linrgb',
            )

            # ---- sRGB uint8 ----
            image = linrrgb2srgb8bit(image)
            blurred = linrrgb2srgb8bit(blurred)
            noised_blurred = linrrgb2srgb8bit(noised_blurred)
            _calc_metrics_in_nonblind_case(
                balance_values_no_noise,
                balance_values_noise,
                metrics,
                (blurred / _MAX_UINT8).astype(np.float32),
                (noised_blurred / _MAX_UINT8).astype(np.float32),
                kernel,
                (image / _MAX_UINT8).astype(np.float32),
                type='srgb',
            )
        else:
            # ---- sRGB uint8 ----
            if image.dtype in [np.float16, np.float32, np.float64]:  # after rgb2gray uint8 might become float
                image = float2srgb8(image)
            blurred = convolve(image, kernel).astype(np.uint8)
            noised_blurred = make_noised(blurred, mu=mu, sigma=sigma).astype(np.uint8)
            _calc_metrics_in_nonblind_case(
                balance_values_no_noise,
                balance_values_noise,
                metrics,
                (blurred / _MAX_UINT8).astype(np.float32),
                (noised_blurred / _MAX_UINT8).astype(np.float32),
                kernel,
                (image / _MAX_UINT8).astype(np.float32),
                type='srgb',
            )

    # finding average metrics for each balance value in the grid and the optimal values
    for noise_type in _NOISED_TYPES:
        logging.info(f'NON_BLIND NOISE: Finding optimal balance value for {noise_type} scenario')
        max_psnr, max_ssim = 0, 0
        optimal_balance = None
        balance_values = balance_values_no_noise if noise_type == _NO_NOISE_NAME else balance_values_noise
        for balance_value in balance_values:
            mean_psnr = np.mean(metrics[noise_type][_PSNR_NAME][balance_value])
            mean_ssim = np.mean(metrics[noise_type][_SSIM_NAME][balance_value])

            metrics[noise_type][_PSNR_NAME][balance_value] = mean_psnr
            metrics[noise_type][_SSIM_NAME][balance_value] = mean_ssim

            if mean_psnr >= max_psnr and mean_ssim >= max_ssim:
                max_psnr, max_ssim = mean_psnr, mean_ssim
                optimal_balance = balance_value
                print(f'Optimal balance: {optimal_balance}; psnr: {max_psnr}, ssim: {max_ssim}')
        logging.info(f'Optimal balance value for {noise_type.upper()} scenario is {optimal_balance}')
        logging.info(f'The best metrics: ssim is {max_ssim}; psnr is {max_psnr}')


def _calc_metrics_in_nonblind_case(
    balance_values_no_noise: tp.List[float],
    balance_values_noise: tp.List[float],
    metrics: dict,
    blurred: np.array,
    noised_blurred: np.array,
    kernel: np.array,
    gt_image: np.array,
    type: tp.Literal['float', 'linrgb', 'srgb'],
):
    if type == 'float':
        for balance_value in balance_values_no_noise:
            restored = wiener_gray(blurred, kernel, balance=balance_value, clip=True)
            metrics[_NO_NOISE_NAME][_PSNR_NAME][balance_value].append(psnr(gt_image, restored))
            metrics[_NO_NOISE_NAME][_SSIM_NAME][balance_value].append(ssim(gt_image, restored))

        for balance_value in balance_values_noise:
            restored_noised = wiener_gray(noised_blurred, kernel, balance=balance_value, clip=True)
            metrics[_NOISE_NAME][_PSNR_NAME][balance_value].append(psnr(gt_image, restored_noised))
            metrics[_NOISE_NAME][_SSIM_NAME][balance_value].append(ssim(gt_image, restored_noised))
    else:
        for balance_value in balance_values_noise:
            restored = wiener_gray(blurred, kernel, balance=balance_value, clip=True)
            metrics[_NOISE_NAME][_PSNR_NAME][balance_value].append(psnr(gt_image, restored))
            metrics[_NOISE_NAME][_SSIM_NAME][balance_value].append(ssim(gt_image, restored))

            restored_noised = wiener_gray(noised_blurred, kernel, balance=balance_value, clip=True)
            metrics[_NOISE_NAME][_PSNR_NAME][balance_value].append(psnr(gt_image, restored_noised))
            metrics[_NOISE_NAME][_SSIM_NAME][balance_value].append(ssim(gt_image, restored_noised))

def get_paths(benchmark_list_path: str) -> tp.Tuple[tp.List[str], tp.List[str]]:
    images_path = []
    kernels_path = []
    with open(benchmark_list_path, 'r+') as file:
        next(file)
        for line in tqdm(file):
            _, _, kernel_path, _, image_path = line.strip().split(',')
            images_path.append(image_path)
            kernels_path.append(kernel_path)
    return images_path, kernels_path


def grid_search_blind_noise(
    balance_values: tp.List[float],
    gt_images: tp.List[str],
    kernels: tp.List[str],
    mu: float,
    sigma: float,
) -> tp.Tuple[float, float, float]:
    """
    Perform simple grid search for finding optimal balance value for the whole dataset (_single value_ for all discretizations and noise/no noise cases.).

    Args:
        balance_values: tp.List[float]
            List with values of balance parameters considered to be a candidate to the optimal value.
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
    metrics = dict()
    for metric_type in _METRIC_TYPES:
        metrics[metric_type] = {balance_value: [] for balance_value in balance_values}

    # calculating metrics for each pair (image, kernel) for each balance value in the grid
    for image_path, kernel_path in tqdm(zip(gt_images, kernels)):
        image = imread(image_path)
        if image.ndim == 3:
            image = rgb2gray(image)
        image = center_crop(image, _IMAGE_SIZE, _IMAGE_SIZE)
        kernel = load_npy(kernel_path, key='psf')

        if image_path.endswith('png'):
            # ---- float ----
            blurred = convolve(image, kernel)
            noised_blurred = make_noised(blurred, mu=mu, sigma=sigma)
            _calc_metrics_in_blind_case(balance_values, metrics, blurred, noised_blurred, kernel, image)
            
            # ---- lin RGB uint16 ----
            image = float2linrgb16bit(image)
            blurred = float2linrgb16bit(blurred)
            noised_blurred = float2linrgb16bit(noised_blurred)
            _calc_metrics_in_blind_case(
                balance_values,
                metrics,
                (blurred / _MAX_UINT16).astype(np.float32),
                (noised_blurred / _MAX_UINT16).astype(np.float32),
                kernel,
                (image / _MAX_UINT16).astype(np.float32),
            )

            # ---- sRGB uint8 ----
            image = linrrgb2srgb8bit(image)
            blurred = linrrgb2srgb8bit(blurred)
            noised_blurred = linrrgb2srgb8bit(noised_blurred)
            _calc_metrics_in_blind_case(
                balance_values,
                metrics,
                (blurred / _MAX_UINT8).astype(np.float32),
                (noised_blurred / _MAX_UINT8).astype(np.float32),
                kernel,
                (image / _MAX_UINT8).astype(np.float32),
            )
        else:
            # ---- sRGB uint8 ----
            if image.dtype in [np.float16, np.float32, np.float64]:  # after rgb2gray uint8 might become float
                image = float2srgb8(image)
            blurred = convolve(image, kernel).astype(np.uint8)
            noised_blurred = make_noised(blurred, mu=mu, sigma=sigma).astype(np.uint8)
            _calc_metrics_in_blind_case(
                balance_values,
                metrics,
                (blurred / _MAX_UINT8).astype(np.float32),
                (noised_blurred / _MAX_UINT8).astype(np.float32),
                kernel,
                (image / _MAX_UINT8).astype(np.float32),
            )

    # finding average metrics for each balance value in the grid and the optimal values
    logging.info(f'BLIND NOISE: Finding optimal balance value')
    max_psnr, max_ssim = 0, 0
    optimal_balance = None
    for balance_value in balance_values:
        mean_psnr = np.mean(metrics[_PSNR_NAME][balance_value])
        mean_ssim = np.mean(metrics[_SSIM_NAME][balance_value])

        metrics[_PSNR_NAME][balance_value] = mean_psnr
        metrics[_SSIM_NAME][balance_value] = mean_ssim

        if (mean_psnr >= max_psnr and mean_ssim >= max_ssim) or (mean_psnr >= 1.05 * max_psnr) or (mean_ssim >= 1.05 * max_ssim):
            max_psnr, max_ssim = mean_psnr, mean_ssim
            optimal_balance = balance_value
            print(f'Optimal balance: {optimal_balance}; psnr: {max_psnr}, ssim: {max_ssim}')
    logging.info(f'Optimal balance value is {optimal_balance}')
    logging.info(f'The best metrics: ssim is {max_ssim}; psnr is {max_psnr}')


def _calc_metrics_in_blind_case(
    balance_values: tp.List[float],
    metrics: tp.List[float],
    blurred: np.array,
    noised_blurred: np.array,
    kernel: np.array,
    gt_image: np.array,
):
    for balance_value in balance_values:
        restored = wiener_gray(blurred, kernel, balance=balance_value, clip=True)
        metrics[_PSNR_NAME][balance_value].append(psnr(gt_image, restored))
        metrics[_SSIM_NAME][balance_value].append(ssim(gt_image, restored))

        restored_noised = wiener_gray(noised_blurred, kernel, balance=balance_value, clip=True)
        metrics[_PSNR_NAME][balance_value].append(psnr(gt_image, restored_noised))
        metrics[_SSIM_NAME][balance_value].append(ssim(gt_image, restored_noised))


def get_paths(benchmark_list_path: str) -> tp.Tuple[tp.List[str], tp.List[str]]:
    images_path = []
    kernels_path = []
    with open(benchmark_list_path, 'r+') as file:
        next(file)
        for line in tqdm(file):
            _, _, kernel_path, _, image_path = line.strip().split(',')
            images_path.append(image_path)
            kernels_path.append(kernel_path)
    return images_path, kernels_path


def main():
    logging.basicConfig(filename=os.path.join('results', 'wiener_tuning_results.log'), level=logging.INFO)

    config = OmegaConf.load(os.path.join('configs', 'config.yml'))

    benchmark_list_path = config.dataset.benchmark_list_path
    balance_values = config.wiener_tuning.balance_values
    images_path, kernels_path = get_paths(benchmark_list_path)

    mu = config.dataset.blur.mu
    sigma = config.dataset.blur.sigma

    logging.info(f'Tuning on {benchmark_list_path}')
    start_time = time()
    grid_search_non_blind_noise(
        balance_values_noise=balance_values.noise,
        balance_values_no_noise=balance_values.no_noise,
        gt_images=images_path,
        kernels=kernels_path,
        mu=mu,
        sigma=sigma,
    )
    grid_search_blind_noise(
        balance_values=balance_values.blind_noise,
        gt_images=images_path,
        kernels=kernels_path,
        mu=mu,
        sigma=sigma,
    )
    logging.info(f'Full time: {(time() - start_time) / 60} minutes')

if __name__ == '__main__':
    np.random.seed(8)
    main()