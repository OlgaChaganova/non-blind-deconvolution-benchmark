"""Script for tuning Wiener parameters (balance) in case of non-blind noise scenario"""

import argparse
import logging
import os
import typing as tp
from time import time

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm 

from constants import IMAGE_SIZE, MAX_UINT16, MAX_UINT8
from deconv.classic.wiener.wiener import wiener_gray
from data.convertation import srgbf_to_linrgbf, uint8_to_float32, float_to_uint16, linrrgb16_to_srgb8 #float2linrgb16bit, float2srgb8, linrrgb2srgb8bit
from data.convolution import convolve
from imutils import make_noised, imread, load_npy, srgb2gray, center_crop
from metrics import psnr, ssim


_NOISE_NAME = 'noise'
_NO_NOISE_NAME = 'no_noise'
_NOISED_TYPES = [_NOISE_NAME, _NO_NOISE_NAME]

_SSIM_NAME = 'ssim'
_PSNR_NAME = 'psnr'
_METRIC_TYPES = [_SSIM_NAME, _PSNR_NAME]


def parse():
    parser = argparse.ArgumentParser(
        description='Parser for nbd models testing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--blur_type',
        type=str,
        choices=['gauss_blur', 'motion_blur', 'eye_blur', 'all'],
        default='all',
        help="Blur type to be tuned."
    )
    return parser.parse_args()


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
        image = center_crop(image, IMAGE_SIZE, IMAGE_SIZE)

        if image_path.endswith('.jpg'):  # sRGB 8 bit
            image = uint8_to_float32(image)  # sRGB float32
        if image.ndim == 3:
            image = srgb2gray(image)
        image = srgbf_to_linrgbf(image)  # convert from float sRGB to float linRGB
        kernel = load_npy(kernel_path, key='psf')

        # ---- linRGB float ----
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
            type='linrgb_float',
        )
        
        # ---- lin RGB uint16 ----
        image = float_to_uint16(image)
        blurred = float_to_uint16(blurred)
        noised_blurred = float_to_uint16(noised_blurred)
        _calc_metrics_in_nonblind_case(
            balance_values_no_noise,
            balance_values_noise,
            metrics,
            (blurred / MAX_UINT16).astype(np.float32),
            (noised_blurred / MAX_UINT16).astype(np.float32),
            kernel,
            (image / MAX_UINT16).astype(np.float32),
            type='linrgb_16bit',
        )

        # ---- sRGB uint8 ----
        image = linrrgb16_to_srgb8(image)
        blurred = linrrgb16_to_srgb8(blurred)
        noised_blurred = linrrgb16_to_srgb8(noised_blurred)
        _calc_metrics_in_nonblind_case(
            balance_values_no_noise,
            balance_values_noise,
            metrics,
            (blurred / MAX_UINT8).astype(np.float32),
            (noised_blurred / MAX_UINT8).astype(np.float32),
            kernel,
            (image / MAX_UINT8).astype(np.float32),
            type='srgb_8bit',
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
    type: tp.Literal['linrgb_float', 'linrgb_16bit', 'srgb_8bit'],
):
    if type == 'linrgb_float':
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
        image = center_crop(image, IMAGE_SIZE, IMAGE_SIZE)

        if image_path.endswith('.jpg'):  # sRGB 8 bit
            image = uint8_to_float32(image)  # sRGB float32
        if image.ndim == 3:
            image = srgb2gray(image)
        image = srgbf_to_linrgbf(image)  # convert from float sRGB to float linRGB

        kernel = load_npy(kernel_path, key='psf')

        # ---- linRGB float ----
        blurred = convolve(image, kernel)
        noised_blurred = make_noised(blurred, mu=mu, sigma=sigma)
        _calc_metrics_in_blind_case(balance_values, metrics, blurred, noised_blurred, kernel, image)
        
        # ---- lin RGB 16 bit ----
        image = float_to_uint16(image)
        blurred = float_to_uint16(blurred)
        noised_blurred = float_to_uint16(noised_blurred)
        _calc_metrics_in_blind_case(
            balance_values,
            metrics,
            (blurred / MAX_UINT16).astype(np.float32),
            (noised_blurred / MAX_UINT16).astype(np.float32),
            kernel,
            (image / MAX_UINT16).astype(np.float32),
        )

        # ---- sRGB uint8 ----
        image = linrrgb16_to_srgb8(image)
        blurred = linrrgb16_to_srgb8(blurred)
        noised_blurred = linrrgb16_to_srgb8(noised_blurred)
        _calc_metrics_in_blind_case(
            balance_values,
            metrics,
            (blurred / MAX_UINT8).astype(np.float32),
            (noised_blurred / MAX_UINT8).astype(np.float32),
            kernel,
            (image / MAX_UINT8).astype(np.float32),
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


def get_paths(
    benchmark_list_path: str,
    target_blur_type: tp.Literal['gauss_blur', 'eye_blur', 'motion_blur', 'all'],
) -> tp.Tuple[tp.List[str], tp.List[str]]:
    images_path = []
    kernels_path = []
    with open(benchmark_list_path, 'r+') as file:
        next(file)
        for line in file:
            blur_type, _, kernel_path, _, image_path = line.strip().split(',')
            if target_blur_type == 'all':
                images_path.append(image_path)
                kernels_path.append(kernel_path)
            else:
                if blur_type == target_blur_type:
                    images_path.append(image_path)
                    kernels_path.append(kernel_path)   
    return images_path, kernels_path


def main():
    args = parse()
    logging.basicConfig(filename=os.path.join('results', 'wiener_tuning_results.log'), level=logging.INFO)

    config = OmegaConf.load(os.path.join('configs', 'config.yml'))

    benchmark_list_path = config.dataset.benchmark_list_path
    balance_values = config.wiener_tuning.balance_values
    images_path, kernels_path = get_paths(benchmark_list_path, target_blur_type=args.blur_type)

    mu = config.dataset.blur.mu
    sigma = config.dataset.blur.sigma

    logging.info(f'Tuning on {benchmark_list_path}; blur type: {args.blur_type.upper()}')
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