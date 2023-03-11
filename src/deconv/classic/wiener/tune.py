"""Script for tuning Wiener parameters (balance) in case on non-blind noise scenario"""

import typing as tp

from deconv.classic.wiener.wiener import wiener_gray
from data.convolution import convolve
from metrics import psnr, ssim


def grid_search(
    balance_values: tp.List[float],
    gt_images: tp.List[np.array],
    kernels: tp.List[np.array],
) -> tp.Tuple[float, float, float]:
    """
    Perform simple grid search for finding optimal balance value.

    Args:
        balance_values: tp.List[float]
            List with values of balance parameters considered to be candidate to the optimal value.
        gt_images: tp.List[np.array]
            List with ground truth (sharp) images.
        kernels: tp.List[np.array]
            List with kernels.
    Returns:
        tp.Tuple[float, float, float]
        Optimal balance value and corresponding ssim and psnr values.
    """
    psnr_per_balance = dict()
    ssim_per_balance = dict()
    for balance_value in balance_values:
        psnr_per_balance[balance_value] = []
        ssim_per_balance[balance_value] = []

    # calculating metrics for each pair (image, kernel) for each balance value in the grid
    for gt_image, kernel in zip(gt_images, kernels):
        blurred = convolve(gt_image, kernel)
        for balance_value in balance_values:
            restored = wiener_gray(blurred, kernel, balance=balance_value)
            psnr_per_balance[balance_value].append(psnr(gt_image, restored))
            ssim_per_balance[balance_value].append(ssim(gt_image, restored))

    # finding average metrics for each balance value in the grid
    for balance_value in balance_values:
        psnr_per_balance[balance_value] = mean(psnr_per_balance[balance_value])
        ssim_per_balance[balance_value] = mean(ssim_per_balance[balance_value])

    # finding maximum metrics values and corresponding balance value
    max_psnr, max_ssim = 0, 0
    optimal_balance = None
    for balance_value in balance_values:
        if psnr_per_balance[balance_value] >= max_psnr and ssim_per_balance[balance_value] >= max_ssim:
            max_psnr, max_ssim
        psnr_per_balance[balance_value] = mean(psnr_per_balance[balance_value])
        ssim_per_balance[balance_value] = mean(ssim_per_balance[balance_value])