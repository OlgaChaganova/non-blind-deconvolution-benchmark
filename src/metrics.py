import numpy as np
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity


def psnr(image1: np.array, image2: np.array, **psnr_params) -> float:
    return peak_signal_noise_ratio(image1, image2, **psnr_params)


def ssim(image1: np.array, image2: np.array, **ssim_params) -> float:
    return structural_similarity(image1, image2, **ssim_params)


def mse(image1: np.array, image2: np.array) -> float:
    return mean_squared_error(image1, image2)


def rmse(image1: np.array, image2: np.array) -> float:
    return mse(image1, image2) ** 0.5