import numpy as np
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity


def psnr(image: np.array, restored: np.array, **psnr_params) -> float:
    return peak_signal_noise_ratio(image, restored, **psnr_params)


def ssim(image: np.array, restored: np.array, **ssim_params) -> float:
    return structural_similarity(image, restored, **ssim_params)


def mse(image: np.array, restored: np.array) -> float:
    return mean_squared_error(image, restored)


def rmse(image: np.array, restored: np.array) -> float:
    return mse(image, restored) ** 0.5
