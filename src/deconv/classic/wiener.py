import logging

import numpy as np
from skimage import restoration


def wiener_gray(blurred_image: np.array, psf: np.array, clip: bool, **algo_params) -> np.array:
    """Apply Wiener deconvolution for one-channel image.

    Parameters
    ----------
    blurred_image : np.array
        Blurred one-channel image.
    psf : np.array
        PSF.
    Returns
    -------
    np.array
        Restored one-channel image.
    """
    if not np.allclose(psf.sum(), 1, rtol=1e-2, atol=1e-2):
        psf = psf / psf.sum()
        logging.warning('PSF has sum more than 1. Normed')
    restored = restoration.wiener(image=blurred_image, psf=psf, **algo_params)
    if clip:
        return np.clip(restored / restored.max(), 0, 1)
    return (restored / restored.max()).astype(np.float32)


def wiener_rgb(blurred_image: np.array, psf: np.array, clip: bool, **algo_params) -> np.array:
    """Apply Wiener deconvolution for RGB image per channel.
    
    Parameters
    ----------
    blurred_image : np.array
        Blurred RGB image.
    psf : np.array
        PSF.
    Returns
    -------
    np.array
        Restored RGB image.
    """
    rgb_restored = []
    for i in range(blurred_image.shape[-1]):
        rgb_restored.append(wiener_gray(blurred_image[:, :, i], psf, clip, **algo_params))
    rgb_restored = np.stack(rgb_restored)
    return np.transpose(rgb_restored, (1, 2, 0))
