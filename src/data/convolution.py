import logging
import typing as tp

import numpy as np
import torch


def fft(inp: np.array, shape: tuple = None) -> np.array:
    if shape is None:
        shape = inp.shape
    return np.fft.fft2(inp, s=shape, norm='ortho')


def ifft(inp: np.array, shape: tuple = None) -> np.array:
    if shape is None:
        shape = inp.shape
    return np.fft.ifft2(inp, s=shape, norm='ortho')

def shift(inp: np.array, inverse: bool = False) -> np.array:
    if inverse:
        return np.fft.ifftshift(inp)
    else:
        return np.fft.fftshift(inp)


def fft_conv(image: np.array, psf: np.array, scale_output: bool = True) -> np.array:
    if image.shape != psf.shape:
        sz = (image.shape[0] - psf.shape[0], image.shape[1] - psf.shape[1])
        psf = np.pad(
            psf,
            (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)),
            'constant'
        )
    f1 = fft(image)
    f2 = fft(psf, shape=image.shape)
    convolved = ifft(f1 * f2, shape=image.shape)
    res = np.abs(convolved)
    if scale_output:
        res = res * np.sum(image) / np.sum(res)
    return shift(res)


def convolve(
    image: np.array,
    psf: np.array,
    scale_output: bool = True,
    data_type: tp.Any = np.float32
) -> np.array:
    """Convolve multichannel images using FFT."""

    if not np.allclose(psf.sum(), 1, rtol=1e-2, atol=1e-2):
        psf = psf / psf.sum()
        logging.warning('PSF has sum more than 1. Normed')
    
    if np.ceil(image.max()) == 255:
        image = image / image.max()
        logging.warning('Image pixels are not in 0..1. Normed')

    ndim = image.ndim

    if ndim == 2:
        return fft_conv(image, psf, scale_output).astype(data_type)

    elif ndim == 3:
        res = []
        for i in range(image.shape[-1]):
            res.append(fft_conv(image[..., i], psf, scale_output=scale_output))
        res = np.stack(res)
        return np.transpose(res, (1, 2, 0)).astype(data_type)

    else:
        raise ValueError(f'Image must be 2- or 3-dimencional but got {ndim}-dimencional image.')
