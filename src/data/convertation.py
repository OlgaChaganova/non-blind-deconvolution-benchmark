import numpy as np
import numpy.typing as npt

from constants import MAX_UINT8, MAX_UINT16

def srgbf_to_linrgbf(image: npt.NDArray[np.float_], eps: float = 1e-3) -> npt.NDArray[np.float32]:
    """Conver image from float sRGB to float linRGB"""
    assert image.dtype in [np.float16, np.float32, np.float64], f'Image dtype must be np.float(16/32/64), but got {image.dtype}'
    assert image.max() <= (1 + eps), f'Image max must be <= 1, but got {image.max()}'
    assert image.min() >= (0 - eps), f'Image min must be >= 0, but got {image.min()}'
    mask = image <= 0.04045
    image_lin = image.copy()
    image_lin[mask] = image_lin[mask] / 12.92
    image_lin[np.invert(mask)] = ((image_lin[np.invert(mask)] + 0.055) / 1.055) ** 2.4
    return image_lin.astype(np.float32)


def linrgbf_to_srgbf(image: npt.NDArray[np.float_], eps: float = 1e-3) -> npt.NDArray[np.float32]:
    """Conver image from float linRGB to float sRGB"""
    assert image.dtype in [np.float16, np.float32, np.float64], f'Image dtype must be np.float(16/32/64), but got {image.dtype}'
    assert image.max() <= (1 + eps), f'Image max must be <= 1, but got {image.max()}'
    assert image.min() >= (0 - eps), f'Image min must be >= 0, but got {image.min()}'
    mask = image <= 0.0031308
    image_srgb = image.copy()
    image_srgb[mask] = image_srgb[mask] * 12.92
    image_srgb[np.invert(mask)] = 1.055 * (image_srgb[np.invert(mask)] ** (1/2.4)) - 0.055
    return image_srgb.astype(np.float32)


def float_to_uint16(image: npt.NDArray[np.float_], eps: float = 1e-3) -> npt.NDArray[np.uint16]:
    """Convert image from float (lin / sRGB) to 16 bit (lin / sRGB)"""
    assert image.dtype in [np.float16, np.float32, np.float64], f'Image dtype must be np.float(16/32/64), but got {image.dtype}'
    assert image.max() <= (1 + eps), f'Image max must be <= 1, but got {image.max()}'
    assert image.min() >= (0 - eps), f'Image min must be >= 0, but got {image.min()}'
    return (image * MAX_UINT16).astype(np.uint16)


def linrrgb16_to_srgb8(image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
    """Convert image from 16 bit Linear RGB 16 bit to SRGB 8 bit """
    assert image.dtype == np.uint16, f'Image dtype must be np.uint16, but got {image.dtype}'
    image_srgb = image.copy()
    image_srgb = image_srgb / MAX_UINT16
    mask = image_srgb <= 0.0031308
    image_srgb[mask] = image_srgb[mask] * 12.92
    image_srgb[np.invert(mask)] = 1.055 * (image_srgb[np.invert(mask)] ** (1/2.4)) - 0.055
    return (image_srgb  * MAX_UINT8).astype(np.uint8)


def uint16_to_uint8(image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
    """Convert image from 16 bit (lin / sRGB) to 8 bit (lin / sRGB)"""
    assert image.dtype == np.uint16, f'Image dtype must be np.uint16, but got {image.dtype}'
    return ((image / MAX_UINT16) * MAX_UINT8).astype(np.uint8)


def uint8_to_float32(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Convert image from 8 bit (lin / sRGB) to float32 (lin / sRGB)"""
    return (image / MAX_UINT8).astype(np.float32)
