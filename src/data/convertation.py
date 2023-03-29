import numpy as np


def srgbf_to_linrgbf(image: np.array) -> np.array:
    """Conver image from float sRGB to float linRGB"""
    assert image.dtype in [np.float16, np.float32, np.float64], f'Image dtype must be np.float(16/32/64), but got {image.dtype}'
    assert image.max() <= 1, f'Image max must be <= 1, but got {image.max()}'
    assert image.min() >= 0, f'Image min must be >= 0, but got {image.min()}'
    mask = image <= 0.04045
    image[mask] = image[mask] / 12.92
    image[np.invert(mask)] = ((image[np.invert(mask)] + 0.055) / 1.055) ** 2.4
    return image


def float_to_uint16(image: np.array) -> np.array:
    """Convert image from float (lin / sRGB) to 16 bit (lin / sRGB)"""
    assert image.dtype in [np.float16, np.float32, np.float64], f'Image dtype must be np.float(16/32/64), but got {image.dtype}'
    assert image.max() <= 1, f'Image max must be <= 1, but got {image.max()}'
    assert image.min() >= 0, f'Image min must be >= 0, but got {image.min()}'
    max_uint16 = 65535
    return (image * max_uint16).astype(np.uint16)


def linrrgb16_to_srgb8(image: np.array) -> np.array:
    """Convert image from 16 bit Linear RGB 16 bit to SRGB 8 bit """
    assert image.dtype == np.uint16, f'Image dtype must be np.uint16, but got {image.dtype}'
    max_uint16 = 65535
    max_uint8 = 255
    image = image / max_uint16
    mask = image <= 0.0031308
    image[mask] = image[mask] * 12.92
    image[np.invert(mask)] = 1.055 * (image[np.invert(mask)] ** (1/2.4)) - 0.055
    return (image  * max_uint8).astype(np.uint8)


def uint16_to_uint8(image: np.array) -> np.array:
    """Convert image from 16 bit (lin / sRGB) to 8 bit (lin / sRGB)"""
    assert image.dtype == np.uint16, f'Image dtype must be np.uint16, but got {image.dtype}'
    max_uint16 = 65535
    max_uint8 = 255
    return ((image / max_uint16) * max_uint8).astype(np.uint8)
