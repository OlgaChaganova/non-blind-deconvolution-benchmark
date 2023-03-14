import numpy as np


def float2srgb8(image: np.array) -> np.array:
    """Conver float image (png) to 8-bit sRGB"""
    assert image.dtype in [np.float16, np.float32, np.float64]
    return (image * 255).astype(np.uint8)


def float2linrgb16bit(image: np.array, gamma: float = 2.2) -> np.array:
    """Convert image in float to Linear RGB 16 bit"""
    assert image.dtype in [np.float16, np.float32, np.float64]
    max_uint16 = 65535
    return (image ** gamma * max_uint16).astype(np.uint16)


def linrrgb2srgb8bit(image: np.array) -> np.array:
    """Convert image from 16 bit Linear RGB 16 bit to SRGB 8 bit """
    assert image.dtype == np.uint16
    max_uint16 = 65535
    max_uint8 = 255
    image = image / max_uint16
    mask = image <= 0.0031308
    image[mask] = image[mask] * 12.92
    image[np.invert(mask)] = 1.055 * (image[np.invert(mask)] ** (1/2.4)) - 0.055
    return (image  * max_uint8).astype(np.uint8)
