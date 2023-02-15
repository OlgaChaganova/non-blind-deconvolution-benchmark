import numpy as np


def float2srgb8(image: np.array) -> np.array:
    """Conver float image (png) to 8-bit sRGB"""
    assert image.dtype in [np.float32, np.float64]
    return (image * 255).astype(np.uint8)


def srgb2linrgb16(image_srgb: np.array) -> np.array:
    """Convert 8-bit sRGB to 16-bit linRGB"""
    
    if image_srgb.dtype == np.uint8:
        image_srgb = image_srgb / 255
        
    mask = image_srgb <= 0.04045

    image_srgb[mask] = image_srgb[mask] / 12.92
    image_srgb[np.invert(mask)] = np.power((image_srgb[np.invert(mask)] + 0.055) / 1.055, 2.4)
    return (image_srgb * 65535).astype(np.uint16)
