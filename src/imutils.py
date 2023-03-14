"""Utils for images."""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import skimage


def imread(img_path: str) -> np.array:
    """Return float image if png and sRGB uint8 if jpg"""
    img_path = str(img_path) if not isinstance(img_path, str) else img_path
    return plt.imread(img_path)


def imshow(image: np.array, figsize: tp.Optional[tp.Tuple[int, int]] = None, title: tp.Optional[str] = None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()


def imsshow(images: tp.List[np.array], figsize: tp.Tuple[int, int], titles: tp.Optional[tp.List[str]] = None):
    """Show several images."""
    _, axs = plt.subplots(1, len(images), figsize=figsize)
    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].axis('off')
        if titles is not None:
            axs[i].set_title(titles[i])


def rgb2gray(image: np.array) -> np.array:
    """Convert RGB to gray image.

    Parameters
    ----------
    image : np.array
        RGB image (3 channels). Shape: [height, width, 3]

    Returns
    -------
    np.array
        Gray image. Shape: [height, width]
    """
    return skimage.color.rgb2gray(image)


def gray2gray3d(image: np.array) -> np.array:
    """Convert gray image to pseudo RGB image by replicating gray image.

    Parameters
    ----------
    image : np.array
        Gray image. Shape: [height, width]

    Returns
    -------
    np.array
        "RGB" image (3 channels). Shape: [height, width, 3]
    """
    assert image.ndim == 2
    return np.stack([image] * 3, axis=-1)


def make_noised(image: np.array, mu: float, sigma: float) -> np.array:
    return (image + (sigma * np.random.randn(*image.shape) + mu)).astype(np.float32)


def crop2even(image: np.array) -> np.array:
    """Crop image to even sizes. Used for DWDN"""
    if image.ndim == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    if h % 2 != 0:
        image = image[:-1, ...]
    if w % 2 != 0:
        image = image[:, :-1, ...]

    return image


def load_npy(filename: str, key: tp.Optional[str] = None):
    arr = np.load(filename, allow_pickle=True).item()
    if key is not None:
        return arr.get(key)
    return arr


def center_crop(img: np.array, new_width: int = None, new_height: int = None) -> np.array:        
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        return img[top:bottom, left:right]
    return img[top:bottom, left:right, ...]