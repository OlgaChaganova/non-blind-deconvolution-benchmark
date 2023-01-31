"""Utils for images."""

import typing as tp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage


def imread(img_path: str) -> np.array:
    img_path = str(img_path) if not isinstance(img_path, str) else img_path
    image = cv2.imread(img_path)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (image / 255).astype(np.float32)


def imshow(image: np.array, figsize: tp.Tuple[int, int], title: tp.Optional[str] = None):
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