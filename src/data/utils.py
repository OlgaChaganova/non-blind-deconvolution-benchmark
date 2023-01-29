import typing as tp

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(img_path: str) -> np.array:
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

