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



def imshow(image: np.array, figsize: tp.Tuple[int, int]):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
