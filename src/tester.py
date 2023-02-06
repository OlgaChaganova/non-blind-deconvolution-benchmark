import itertools
import logging
import os
import sqlite3
import typing as tp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.convolution import convolve
from deconv.neural.usrnet.predictor import USRNetPredictor
from deconv.neural.dwdn.predictor import DWDNPredictor
from deconv.neural.kerunc.predictor import KerUncPredictor
from deconv.neural.rgdn.predictor import RGDNPredictor
from imutils import imread, rgb2gray, load_npy, make_noised, gray2gray3d, crop2even
from metrics import psnr, ssim


class Tester(object):
    def __init__(
        self,
        is_full: bool,
        models: tp.List[tp.Tuple[tp.Union[USRNetPredictor, DWDNPredictor, KerUncPredictor, RGDNPredictor, tp.Callable], str]],
        db_path: str,
        table_name: str,
        model_config: dict,
    ):
        self._is_full =  is_full
        self._models = models
        self._db_path = db_path
        self._table_name = table_name
        self._model_config = model_config
        self._kernels = {}
        self._gt_images = {}

        self._prepare() 

    def test(self):
        connection = sqlite3.connect(self._db_path)
        cursor = connection.cursor()
        insert_query = f'''INSERT INTO {self._table_name}
            (blur_type, blur_dataset, kernel, image_dataset, image, discretization, noised, model, ssim, psnr) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''

        for blur_type in tqdm(self._kernels.keys()):
            for image_dataset in self._gt_images.keys():
                image_kernel_pairs = list(itertools.product(self._kernels[blur_type], self._gt_images[image_dataset]))  # сartesian product
                for pair in image_kernel_pairs:
                    kernel = pair[0][0]
                    kernel_path = pair[0][1]
                    blur_dataset = kernel_path.split(os.sep)[-2]
                    image = pair[1][0]
                    image_path = pair[1][1]

                    image = crop2even(image)
                    image = rgb2gray(image)
                    blurred = convolve(image, kernel)
                    noised_blurred = make_noised(blurred, mu=0, sigma=0.01)

                    for model, model_name in self._models:
                        if self._model_config[model_name][blur_type]:                
                            try:
                                # no noise
                                metrcics = self._calculate_metrics(model, model_name, image, blurred, kernel)
                                cursor.execute(
                                    insert_query,
                                    (blur_type, blur_dataset, kernel_path, image_dataset, image_path, 'float32', False, model_name, metrcics['ssim'], metrcics['psnr']),
                                )
                                connection.commit()
                                
                                # with noise
                                metrcics = self._calculate_metrics(model, model_name, image, noised_blurred, kernel)
                                cursor.execute(
                                    insert_query,
                                    (blur_type, blur_dataset, kernel_path, image_dataset, image_path, 'float32', True, model_name, metrcics['ssim'], metrcics['psnr']),
                                )
                                connection.commit()

                            except RuntimeError:
                                logging.error(f'{model_name} can not be test on the image {image_path} with shape: {image.shape}')
                                continue
        
    def _prepare(self):  # HARD-HARD-HARDCODE
        assert self._is_full == False

        if not self._is_full:
            self._kernels['motion_blur'] = list(Path('datasets/kernels/motion-blur/processed/Levin').rglob('*.npy'))[:1] +\
                                list(Path('datasets/kernels/motion-blur/processed/Sun').rglob('*.npy'))[:1] + \
                                list(Path('datasets/kernels/motion-blur/processed/synthetic').rglob('*.npy'))[:1]
            self._kernels['gauss_blur'] = list(Path('datasets/kernels/gauss-blur/processed/synthetic').rglob('*.npy'))[:1]
            self._kernels['eye_blur'] = list(Path('datasets/kernels/eye-psf/processed/synthetic').rglob('*.npy'))[:1]

            self._gt_images['bsds300'] = list(Path('datasets/gt/BSDS300').rglob('*.jpg'))[:1]
            self._gt_images['sun'] = list(Path('datasets/gt/Sun-gray').rglob('*.png'))[:1]

            for blur_type in self._kernels.keys():
                self._kernels[blur_type] = list(map(lambda path: (load_npy(path, 'psf'), str(path)), self._kernels[blur_type]))

            for image_dataset in self._gt_images.keys():
                self._gt_images[image_dataset] = list(map(lambda path: (imread(path), str(path)), self._gt_images[image_dataset]))

        logging.info('Everything is ready for the test to begin.')
    

    def _calculate_metrics(
            self,
            model: tp.Union[USRNetPredictor, DWDNPredictor, KerUncPredictor, RGDNPredictor, tp.Callable],
            model_name: str,
            image: np.array, 
            blurred: np.array,
            kernel: np.array,
        ) -> dict:
        blurred_3d = gray2gray3d(blurred)

        restored = (
            model(blurred_3d, kernel)[..., 0]
            if model_name in ['usrnet', 'dwdn']
            else model(blurred, kernel)
        )

        return {
            'ssim': ssim(image, restored),
            'psnr': psnr(image, restored)
        }