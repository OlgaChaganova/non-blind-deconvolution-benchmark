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
from imutils import imread, rgb2gray, load_npy, make_noised, gray2gray3d, center_crop
from metrics import psnr, ssim


class Tester(object):
    def __init__(
        self,
        benchmark_list_path: str, 
        models: tp.List[tp.Tuple[tp.Union[USRNetPredictor, DWDNPredictor, KerUncPredictor, RGDNPredictor, tp.Callable], str]],
        db_path: str,
        table_name: str,
        model_config: dict,
        data_config: dict,
    ):
        self._benchmark_list_path = benchmark_list_path
        self._models = models
        self._db_path = db_path
        self._table_name = table_name
        self._model_config = model_config
        self._data_config = data_config

    def test(self):
        connection = sqlite3.connect(self._db_path)
        cursor = connection.cursor()
        insert_query = f'''INSERT INTO {self._table_name}
            (blur_type, blur_dataset, kernel, image_dataset, image, discretization, noised, model, ssim, psnr) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''

        with open(self._benchmark_list_path, 'r+') as file:
            next(file)
            for line in tqdm(file):
                blur_type, blur_dataset, kernel_path, image_dataset, image_path = line.strip().split(',')
                kernel = load_npy(kernel_path, key='psf')
                image = imread(image_path)
                image = center_crop(image, 256, 256)
                image = rgb2gray(image)
                blurred = convolve(image, kernel)
                noised_blurred = make_noised(blurred, self._data_config['blur']['mu'], sigma=self._data_config['blur']['sigma'])

                for model, model_name in self._models:
                    if self._model_config[model_name][blur_type]:
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