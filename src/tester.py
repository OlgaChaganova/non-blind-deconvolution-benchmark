import sqlite3
import typing as tp

import numpy as np
from tqdm import tqdm

from data.convertation import float2srgb8, srgb2linrgb16
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
        benchmark_list_path: str, 
        models: tp.List[tp.Tuple[dict, str]],
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

        with open(self._benchmark_list_path, 'r+') as file:
            next(file)
            for line in tqdm(file):
                blur_type, blur_dataset, kernel_path, image_dataset, image_path = line.strip().split(',')

                image = imread(image_path)
                kernel = load_npy(kernel_path, key='psf')
                image = crop2even(image)
                image = rgb2gray(image)
                blurred = convolve(image, kernel)
                noised_blurred = make_noised(blurred, self._data_config['blur']['mu'], sigma=self._data_config['blur']['sigma'])


                if '.png' in image_path:
                    self._run_models(
                        image=image, blurred=blurred, noised_blurred=noised_blurred, kernel=kernel, blur_type=blur_type, 
                        blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
                        dicretization='float',
                    )

                    blurred = float2srgb8(blurred)
                    noised_blurred = float2srgb8(noised_blurred)
                    self._run_models(
                        image=image, blurred=(blurred / 255), noised_blurred=noised_blurred, kernel=kernel, blur_type=blur_type, 
                        blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
                        dicretization='srgb_8bit',
                    )

                    blurred = srgb2linrgb16(blurred)
                    noised_blurred = srgb2linrgb16(noised_blurred)
                    self._run_models(
                        image=image, blurred=(blurred / 65535), noised_blurred=noised_blurred, kernel=kernel, blur_type=blur_type, 
                        blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
                        dicretization='linrgb_16bit',
                    )

                else:
                    self._run_models(
                        image=image, blurred=(blurred / 255), noised_blurred=noised_blurred, kernel=kernel, blur_type=blur_type, 
                        blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
                        dicretization='srgb_8bit',
                    )
            cursor.close()

    def _run_models(
        self,
        image: np.array,
        blurred: np.array,
        noised_blurred: np.array,
        kernel: np.array,

        blur_type: str,
        blur_dataset: str,
        kernel_path: str,
        image_dataset: str,
        image_path: str,
        dicretization: str,

        cursor: tp.Any,
        connection: tp.Any,
    ):
        insert_query = f'''INSERT INTO {self._table_name}
                        (blur_type, blur_dataset, kernel, image_dataset, image, discretization, noised, model, ssim, psnr) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''
        
        for model, model_name in self._models:
            if self._model_config[model_name][blur_type]:                
                # no noise
                metrcics = self._calculate_metrics(model['no_noise'], model_name, image, blurred, kernel)
                cursor.execute(
                    insert_query,
                    (blur_type, blur_dataset, kernel_path, image_dataset, image_path, dicretization, False, model_name, metrcics['ssim'], metrcics['psnr']),
                )
                connection.commit()
                
                # with noise
                metrcics = self._calculate_metrics(model['noise'], model_name, image, noised_blurred, kernel)
                cursor.execute(
                    insert_query,
                    (blur_type, blur_dataset, kernel_path, image_dataset, image_path, dicretization, True, model_name, metrcics['ssim'], metrcics['psnr']),
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
