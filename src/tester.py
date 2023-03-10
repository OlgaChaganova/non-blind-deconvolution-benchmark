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
from imutils import imread, rgb2gray, load_npy, make_noised, gray2gray3d, center_crop
from metrics import psnr, ssim


_MAX_UINT8 = 2 ** 8 - 1
_MAX_UINT16 = 2 ** 16 - 1
_IMAGE_SIZE = 256  # required image size 


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
                image = center_crop(image, _IMAGE_SIZE, _IMAGE_SIZE)
                if image.ndim == 3:
                    image = rgb2gray(image)

                if '.png' in image_path:
                    # ---- float ----
                    blurred = convolve(image, kernel)
                    noised_blurred = make_noised(blurred, mu=self._data_config['blur']['mu'], sigma=self._data_config['blur']['sigma'])
                    blurred_3d = gray2gray3d(blurred)
                    noised_blurred_3d = make_noised(blurred_3d, mu=self._data_config['blur']['mu'], sigma=self._data_config['blur']['sigma'])
                    blurred_images = {
                        'no_noise': {'1d': blurred, '3d': blurred_3d},
                        'noise': {'1d': noised_blurred, '3d': noised_blurred_3d},
                    }

                    self._run_models(
                        image=image, blurred_images=blurred_images, kernel=kernel, blur_type=blur_type, 
                        blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
                        dicretization='float',
                    )

                    # ---- uint8 ----
                    image = float2srgb8(image)
                    blurred = float2srgb8(blurred)
                    noised_blurred = float2srgb8(noised_blurred)
                    blurred_3d = float2srgb8(blurred_3d)
                    noised_blurred_3d = float2srgb8(noised_blurred_3d)

                    blurred_images = {
                        'no_noise': {'1d': (blurred / _MAX_UINT8).astype(np.float32), '3d': (blurred_3d / _MAX_UINT8).astype(np.float32)},
                        'noise': {'1d': (noised_blurred / _MAX_UINT8).astype(np.float32), '3d': (noised_blurred_3d / _MAX_UINT8).astype(np.float32)},
                    }

                    self._run_models(
                        image=(image / _MAX_UINT8).astype(np.float32), blurred_images=blurred_images, kernel=kernel, blur_type=blur_type, 
                        blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
                        dicretization='srgb_8bit',
                    )

                    # ---- uint16 ----
                    image = srgb2linrgb16(image)
                    blurred = srgb2linrgb16(blurred)
                    noised_blurred = srgb2linrgb16(noised_blurred)
                    blurred_3d = srgb2linrgb16(blurred_3d)
                    noised_blurred_3d = srgb2linrgb16(noised_blurred_3d)

                    blurred_images = {
                        'no_noise': {'1d': (blurred / _MAX_UINT16).astype(np.float32), '3d': (blurred_3d / _MAX_UINT16).astype(np.float32)},
                        'noise': {'1d': (noised_blurred / _MAX_UINT16).astype(np.float32), '3d': (noised_blurred_3d / _MAX_UINT16).astype(np.float32)},
                    }

                    self._run_models(
                        image=(image / _MAX_UINT16).astype(np.float32), blurred_images=blurred_images, kernel=kernel, blur_type=blur_type,
                        blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
                        dicretization='linrgb_16bit',
                    )

                else:
                    # ---- uint8 ----
                    if image.dtype in [np.float16, np.float32, np.float64]:  # after rgb2gray uint8 might become float
                        image = float2srgb8(image)

                    blurred = convolve(image, kernel).astype(np.uint8)
                    noised_blurred = make_noised(blurred, self._data_config['blur']['mu'], sigma=self._data_config['blur']['sigma']).astype(np.uint8)
                    blurred_3d = gray2gray3d(blurred)
                    noised_blurred_3d = make_noised(blurred_3d, mu=self._data_config['blur']['mu'], sigma=self._data_config['blur']['sigma']).astype(np.uint8)
                    blurred_images = {
                        'no_noise': {'1d': (blurred / _MAX_UINT8).astype(np.float32), '3d': (blurred_3d / _MAX_UINT8).astype(np.float32)},
                        'noise': {'1d': (noised_blurred / _MAX_UINT8).astype(np.float32), '3d': (noised_blurred_3d / _MAX_UINT8).astype(np.float32)},
                    }

                    self._run_models(
                        image=(image / _MAX_UINT8).astype(np.float32), blurred_images=blurred_images, kernel=kernel, blur_type=blur_type, 
                        blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
                        dicretization='srgb_8bit',
                    )
            cursor.close()

    def _run_models(
        self,
        image: np.array,
        blurred_images: dict,
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
                metrcics = self._calculate_metrics(model['no_noise'], model_name, image, blurred_images['no_noise'], kernel)
                cursor.execute(
                    insert_query,
                    (blur_type, blur_dataset, kernel_path, image_dataset, image_path, dicretization, False, model_name, metrcics['ssim'], metrcics['psnr']),
                )
                connection.commit()
                
                # with noise
                metrcics = self._calculate_metrics(model['noise'], model_name, image, blurred_images['noise'], kernel)
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
            blurred_images: dict,
            kernel: np.array,
        ) -> dict:
        restored = (
            model(blurred_images['3d'], kernel)[..., 0]
            if model_name in ['usrnet', 'dwdn']
            else model(blurred_images['1d'], kernel)
        )

        return {
            'ssim': ssim(image, restored),
            'psnr': psnr(image, restored)
        }
