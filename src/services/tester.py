import sqlite3
import typing as tp

import numpy as np
from tqdm import tqdm

from constants import MAX_UINT16, MAX_UINT8, IMAGE_SIZE
from data.convertation import srgbf_to_linrgbf, float_to_uint16, linrrgb16_to_srgb8, uint16_to_uint8, uint8_to_float32
from data.convolution import convolve
from deconv.neural.usrnet.predictor import USRNetPredictor
from deconv.neural.dwdn.predictor import DWDNPredictor
from deconv.neural.kerunc.predictor import KerUncPredictor
from deconv.neural.rgdn.predictor import RGDNPredictor
from imutils import imread, srgb2gray, load_npy, make_noised, gray2gray3d, center_crop
from metrics import psnr, ssim


class BaseTester(object):
    """Tester for models evaluation in different scenarios (discretization type and noise)."""
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
        ...

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
        discretization: tp.Literal['linrgb_float', 'linrgb_16bit', 'linrgb_8bit', 'srgb_8bit'],

        cursor: tp.Any,
        connection: tp.Any,
    ):
        insert_query = f'''INSERT INTO {self._table_name}
                        (blur_type, blur_dataset, kernel, image_dataset, image, discretization, noised, model, ssim, psnr) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''
        
        for model, model_name in self._models:
            if self._model_config[model_name][blur_type]:
                # for wiener_nonblind_noise we use noise version for noise float as well as for sRGB and linRGB in cases with and without noise
                no_noise_model = (
                    model['noise']
                    if model_name == 'wiener_nonblind_noise' and discretization != 'linrgb_float'
                    else model['no_noise']
                )
                # no noise
                metrcics = self._calculate_metrics(no_noise_model, model_name, image, blurred_images['no_noise'], kernel)
                cursor.execute(
                    insert_query,
                    (blur_type, blur_dataset, kernel_path, image_dataset, image_path, discretization, False, model_name, metrcics['ssim'], metrcics['psnr']),
                )
                connection.commit()
                
                # with noise
                metrcics = self._calculate_metrics(model['noise'], model_name, image, blurred_images['noise'], kernel)
                cursor.execute(
                    insert_query,
                    (blur_type, blur_dataset, kernel_path, image_dataset, image_path, discretization, True, model_name, metrcics['ssim'], metrcics['psnr']),
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


class MainTester(BaseTester):
    """
    Main tester for models evaluation:

    1) float sRGB - (srgb2lin) -> float linear -> linear 16 bit - (linear2srgb) -> sRGB 8 bit.
    2) sRGB 8bit - (uint8_to_float32) -> float sRGB -> 1)
    """
    def __init__(
        self,
        benchmark_list_path: str, 
        models: tp.List[tp.Tuple[dict, str]],
        db_path: str,
        table_name: str,
        model_config: dict,
        data_config: dict,
    ):
        super().__init__(
            benchmark_list_path,
            models,
            db_path,
            table_name,
            model_config,
            data_config,
        )

    def test(self):
        connection = sqlite3.connect(self._db_path)
        cursor = connection.cursor()

        with open(self._benchmark_list_path, 'r+') as file:
            next(file)  # skip line with headers
            for line in tqdm(file):
                blur_type, blur_dataset, kernel_path, image_dataset, image_path = line.strip().split(',')

                kernel = load_npy(kernel_path, key='psf')

                image = imread(image_path)
                image = center_crop(image, IMAGE_SIZE, IMAGE_SIZE)

                if image_path.endswith('.jpg'):  # sRGB 8 bit
                    image = uint8_to_float32(image)

                if image.ndim == 3:
                    image = srgb2gray(image)

                image = srgbf_to_linrgbf(image)  # convert from float sRGB to linRGB
        
                # ---- lin float ----
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
                    discretization='linrgb_float',
                )

                # ---- lin uint16 ----
                image = float_to_uint16(image)
                blurred = float_to_uint16(blurred)
                noised_blurred = float_to_uint16(noised_blurred)
                blurred_3d = float_to_uint16(blurred_3d)
                noised_blurred_3d = float_to_uint16(noised_blurred_3d)

                blurred_images = {
                    'no_noise': {'1d': (blurred / MAX_UINT16).astype(np.float32), '3d': (blurred_3d / MAX_UINT16).astype(np.float32)},
                    'noise': {'1d': (noised_blurred / MAX_UINT16).astype(np.float32), '3d': (noised_blurred_3d / MAX_UINT16).astype(np.float32)},
                }

                self._run_models(
                    image=(image / MAX_UINT16).astype(np.float32), blurred_images=blurred_images, kernel=kernel, blur_type=blur_type, 
                    blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                    cursor=cursor, connection=connection,
                    discretization='linrgb_16bit',
                )

                # ---- lin RGB uint8 ----
                image_lin8 = uint16_to_uint8(image)
                blurred_lin8 = uint16_to_uint8(blurred)
                noised_blurred_lin8 = uint16_to_uint8(noised_blurred)
                blurred_3d_lin8 = uint16_to_uint8(blurred_3d)
                noised_blurred_3d_lin8 = uint16_to_uint8(noised_blurred_3d)

                blurred_images = {
                    'no_noise': {'1d': (blurred_lin8 / MAX_UINT8).astype(np.float32), '3d': (blurred_3d_lin8 / MAX_UINT8).astype(np.float32)},
                    'noise': {'1d': (noised_blurred_lin8 / MAX_UINT8).astype(np.float32), '3d': (noised_blurred_3d_lin8 / MAX_UINT8).astype(np.float32)},
                }

                self._run_models(
                    image=(image_lin8 / MAX_UINT8).astype(np.float32), blurred_images=blurred_images, kernel=kernel, blur_type=blur_type,
                    blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                    cursor=cursor, connection=connection,
                    discretization='linrgb_8bit',
                )

                # ---- sRGB uint8 ----
                image = linrrgb16_to_srgb8(image)
                blurred = linrrgb16_to_srgb8(blurred)
                noised_blurred = linrrgb16_to_srgb8(noised_blurred)
                blurred_3d = linrrgb16_to_srgb8(blurred_3d)
                noised_blurred_3d = linrrgb16_to_srgb8(noised_blurred_3d)

                blurred_images = {
                    'no_noise': {'1d': (blurred / MAX_UINT8).astype(np.float32), '3d': (blurred_3d / MAX_UINT8).astype(np.float32)},
                    'noise': {'1d': (noised_blurred / MAX_UINT8).astype(np.float32), '3d': (noised_blurred_3d / MAX_UINT8).astype(np.float32)},
                }

                self._run_models(
                    image=(image / MAX_UINT8).astype(np.float32), blurred_images=blurred_images, kernel=kernel, blur_type=blur_type,
                    blur_dataset=blur_dataset, kernel_path=kernel_path, image_dataset=image_dataset, image_path=image_path,
                    cursor=cursor, connection=connection,
                    discretization='srgb_8bit',
                )

        cursor.close()


class ModelPipelineTester(BaseTester):
    """
    Tester for models evaluation (model pipeline VS real-life pipeline):

    1) Model pipeline: [sRGB 8bit - (uint8_to_float32) ->] float sRGB -> *CONVOLUTION* -> RESTORATION.
    """
    def __init__(
        self,
        benchmark_list_path: str, 
        models: tp.List[tp.Tuple[dict, str]],
        db_path: str,
        table_name: str,
        model_config: dict,
        data_config: dict,
    ):
        super().__init__(
            benchmark_list_path,
            models,
            db_path,
            table_name,
            model_config,
            data_config,
        )

    def test(self):
        ...


class RealPipileneTester(BaseTester):
    """
    Tester for models evaluation (model pipeline):

    2) Real-life pipeline: [sRGB 8bit - (uint8_to_float32) ->] float sRGB - (srgb2lin) -> float linear -> *CONVOLUTION* -> RESTORATION.
    """
    def __init__(
        self,
        benchmark_list_path: str, 
        models: tp.List[tp.Tuple[dict, str]],
        db_path: str,
        table_name: str,
        model_config: dict,
        data_config: dict,
    ):
        super().__init__(
            benchmark_list_path,
            models,
            db_path,
            table_name,
            model_config,
            data_config,
        )

    def test(self):
        ...

