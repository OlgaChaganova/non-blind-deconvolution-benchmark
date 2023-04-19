import sqlite3
import typing as tp

import numpy as np
from tqdm import tqdm

from constants import IMAGE_SIZE
from data.convertation import srgbf_to_linrgbf, float_to_uint16, linrrgb16_to_srgb8, uint16_to_uint8, uint8_to_float32, linrgbf_to_srgbf
from data.convolution import convolve
from deconv.neural.usrnet.predictor import USRNetPredictor
from deconv.neural.dwdn.predictor import DWDNPredictor
from deconv.neural.kerunc.predictor import KerUncPredictor
from deconv.neural.rgdn.predictor import RGDNPredictor
from imutils import imread, srgb2gray, load_npy, make_noised, gray2gray3d, center_crop, norm_values
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
        self._tester_type = '_base'
    
    @classmethod
    def get_data(cls, image_path: str, psf_path: str, image_size: int, noise_mu: float, noise_std: float) -> tp.Tuple[dict, np.ndarray]:
        pass

    def test(self):
        connection = sqlite3.connect(self._db_path)
        cursor = connection.cursor()

        with open(self._benchmark_list_path, 'r+') as file:
            next(file)  # skip line with headers
            for line in tqdm(file):
                blur_type, blur_dataset, kernel_path, image_dataset, image_path = line.strip().split(',')

                images, kernel = self.get_data(
                    image_path,
                    kernel_path,
                    image_size=IMAGE_SIZE,
                    noise_mu=self._data_config['blur']['mu'],
                    noise_std=self._data_config['blur']['sigma'],
                )

                for discr_type in images.keys():
                    self._run_models(
                        images=images[discr_type], kernel=kernel, blur_type=blur_type, blur_dataset=blur_dataset, kernel_path=kernel_path,
                        image_dataset=image_dataset, image_path=image_path, cursor=cursor, connection=connection,
                        discretization=discr_type,
                    )

        cursor.close()

    def _run_models(
        self,
        images: dict,
        kernel: np.array,

        blur_type: str,
        blur_dataset: str,
        kernel_path: str,
        image_dataset: str,
        image_path: str,
        discretization: tp.Literal['linrgb_float', 'srgb_float', 'linrgb_16bit', 'linrgb_8bit', 'srgb_8bit'],

        cursor: tp.Any,
        connection: tp.Any,
    ):
        insert_query = f'''INSERT INTO {self._table_name}
                        (tester_type, blur_type, blur_dataset, kernel, image_dataset, image, discretization, noised, model, ssim, psnr) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''
        
        for model, model_name in self._models:
            if self._model_config[model_name][blur_type]:
                # for wiener_nonblind_noise we use noise version for noise float as well as for sRGB and linRGB in cases with and without noise
                no_noise_model = (
                    model['noise']
                    if model_name == 'wiener_nonblind_noise' and discretization not in ['linrgb_float', 'srgb_float']
                    else model['no_noise']
                )
                # no noise
                metrcics = self._calculate_metrics(no_noise_model, model_name, images['image'], images['blurred_no_noise'], kernel, discretization)
                cursor.execute(
                    insert_query,
                    (
                        self._tester_type, blur_type, blur_dataset, kernel_path, image_dataset,
                        image_path, discretization, False, model_name, metrcics['ssim'], metrcics['psnr'],
                    ),
                )
                connection.commit()
                
                # with noise
                metrcics = self._calculate_metrics(model['noise'], model_name, images['image'], images['blurred_noise'], kernel, discretization)
                cursor.execute(
                    insert_query,
                    (
                        self._tester_type, blur_type, blur_dataset, kernel_path, image_dataset,
                        image_path, discretization, True, model_name, metrcics['ssim'], metrcics['psnr'],
                    ),
                )
                connection.commit()    

    def _calculate_metrics(
        self,
        model: tp.Union[USRNetPredictor, DWDNPredictor, KerUncPredictor, RGDNPredictor, tp.Callable],
        model_name: str,
        image: np.array, 
        blurred_images: dict,
        kernel: np.array,
        discr_type: str,
    ) -> dict:
        restored = (
            model(blurred_images['3d'], kernel)[..., 0]
            if model_name in ['usrnet', 'dwdn']
            else model(blurred_images['1d'], kernel)
        )

        restored = np.clip(restored, 0, 1)

        if discr_type.startswith('srgb'):
            restored = linrgbf_to_srgbf(restored)

        return {
            'ssim': ssim(image, restored),
            'psnr': psnr(image, restored)
        }


class MainTester(BaseTester):
    """
    Main tester for models evaluation:
    @ means evaluation with that discretization

    1) @ float sRGB - (srgb2lin) -> @ float linear -> @ linear 16 bit - (linear2srgb) -> @ sRGB 8 bit.
    2) sRGB 8bit - (uint8_to_float32) -> @ float sRGB -> 1)

    For sRGB images, inverse gamma correction is applied before deconvolution.
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
        self._tester_type = 'main'
    
    @classmethod
    def get_data(cls, image_path: str, psf_path: str, image_size: int, noise_mu: float, noise_std: float) -> tp.Tuple[dict, np.ndarray]:
        """Return images with different discretization"""
        transformed_images = {}

        psf = load_npy(psf_path, key='psf')

        image = imread(image_path)
        image = center_crop(image, image_size, image_size)

        if image_path.endswith('.jpg'):  # sRGB 8 bit
            image = uint8_to_float32(image)  # sRGB float

        if image.ndim == 3:
            image = srgb2gray(image)  # sRGB float

        # ----- linRGB float -----
        image = srgbf_to_linrgbf(image)  # linRGB float
        blurred = convolve(image, psf)  # convolution in linear space
        blurred_3d = gray2gray3d(blurred)
        noised_blurred = make_noised(blurred, mu=noise_mu, sigma=noise_std)
        noised_blurred_3d = make_noised(blurred_3d, mu=noise_mu, sigma=noise_std)

        transformed_images['linrgb_float'] = {
            'image': image,
            'blurred_noise':
                {'1d': noised_blurred, 
                '3d': noised_blurred_3d},
            'blurred_no_noise':
                {'1d': blurred, 
                '3d': blurred_3d}
        }

        # ----- linRGB 16 bit -----
        image = float_to_uint16(image)
        blurred = float_to_uint16(blurred)
        noised_blurred = float_to_uint16(noised_blurred)
        blurred_3d = float_to_uint16(blurred_3d)
        noised_blurred_3d = float_to_uint16(noised_blurred_3d)

        transformed_images['linrgb_16bit'] = {
            'image': norm_values(image),
            'blurred_noise':
                {'1d': norm_values(noised_blurred),
                '3d': norm_values(noised_blurred_3d)},
            'blurred_no_noise':
                {'1d': norm_values(blurred),
                '3d': norm_values(noised_blurred_3d)},
        }

        # ---- lin RGB 8 bit ----
        transformed_images['linrgb_8bit'] = {
            'image': norm_values(uint16_to_uint8(image)),
            'blurred_noise':
                {'1d': norm_values(uint16_to_uint8(noised_blurred)),
                '3d': norm_values(uint16_to_uint8(noised_blurred_3d))},
            'blurred_no_noise':
                {'1d': norm_values(uint16_to_uint8(blurred)),
                '3d': norm_values(uint16_to_uint8(blurred_3d))},
        }

        # ---- sRGB 8 bit ----
        transformed_images['srgb_8bit'] = {
            'image': norm_values(linrrgb16_to_srgb8(image)),
            'blurred_noise':
                {'1d': srgbf_to_linrgbf(norm_values(linrrgb16_to_srgb8(noised_blurred))),
                '3d': srgbf_to_linrgbf(norm_values(linrrgb16_to_srgb8(noised_blurred_3d)))},
            'blurred_no_noise':
                {'1d': srgbf_to_linrgbf(norm_values(linrrgb16_to_srgb8(blurred))),  # translate to linRGB for correct deconvolution
                '3d': srgbf_to_linrgbf(norm_values(linrrgb16_to_srgb8(blurred_3d)))},  # translate to linRGB for correct deconvolution
        }

        return transformed_images, psf


class NNPipelineTester(BaseTester):
    """
    Tester for models evaluation (model pipeline with convolution in non-linear space).

    Pipeline:
    [sRGB 8bit - (uint8_to_float32) ->] @ float sRGB -> *CONVOLUTION* -> RESTORATION.
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
        self._tester_type = 'nn'
    
    @classmethod
    def get_data(cls, image_path: str, psf_path: str, image_size: int, noise_mu: float, noise_std: float) -> tp.Tuple[dict, np.ndarray]:
        """Return images with different discretization"""
        transformed_images = {}

        psf = load_npy(psf_path, key='psf')

        image = imread(image_path)
        image = center_crop(image, image_size, image_size)

        if image_path.endswith('.jpg'):  # sRGB 8 bit
            image = uint8_to_float32(image)  # sRGB float32

        if image.ndim == 3:
            image = srgb2gray(image)
        
        blurred = convolve(image, psf)  # CONVOLUTION in NON-LINEAR space
        blurred_3d = gray2gray3d(blurred)

        noised_blurred = make_noised(blurred, mu=noise_mu, sigma=noise_std)
        noised_blurred_3d = make_noised(blurred_3d, mu=noise_mu, sigma=noise_std)

        transformed_images['srgb_float'] = {
            'image': image,
            'blurred_noise':
                {'1d': noised_blurred, 
                '3d': noised_blurred_3d},
            'blurred_no_noise':
                {'1d': blurred, 
                '3d': blurred_3d}
        }

        return transformed_images, psf


class RealPipileneTester(BaseTester):
    """
    Tester for models evaluation (real-life pipeline with convolution in linear space).
    Restoration algorithms are applyed to linRGB images.

    Pipeline:
    1) [sRGB 8bit - (uint8_to_float32) ->] float sRGB - (srgb2lin) -> @ float linear -> *CONVOLUTION* -> RESTORATION.
    2) [sRGB 8bit - (uint8_to_float32) ->] float sRGB - (srgb2lin) ->   float linear -> *CONVOLUTION* -> *GAMMA-CORRECTION* -> @ float sRGB -> RESTORATION.
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
        self._tester_type = 'real'
    
    @classmethod
    def get_data(cls, image_path: str, psf_path: str, image_size: int, noise_mu: float, noise_std: float) -> tp.Tuple[dict, np.ndarray]:
        transformed_images = {}

        kernel = load_npy(psf_path, key='psf')

        image = imread(image_path)
        image = center_crop(image, image_size, image_size)

        if image_path.endswith('.jpg'):  # sRGB 8 bit
            image = uint8_to_float32(image)  # sRGB float32

        if image.ndim == 3:
            image = srgb2gray(image)

        image = srgbf_to_linrgbf(image)  # convert from float sRGB to float linRGB

        # ---- lin float ----
        blurred = convolve(image, kernel)  # CONVOLUTION in LINEAR space
        noised_blurred = make_noised(blurred, mu=noise_mu, sigma=noise_std)
        blurred_3d = gray2gray3d(blurred)
        noised_blurred_3d = make_noised(blurred_3d, mu=noise_mu, sigma=noise_std)

        transformed_images['linrgb_float'] = {
            'image': image,
            'blurred_noise':
                {'1d': noised_blurred, 
                '3d': noised_blurred_3d},
            'blurred_no_noise':
                {'1d': blurred, 
                '3d': blurred_3d}
        }

        transformed_images['srgb_float'] = {
            'image': linrgbf_to_srgbf(image),
            'blurred_noise':
                {'1d': linrgbf_to_srgbf(noised_blurred), 
                '3d': linrgbf_to_srgbf(noised_blurred_3d)},
            'blurred_no_noise':
                {'1d': linrgbf_to_srgbf(blurred), 
                '3d': linrgbf_to_srgbf(blurred_3d)}
        }

        return transformed_images

    def _calculate_metrics(
        self,
        model: tp.Union[USRNetPredictor, DWDNPredictor, KerUncPredictor, RGDNPredictor, tp.Callable],
        model_name: str,
        image: np.array, 
        blurred_images: dict,
        kernel: np.array,
        discr_type: str,
    ) -> dict:
        restored = (
            model(blurred_images['3d'], kernel)[..., 0]
            if model_name in ['usrnet', 'dwdn']
            else model(blurred_images['1d'], kernel)
        )

        # no inverse gamma correction for blurred sRGB images

        return {
            'ssim': ssim(image, restored),
            'psnr': psnr(image, restored)
        }

