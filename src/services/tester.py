import sqlite3
import typing as tp

import numpy as np
from tqdm import tqdm

from constants import IMAGE_SIZE
from data.convertation import float_to_uint16, uint16_to_uint8, linrgbf_to_srgbf
from data.convolution import convolve
from deconv.neural.usrnet.predictor import USRNetPredictor
from deconv.neural.dwdn.predictor import DWDNPredictor
from deconv.neural.kerunc.predictor import KerUncPredictor
from deconv.neural.rgdn.predictor import RGDNPredictor
from imutils import load_npy, make_noised, gray2gray3d, norm_values, impreprocess
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
                        images=images[discr_type], kernel=kernel, blur_type=blur_type,
                        blur_dataset=blur_dataset, kernel_path=kernel_path,
                        image_dataset=image_dataset, image_path=image_path,
                        cursor=cursor, connection=connection,
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
        discretization: tp.Literal['linrgb_float', 'linrgb_16bit', 'linrgb_8bit', 'srgb_float', 'srgb_8bit', 'srgb_16bit'],

        cursor: tp.Any,
        connection: tp.Any,
    ):
        insert_query = f'''INSERT INTO {self._table_name}
                        (tester_type, blur_type, blur_dataset, kernel, image_dataset, image, discretization, noised, model, ssim, psnr) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''
        
        for model, model_name in self._models:
            if self._model_config[model_name][blur_type]:
                # no noise
                metrcics = self._calculate_metrics(model['no_noise'], model_name, images['image'], images['blurred_no_noise'], kernel)
                cursor.execute(
                    insert_query,
                    (
                        self._tester_type, blur_type, blur_dataset, kernel_path, image_dataset,
                        image_path, discretization, False, model_name, metrcics['ssim'], metrcics['psnr'],
                    ),
                )
                connection.commit()
                
                # with noise
                metrcics = self._calculate_metrics(model['noise'], model_name, images['image'], images['blurred_noise'], kernel)
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
    ) -> dict:
        restored = (
            model(blurred_images['3d'], kernel)[..., 0]
            if model_name in ['usrnet', 'dwdn']
            else model(blurred_images['1d'], kernel)
        )
        restored = np.clip(restored, 0, 1)

        return {
            'ssim': ssim(image, restored),
            'psnr': psnr(image, restored)
        }


class MainTester(BaseTester):
    """Main tester for models evaluation."""
    def __init__(
        self,
        benchmark_list_path: str, 
        models: tp.List[tp.Tuple[dict, str]],
        db_path: str,
        table_name: str,
        model_config: dict,
        data_config: dict,
    ):
        super().__init__(benchmark_list_path, models, db_path, table_name, model_config, data_config)
        self._tester_type = 'main'

    @classmethod
    def get_data(cls, image_path: str, psf_path: str, image_size: int, noise_mu: float, noise_std: float) -> tp.Tuple[dict, np.ndarray]:
        """Return images with different discretization"""
        transformed_images = {}

        image = impreprocess(image_path, crop=True, image_size=image_size)
        psf = load_npy(psf_path, key='psf')

        # ----- linRGB float -----
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
        image_linrgb = float_to_uint16(image)
        noised_blurred_linrgb = float_to_uint16(noised_blurred)
        noised_blurred_3d_linrgb = float_to_uint16(noised_blurred_3d)
        blurred_linrgb = float_to_uint16(blurred)
        blurred_3d_linrgb = float_to_uint16(blurred_3d)

        transformed_images['linrgb_16bit'] = {
            'image': norm_values(image_linrgb),
            'blurred_noise':
                {'1d': norm_values(noised_blurred_linrgb),
                '3d': norm_values(noised_blurred_3d_linrgb)},
            'blurred_no_noise':
                {'1d': norm_values(blurred_linrgb),
                '3d': norm_values(blurred_3d_linrgb)},
        }

        # ---- lin RGB 8 bit ----
        transformed_images['linrgb_8bit'] = {
            'image': norm_values(uint16_to_uint8(image_linrgb)),
            'blurred_noise':
                {'1d': norm_values(uint16_to_uint8(noised_blurred_linrgb)),
                '3d': norm_values(uint16_to_uint8(noised_blurred_3d_linrgb))},
            'blurred_no_noise':
                {'1d': norm_values(uint16_to_uint8(blurred_linrgb)),
                '3d': norm_values(uint16_to_uint8(blurred_3d_linrgb))},
        }

        # # ----- sRGB float -----
        image_srgb = linrgbf_to_srgbf(image)
        noised_blurred_srgb = linrgbf_to_srgbf(noised_blurred)
        noised_blurred_3d_srgb = linrgbf_to_srgbf(noised_blurred_3d)
        blurred_srgb = linrgbf_to_srgbf(blurred)
        blurred_3d_srgb = linrgbf_to_srgbf(blurred_3d)

        transformed_images['srgb_float'] = {
            'image': image_srgb,
            'blurred_noise':
                {'1d': noised_blurred_srgb, 
                '3d': noised_blurred_3d_srgb},
            'blurred_no_noise':
                {'1d': blurred_srgb, 
                '3d': blurred_3d_srgb}
        }

        # ----- sRGB 16 bit -----
        image_srgb = float_to_uint16(image_srgb)
        blurred_srgb = float_to_uint16(blurred_srgb)
        noised_blurred_srgb = float_to_uint16(noised_blurred_srgb)
        blurred_3d_srgb = float_to_uint16(blurred_3d_srgb)
        noised_blurred_3d_srgb = float_to_uint16(noised_blurred_3d_srgb)

        transformed_images['srgb_16bit'] = {
            'image': norm_values(image_srgb),
            'blurred_noise':
                {'1d': norm_values(noised_blurred_srgb),
                '3d': norm_values(noised_blurred_3d_srgb)},
            'blurred_no_noise':
                {'1d': norm_values(blurred_srgb),
                '3d': norm_values(blurred_3d_srgb)},
        }

        # ---- sRGB 8 bit ----
        transformed_images['srgb_8bit'] = {
            'image': norm_values(uint16_to_uint8(image_srgb)),
            'blurred_noise':
                {'1d': norm_values(uint16_to_uint8(noised_blurred_srgb)),
                '3d': norm_values(uint16_to_uint8(noised_blurred_3d_srgb))},
            'blurred_no_noise':
                {'1d': norm_values(uint16_to_uint8(blurred_srgb)),
                '3d': norm_values(uint16_to_uint8(blurred_3d_srgb))},
        }

        return transformed_images, psf
