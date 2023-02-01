import itertools
import logging
import os
import typing as tp
from pathlib import Path

import pandas as pd
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
    ):
        self.is_full =  is_full
        self.models = models

        self.kernels = {}
        self.gt_images = {}

        self.results = []

        self._prepare()

    def _prepare(self):  # HARD-HARD-HARDCODE
        assert self.is_full == False

        if not self.is_full:
            self.kernels['motion'] = list(Path('datasets/kernels/motion-blur/processed/Levin').rglob('*.npy'))[:1] +\
                                list(Path('datasets/kernels/motion-blur/processed/Sun').rglob('*.npy'))[:1] + \
                                list(Path('datasets/kernels/motion-blur/processed/synthetic').rglob('*.npy'))[:1]
            self.kernels['gauss'] = list(Path('datasets/kernels/gauss-blur/processed/synthetic').rglob('*.npy'))[:1]
            self.kernels['eye'] = list(Path('datasets/kernels/eye-psf/processed/synthetic').rglob('*.npy'))[:1]

            self.gt_images['bsds300'] = list(Path('datasets/gt/BSDS300').rglob('*.jpg'))[:1]
            self.gt_images['sun'] = list(Path('datasets/gt/Sun-gray').rglob('*.png'))[:1]

            for blur_type in self.kernels.keys():
                self.kernels[blur_type] = list(map(lambda path: (load_npy(path, 'psf'), str(path)), self.kernels[blur_type]))

            for image_dataset in self.gt_images.keys():
                self.gt_images[image_dataset] = list(map(lambda path: (imread(path), str(path)), self.gt_images[image_dataset]))

        logging.info('Everything is ready for the test to begin.')
    

    def test(self):
        results = []
        for blur_type in tqdm(self.kernels.keys()):
            for image_dataset in self.gt_images.keys():
                image_kernel_pairs = list(itertools.product(self.kernels[blur_type], self.gt_images[image_dataset]))  # сartesian product
                for pair in image_kernel_pairs:
                    kernel = pair[0][0]
                    kernel_path = pair[0][1]
                    blur_dataset = kernel_path.split(os.sep)[-2]
                    image = pair[1][0]
                    image_path = pair[1][1]

                    image = crop2even(image)
                    logging.info(image.shape)
                    image = rgb2gray(image)

                    try:
                        blurred = convolve(image, kernel)
                        blurred_3d = gray2gray3d(blurred)
                    except ValueError: #index can't contain negative values
                        continue
                    
                    noised_blurred = make_noised(blurred, mu=0, sigma=0.01)
                    noised_blurred_3d = gray2gray3d(noised_blurred)

                    for model, model_name in self.models:                      
                        if model_name in ['dwdn', 'kerunc'] and blur_type in ['gauss', 'eye']:
                            continue
                        
                        # without noise:
                        restored = (
                            model(blurred_3d, kernel)[..., 0]
                            if model_name in ['usrnet', 'dwdn']
                            else model(blurred, kernel)
                        )
                        ssim_value = ssim(image, restored)
                        psnr_value = psnr(image, restored)
                        results.append([blur_type, blur_dataset, kernel_path, image_dataset, image_path, 'float32', False, model_name, ssim_value, psnr_value, None])

                        # with noise:
                        restored = (
                            model(noised_blurred_3d, kernel)[..., 0]
                            if model_name in ['usrnet', 'dwdn']
                            else model(noised_blurred, kernel)
                        )
                        ssim_value = ssim(image, restored)
                        psnr_value = psnr(image, restored)
                        self.results.append([blur_type, blur_dataset, kernel_path, image_dataset, image_path, 'float32', True, model_name, ssim_value, psnr_value, None])
        
        self.results = pd.DataFrame(
            results,
            columns=[
                'blur_type',
                'blur_dataset',
                'kernel',  # path to kernel.npy
                'image_dataset',
                'image',  # path to image.png
                'discretization',
                'noised',
                'model',
                'SSIM',
                'PSNR',
                'Sharpness',
            ]
        )