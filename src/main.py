import argparse
import logging
import os
import typing as tp
import warnings
from datetime import datetime

import pandas as pd
from omegaconf import OmegaConf

from deconv.classic.wiener import wiener_gray
from deconv.neural.usrnet.predictor import USRNetPredictor
from deconv.neural.dwdn.predictor import DWDNPredictor
from deconv.neural.kerunc.predictor import KerUncPredictor
from deconv.neural.rgdn.predictor import RGDNPredictor
from tester import Tester


warnings.filterwarnings('ignore', category=UserWarning)  # due to old pythorch version (1.7)


_AVAILABLE_MODELS = tp.Literal[
    'wiener_blind_noise',
    'usrnet',
    'dwdn',
    'kerunc',
    # 'rgdn'
]


def parse():
    parser = argparse.ArgumentParser(description='Parser for nbd models testing.')
    parser.add_argument(
        '--models',
        nargs='+',
        default=tp.get_args(_AVAILABLE_MODELS),
        help=f'List of models to be tested. By default, all available models: {tp.get_args(_AVAILABLE_MODELS)} will be tested.',
    )
    return parser.parse_args()


def main():
    args = parse()

    config = OmegaConf.load('config.yml')
    cm = config.models
    cd = config.dataset

    models = []

    if 'wiener_blind_noise' in args.models:
        models.append(
            (lambda image, psf: wiener_gray(image, psf, **cm.wiener_noise_blind.params), 'wiener_noise_blind')
        )

    if 'kerunc' in args.models:
        models.append(
            (KerUncPredictor(
                model_path=cm.kerunc.model_path,
                **cm.kerunc.params,
            ), 'kerunc')
        )

    if 'usrnet' in args.models:
        models.append(
            (USRNetPredictor(
                model_path=cm.usrnet.model_path,
                **cm.usrnet.params,
            ), 'usrnet')
        )

    if 'dwdn' in args.models:
        models.append(
            (DWDNPredictor(
                model_path=cm.dwdn.model_path,
                **cm.dwdn.params
            ), 'dwdn')
        )
    
    if 'rgdn' in args.models:
        models.append(
            (RGDNPredictor(
                model_path=cm.rgdn.model_path,
                **cm.rgdn.params
            ), 'rgdn')
        )
    
    if len(models) > 0:
        tester = Tester(
            is_full=cd.full,
            models=models,
        )

        try:
            tester.test()
            results = tester.results

        except (KeyboardInterrupt, RuntimeError):
            results = tester.results
            results = pd.DataFrame(
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
    
        timestamp = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
        results.to_csv(os.path.join('results', f'results_{timestamp}.csv'), index_label='id')

    else:
        logging.error('Results are empty, there is nothing to be written')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

