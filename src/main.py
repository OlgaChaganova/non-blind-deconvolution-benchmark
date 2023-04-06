import argparse
import logging
import os
import typing as tp
import warnings

import numpy as np
from omegaconf import OmegaConf

from services.database import DatabaseMetrics, DatabaseConfigs
from deconv.classic.wiener.wiener import wiener_gray
from deconv.neural.usrnet.predictor import USRNetPredictor
from deconv.neural.dwdn.predictor import DWDNPredictor
from deconv.neural.kerunc.predictor import KerUncPredictor
from deconv.neural.rgdn.predictor import RGDNPredictor
from services.tester import MainTester, ModelPipelineTester, RealPipileneTester


warnings.filterwarnings('ignore', category=UserWarning)  # due to old pythorch version (1.7)


_AVAILABLE_MODELS = tp.Literal[
    'wiener_blind_noise',
    'wiener_nonblind_noise',
    'usrnet',
    'dwdn',
    'kerunc',
]


def parse():
    parser = argparse.ArgumentParser(
        description='Parser for nbd models testing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['main', 'model_vs_real'],
        default='main',
        help='''Mode of benchmarking.\n
                1) `main`:  models will be tested on the whole pipeline with convolution in linear space and all discrization types.
                2) `model-vs-real`: models will be tested on the model pipeline (convolution in sRGB)
                and real-life pipeline (convolution in linear space) with float discritization.'''
    )
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join('configs', 'config.yml'),
        help='Config for benchmarking.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=tp.get_args(_AVAILABLE_MODELS),
        help='List of models to be tested. By default, all available models will be tested.',
    )
    parser.add_argument(
        '--db_name',
        type=str,
        default='results/metrics',
        help='Database of metrics name.',
    )
    parser.add_argument(
        '--db_config_name',
        type=str,
        default='results/configs',
        help='Database of configs name.',
    )
    parser.add_argument(
        '--table_name',
        type=str,
        default='all_models',
        help='Table name.',
    )
    return parser.parse_args()


def main():
    args = parse()

    config = OmegaConf.load(args.config)
    cm = config.models
    cd = config.dataset

    models = []

    if 'wiener_blind_noise' in args.models:
        models.append(
                (
                    {
                        'noise': lambda image, psf: wiener_gray(image, psf, **cm.wiener_blind_noise.noise_params),
                        'no_noise': lambda image, psf: wiener_gray(image, psf, **cm.wiener_blind_noise.no_noise_params),
                    },
                    'wiener_blind_noise'
                )
        )
    
    if 'wiener_nonblind_noise' in args.models:
        models.append(
                (
                    {
                        'noise': lambda image, psf: wiener_gray(image, psf, **cm.wiener_nonblind_noise.noise_params),
                        'no_noise': lambda image, psf: wiener_gray(image, psf, **cm.wiener_nonblind_noise.no_noise_params),
                    },
                    'wiener_nonblind_noise'
                )
        )

    if 'kerunc' in args.models:
        models.append(
            (
                {
                    'noise': KerUncPredictor(**cm.kerunc.noise_params),
                    'no_noise': KerUncPredictor(**cm.kerunc.no_noise_params),
                },
                'kerunc'
            )
        )

    if 'usrnet' in args.models:
        models.append(
            (
                {
                    'noise': USRNetPredictor(model_path=cm.usrnet.model_path, **cm.usrnet.noise_params),
                    'no_noise': USRNetPredictor(model_path=cm.usrnet.model_path, **cm.usrnet.no_noise_params),
                },
                'usrnet'
            )
        )

    if 'dwdn' in args.models:
        models.append(
            (
                {
                    'noise': DWDNPredictor(model_path=cm.dwdn.model_path, **cm.dwdn.noise_params),
                    'no_noise': DWDNPredictor(model_path=cm.dwdn.model_path, **cm.dwdn.no_noise_params),
                },
                'dwdn'
            )
        )
    
    if 'rgdn' in args.models:
        models.append(
            (
                {
                    'noise': RGDNPredictor(model_path=cm.rgdn.model_path, **cm.rgdn.noise_params),
                    'no_noise': RGDNPredictor(model_path=cm.rgdn.model_path, **cm.rgdn.no_noise_params),
                },
                'rgdn'
            )
        )
    
    if len(models) > 0:
        selected_models = [model[1] for model in models]
        logging.info(f'The following models were selected for testing: {selected_models}')

        table_name = '_'.join([args.mode, args.table_name])

        db_metrics = DatabaseMetrics(db_name=args.db_name)
        db_metrics.create_or_connect_db()
        db_metrics.create_table(table_name=table_name)

        db_configs = DatabaseConfigs(db_name=args.db_config_name)
        db_configs.create_or_connect_db()
        db_configs.create_table(table_name=table_name)
        db_configs.add(args.config, selected_models)

        if args.mode == 'main':
            tester = MainTester(
                benchmark_list_path=cd.benchmark_list_path,
                models=models,
                db_path=db_metrics.db_path,
                table_name=table_name,
                model_config=cm,
                data_config=cd,
            )
            tester.test()

        elif args.mode == 'model_vs_real':
            tester = ModelPipelineTester(
                benchmark_list_path=cd.benchmark_list_path,
                models=models,
                db_path=db_metrics.db_path,
                table_name=table_name,
                model_config=cm,
                data_config=cd,
            )
            tester.test()

            tester = RealPipileneTester(
                benchmark_list_path=cd.benchmark_list_path,
                models=models,
                db_path=db_metrics.db_path,
                table_name=table_name,
                model_config=cm,
                data_config=cd,
            )
            tester.test()

        logging.info('Done!')

    else:
        logging.error(f'No model was selected. Available models: {tp.get_args(_AVAILABLE_MODELS)}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.random.seed(8)
    main()
