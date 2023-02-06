import argparse
import logging
import typing as tp
import warnings

import numpy as np
from omegaconf import OmegaConf

from database import Database
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
    parser.add_argument(
        '--db_name',
        type=str,
        default='results',
        help='Database name.',
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

    config = OmegaConf.load('config.yml')
    cm = config.models
    cd = config.dataset

    models = []

    if 'wiener_blind_noise' in args.models:
        models.append(
            (lambda image, psf: wiener_gray(image, psf, **cm.wiener_blind_noise.params), 'wiener_blind_noise')
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
        database = Database(db_name=args.db_name)

        tester = Tester(
            is_full=cd.full,
            models=models,
            db_path=database.db_path,
            table_name=args.table_name,
            model_config=cm,
            data_config=cd,
        )

        database.create_or_connect_db()
        database.create_table(table_name=args.table_name)

        tester.test()

    else:
        logging.error(f'No model was selected. Available models: {tp.get_args(_AVAILABLE_MODELS)}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.random.seed(8)
    main()

