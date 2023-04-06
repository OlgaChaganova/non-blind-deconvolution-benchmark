"""Converts torch nn.module to TorshScript compiled model."""

"""
TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect.
We can't record the data flow of Python values, so this value will be treated as a constant in the future.
This means that the trace might not generalize to other inputs!
"""

import argparse
import logging
import os
import typing as tp

import numpy as np
import torch
from omegaconf import OmegaConf

from constants import IMAGE_SIZE as image_size, CONFIG as config
from deconv.neural.usrnet.model.model import USRNet
from deconv.neural.usrnet.predictor import load_weights as load_weights_usrnet
from deconv.neural.dwdn.model.model import DEBLUR
from deconv.neural.dwdn.predictor import load_weights as load_weights_dwdn
from deconv.neural.kerunc.model.model import KernelErrorModel
from deconv.neural.kerunc.predictor import load_weights as load_weights_kerunc


def parse() -> tp.Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', nargs='+', default=['usrnet', 'dwdn', 'kerunc'], help='Name of models to be converted to ONNX',
    )
    parser.add_argument(
        '--is_noised', action='store_true', help='Use noised version of model or not',
    )
    parser.add_argument(
        '--save_dir', type=str, default='models/', help='Path to directory where onnx models will be saved',
    )
    parser.add_argument(
        '--check', action='store_false', help='Check correctness of converting',
    )
    return parser.parse_args()


def convert(
    model_name: str,
    model_path: str,
    save_dir: str,
    check: bool,
    num_channels: int, 
    model_params: tp.Optional[dict] = None,
) -> tp.Tuple[str, torch.tensor, torch.tensor]:
    
    batch_size = 1
    if model_name == 'usrnet':
        model = USRNet(
            n_iter=8,
            h_nc=64,
            in_nc=4,
            out_nc=3,
            nc=[64, 128, 256, 512],
            nb=2, 
            act_mode='R', 
            downsample_mode='strideconv',
            upsample_mode='convtranspose',
        )
        model = load_weights_usrnet(model=model, model_path=model_path)
        model.eval()

        # outputs
        y_sample = torch.randn((batch_size, num_channels, image_size, image_size))
        psf_sample = torch.randn((batch_size, 1, 41, 41))
        sigma = torch.tensor(1).float().view([batch_size, 1, 1, 1])
        scale_factor = torch.tensor(1)
        input_sample = (y_sample, psf_sample, scale_factor, sigma)

    elif model_name == 'dwdn':
        model = DEBLUR(
            n_levels=2,
            scale=0.5,
        )
        model = load_weights_dwdn(model=model, model_path=model_path)
        model.eval()

        # outputs
        y_sample = torch.randn((batch_size, num_channels, image_size, image_size))
        psf_sample = torch.randn((batch_size, 1, 41, 41))
        input_sample = (y_sample, psf_sample)
    
    elif model_name == 'kerunc':
        model = KernelErrorModel(
            layers=4,
            deep=17,
            **model_params,
        )
        model = load_weights_kerunc(model=model, model_path=model_path)
        model.eval()

        # outputs
        y_sample = torch.randn((batch_size, num_channels, image_size, image_size))
        psf_sample = torch.randn((batch_size, 1, image_size, image_size))
        input_sample = (y_sample, psf_sample)
    
    output_sample = model(*input_sample)

    ts_model_name = os.path.split(model_path)[-1].split('.')[0] + '-scripted.pt'
    ts_model_path = os.path.join(save_dir, ts_model_name)

    traced_model = torch.jit.trace(model, example_inputs=input_sample)
    torch.jit.save(traced_model, f=ts_model_path)

    if os.path.isfile(ts_model_path):
        logging.info(f'Model was successfully saved. Path: {ts_model_path}')
    else:
        raise ValueError('An error was occurred. Check paths and try again.')
    
    if check:
        return ts_model_path, input_sample, output_sample


def check(ts_model_path: str, input_sample: tp.List[torch.tensor], output_sample: torch.tensor):
    model = torch.jit.load(ts_model_path)
    output_sample_ts = model(*input_sample)
    if isinstance(output_sample_ts, list) or isinstance(output_sample_ts, tuple):
        output_sample_ts = output_sample_ts[0]
        output_sample = output_sample[0]

    if np.allclose(output_sample_ts, output_sample.numpy(), atol=1e-3):
        logging.info('Model can be loaded and outputs are closed to outputs from torch model!')
    else:
        logging.info(f'TorchScript: {output_sample_ts.shape}, torch: {output_sample.numpy().shape}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse()

    config = OmegaConf.load(config)
    cm = config['models']

    noise_version = 'noise_params' if args.is_noised else 'no_noise_params'
    for model_name in args.model:
        model_path = (cm[model_name][noise_version]['model_path']
                      if model_name == 'kerunc'
                      else cm[model_name]['model_path'])
        if model_name == 'kerunc':
            model_params = cm[model_name][noise_version]
            model_params = {key: value for key, value in model_params.items() if key not in ['model_path', 'device']}
        else:
            model_params = None
        num_channels = 3 if cm[model_name]['RGB'] else 1

        logging.info(f'Converting {model_name}...')
        ts_model_path, input_sample, output_sample = convert(
            model_name=model_name,
            model_path=model_path,
            save_dir=args.save_dir,
            model_params=model_params,
            num_channels=num_channels,
            check=args.check,
        )
        if args.check:
            check(ts_model_path, input_sample, output_sample)

