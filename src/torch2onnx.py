import argparse
import logging
import os
import typing as tp

import numpy as np
import onnxruntime
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
        '--check', action='store_true', help='Check correctness of converting',
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

    elif model_name == 'dwdn':
        model = DEBLUR(
            n_levels=2,
            scale=0.5,
        )
        model = load_weights_dwdn(model=model, model_path=model_path)
        model.eval()
    
    elif model_name == 'kerunc':
        model = KernelErrorModel(
            layers=4,
            deep=17,
            **model_params,
        )
        model = load_weights_kerunc(model=model, model_path=model_path)
        model.eval()

    onnx_model_name = os.path.split(model_path)[-1].split('.')[0] + '.onnx'
    onnx_model_path = os.path.join(save_dir, onnx_model_name)
    print(onnx_model_path)

    batch_size = 2
    y_sample = torch.randn((batch_size, num_channels, image_size, image_size))
    psf_sample = torch.randn((batch_size, 1, image_size, image_size))
    output_sample = model(y_sample, psf_sample)

    torch.onnx.export(
        model,
        (y_sample, psf_sample),
        f=onnx_model_path,
        export_params=True,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output' : {0 : 'batch_size'},
        },
        opset_version=11,
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
    )

    if os.path.isfile(onnx_model_path):
        logging.info(f'Model was successfully saved. Path: {onnx_model_path}')
    else:
        raise ValueError('An error was occurred. Check paths and try again.')
    if check:
        return onnx_model_path, y_sample, psf_sample, output_sample


def check(onnx_model_path: str, y_sample: torch.tensor, psf_sample: torch.tensor, output_sample: torch.tensor):
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: (y_sample.numpy(), psf_sample.numpy())}
    output_sample_onnx = ort_session.run(None, ort_inputs)[0]

    if np.allclose(output_sample_onnx, output_sample.numpy(), atol=1e-3):
        logging.info('Model can be loaded and outputs look good!')
    else:
        logging.info(f'ONNX: {output_sample_onnx.shape}, torch: {output_sample.numpy().shape}')
        logging.info(f'ONNX: {output_sample_onnx[0]}, torch: {output_sample.numpy()[0]}')
        logging.error('Outputs of the converted model do not match output of the original torch model.')


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

        onnx_model_path, input_sample, output_sample = convert(
            model_name=model_name,
            model_path=model_path,
            save_dir=args.save_dir,
            model_params=model_params,
            num_channels=num_channels,
            check=args.check,
        )
        # if args.check:
        #     check(onnx_model_path, input_sample, output_sample)
