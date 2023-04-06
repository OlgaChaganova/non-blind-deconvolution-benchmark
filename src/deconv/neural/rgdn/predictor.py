import logging
import typing as tp

import numpy as np
import torch
from torch import nn

from deconv.neural.rgdn.model.model import OptimizerRGDN


def load_weights(model: nn.Module, model_path: str) -> nn.Module:
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=True)
    logging.info('Model\'s state was loaded successfully.')
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model


class RGDNPredictor(object):
    def __init__(
        self,
        model_path,
        num_steps: int,
        in_channels: int,
        device: tp.Literal['cpu', 'cuda', 'auto'] = 'auto',
    ):
        self._device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            if device == 'auto'
            else torch.device(device)
        )
        self.in_channels = in_channels

        model = OptimizerRGDN(
            num_steps=num_steps,
            in_channels=in_channels,
        )
        self._model = load_weights(model=model, model_path=model_path).to(self._device)
    
    def __call__(self, blurred_image: np.array, psf: np.array) -> np.array:
        """Forward pass on the inference stage.

        Parameters
        ----------
        blurred_image : np.array
            Blurred image. Shape: [height, width]. Supports only GRAY (1 channel) images.
        psf : np.array
            PSF. Shape: [height, width]

        Returns
        -------
        np.array
           Restored GRAY image. Shape: [height, width]
        """
        blurred_image, psf, psf_t = self._preprocess(blurred_image, psf)
        with torch.no_grad():
            model_output = self._model(blurred_image, psf, psf_t)
        return self._postprocess(model_output)
    
    def _preprocess(self, blurred_image: np.array, psf: np.array) -> tp.Tuple[torch.tensor, torch.tensor, torch.tensor]:
        psf_t = np.flip(psf).copy()
        input_image = (
            torch.from_numpy(blurred_image).permute(2, 0, 1).unsqueeze(dim=0)
            if self.in_channels == 3
            else torch.from_numpy(blurred_image).unsqueeze(dim=0).unsqueeze(dim=0)
        )
        return (
            input_image,
            torch.from_numpy(psf).unsqueeze(dim=0).unsqueeze(dim=0),
            torch.from_numpy(psf_t).unsqueeze(dim=0).unsqueeze(dim=0)
        )
    
    def _postprocess(self, model_output: tp.List[torch.tensor]) -> np.array:
        return model_output[-1][0][0].cpu().numpy()