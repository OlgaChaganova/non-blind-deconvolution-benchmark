import logging
import typing as tp

import numpy as np
import torch
from torch import nn

from deconv.neural.dwdn.model.model import DEBLUR
from deconv.neural.dwdn.model.utils_deblur import postprocess


def load_weights(model: nn.Module, model_path: str) -> nn.Module:
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    logging.info('Model\'s state was loaded successfully.')
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model


class DWDNPredictor(object):
    def __init__(
        self,
        model_path: str,
        n_levels: int = 2,
        scale: float = 0.5,
        rgb_range: float = 1,
        device: tp.Literal['cpu', 'cuda', 'auto'] = 'auto',
    ):
        self._device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            if device == 'auto'
            else torch.device(device)
        )

        model = DEBLUR(
            n_levels=n_levels,
            scale=scale,
        )
        self._model = load_weights(model=model, model_path=model_path).to(self._device)
        self.rgb_range = rgb_range
    
    def __call__(self, blurred_image: np.array, psf: np.array) -> np.array:
        """Forward pass on the inference stage.

        Parameters
        ----------
        blurred_image : np.array
            Blurred image. Shape: [height, width, num_channels]
        psf : np.array
            PSF. Shape: [height, width]

        Returns
        -------
        np.array
           Restored image. Shape: [height, width, num_channels]
        """
        blurred_image, psf = self._preprocess(blurred_image, psf)
        with torch.no_grad():
            model_output = self._model(blurred_image, psf)
        return self._postprocess(model_output)
    
    def _preprocess(self, blurred_image: np.array, psf: np.array) -> tp.Tuple[torch.tensor, torch.tensor]:
        h, w = psf.shape
        if h % 2 == 0:
            psf = np.pad(psf, ((1, 0), (0, 0)), mode='constant')
        if w % 2 == 0:
            psf = np.pad(psf, ((0, 0), (1, 0)), mode='constant')
        return (
            torch.from_numpy(blurred_image).permute(2, 0, 1).unsqueeze(dim=0).to(self._device),
            torch.from_numpy(psf).unsqueeze(dim=0).unsqueeze(dim=0).to(self._device)
        )
    
    def _postprocess(self, model_output: tp.List[torch.tensor]) -> np.array:
        deblur = postprocess(model_output[-1], rgb_range=self.rgb_range)
        return deblur[0].cpu().permute(0, 2, 3, 1).numpy()[0]