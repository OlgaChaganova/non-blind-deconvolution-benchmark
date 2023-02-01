import logging
import typing as tp

import numpy as np
import torch
from torch import nn

from deconv.neural.kerunc.model.model import KernelErrorModel
from deconv.neural.kerunc.model.utils.comfft import fft
from deconv.neural.kerunc.model.utils.imtools import for_fft


def load_weights(model: nn.Module, model_path: str) -> nn.Module:
    model.load_state_dict(torch.load(model_path)['model'], strict=True)
    logging.info('Model\'s state was loaded successfully.')
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model


class KerUncPredictor(object):
    def __init__(
        self,
        model_path,
        lmds: tp.List[float],
        layers: int = 4,
        deep: int = 17,
        device: tp.Literal['cpu', 'cuda', 'auto'] = 'auto',
    ):
        self._device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            if device == 'auto'
            else torch.device(device)
        )

        model = KernelErrorModel(
            lmds=lmds,
            layers=layers,
            deep=deep,
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
        blurred_image, psf = self._preprocess(blurred_image, psf)
        with torch.no_grad():
            model_output = self._model(blurred_image, psf)
        return self._postprocess(model_output)
    
    def _preprocess(self, blurred_image: np.array, psf: np.array) -> tp.Tuple[torch.tensor, torch.tensor]:
        psf = torch.FloatTensor(for_fft(psf, shape=np.shape(blurred_image)))
        psf = fft(psf).unsqueeze(0)
        return torch.from_numpy(blurred_image).unsqueeze(dim=0).unsqueeze(dim=0), psf.unsqueeze(dim=0)
    
    def _postprocess(self, model_output: tp.List[torch.tensor]) -> np.array:
        return model_output[-1].cpu().squeeze(0).squeeze(0).numpy()