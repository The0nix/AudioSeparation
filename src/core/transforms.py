import warnings
from typing import Union, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio


class MelSpectrogram(nn.Module):
    """
    torchaudio MelSpectrogram wrapper for audiomentations's Compose
    """
    def __init__(self, clip_min_value=1e-5, *args, **kwargs):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(**kwargs)
        self.clip_min_value = clip_min_value

        mel_basis = librosa.filters.mel(
            sr=kwargs["sample_rate"],
            n_fft=kwargs["n_fft"],
            n_mels=kwargs["n_mels"],
            fmin=kwargs["f_min"],
            fmax=kwargs["f_max"],
        ).T
        self.transform.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int) -> torch.Tensor:
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = self.transform.forward(samples)
        samples.clamp_(min=self.clip_min_value)
        return samples


class Spectrogram(nn.Module):
    """
    Apply stft and magphase transformations
    """
    def __init__(self, n_fft, win_length, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transfrom
        :return: two tensors
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = torch.stft(samples, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        mag, phase = torchaudio.functional.magphase(spec)
        return mag, phase


class InverseSpectrogram(nn.Module):
    """
    Convert from magphase to complex and perform istft
    """
    def __init__(self, n_fft, win_length, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int = None) -> torch.Tensor:
        mag, phase = samples
        spec = torch.stack([torch.cos(phase) * mag, torch.sin(phase) * mag], dim=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = torch.istft(spec, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        return signal


class ToMono(nn.Module):
    """
    Convert stereo signal to mono
    """
    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int = None) -> torch.Tensor:
        """

        :param samples:
        :param sample_rate: dummy parameter for compatibility
        :return:
        """
        return torch.mean(samples, dim=0)


class Squeeze:
    """
    Transform to squeeze monochannel waveform
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return samples.squeeze(0)


class ToNumpy:
    """
    Transform to make numpy array
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return np.array(samples)


class ToTorch:
    """
    Transform to make torch.tensor
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return torch.tensor(samples)


class LogTransform(nn.Module):
    """
    Transform for taking logarithm of mel spectrograms (or anything else)
    :param fill_value: value to substitute non-positive numbers with before applying log
    """
    def __init__(self, fill_value: float = 1e-5) -> None:
        super().__init__()
        self.fill_value = fill_value

    def __call__(self, samples: torch.Tensor, sample_rate: int):
        samples = samples.masked_fill((samples <= 0), self.fill_value)
        return torch.log(samples)
