from pathlib import Path

import hydra
import torch
import torchaudio
from omegaconf import DictConfig

import core


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    audio_path = Path(hydra.utils.to_absolute_path(cfg.audio_to_spec.audio_path))
    output_path = Path(hydra.utils.to_absolute_path(cfg.audio_to_spec.output_path)) / (audio_path.stem + ".pt")

    inference_transforms = core.utils.get_transforms(cfg.inference_transforms)
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.transforms.Resample(sr, cfg.data.sample_rate)(wav)
    spectrogram = inference_transforms(samples=wav, sample_rate=sr)
    torch.save(spectrogram, output_path)


if __name__ == "__main__":
    main()
