from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torchaudio
from omegaconf import DictConfig

import core


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    spectrogram_path = Path(hydra.utils.to_absolute_path(cfg.inference.spectrogram_path))
    output_path = Path(hydra.utils.to_absolute_path(cfg.inference.inferenced_path))
    output_audio_path = output_path / (spectrogram_path.stem + ".wav")
    output_plot_path = output_path / (spectrogram_path.stem + ".png")

    model = core.model.WaveNet.load_from_checkpoint(hydra.utils.to_absolute_path(cfg.inference.checkpoint_path))
    model = model.eval().to(cfg.inference.device)

    spectrogram = torch.load(spectrogram_path).to(model.device)
    if len(spectrogram.shape) < 3:
        spectrogram = spectrogram.unsqueeze(0)
    if cfg.inference.cut_size is not None:
        spectrogram = spectrogram[:, :, :int(cfg.inference.cut_size * cfg.data.sample_rate / cfg.preprocessing.hop_length)]
    audio = model.inference(spectrogram).squeeze(0)
    audio = torchaudio.functional.mu_law_decoding(audio, model.n_mu_law)

    torchaudio.save(str(output_audio_path), audio.detach().cpu(), sample_rate=cfg.data.sample_rate)
    plt.plot(audio[0].detach().cpu().numpy())
    plt.savefig(output_plot_path)


if __name__ == "__main__":
    main()
