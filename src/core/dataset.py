import random
from pathlib import Path
from typing import Tuple

import torch
import torch.utils.data as torchdata
import torchaudio


class DSD100(torchdata.Dataset):
    """
    DSD100 dataset for audio source separation (2-stem)
    :param root: Path to the directory where the dataset is found.
    :param transforms: audiomentations transform object for waveform transform. Must end with Spectrogram transform
    :param use_cuda: If true, will move tensor to cuda before applying transforms. (default: False)
    """
    def __init__(self, root: str, transforms, crop_size=None, use_cuda=False) -> None:
        root = Path(root)
        assert root.is_dir(), f"Path does not exist or is not a directory: {root}"
        self.transforms = transforms
        self.crop_size = crop_size
        self.use_cuda = use_cuda

        self.paths = {name: sorted(root.glob(f"*/**/{name}.wav"))
                      for name in ["mixture", "vocals", "drums", "bass", "other"]}

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        :return: four tensors: magnitude and phase of mixture, magnitude and phase without vocals
        """
        if self.crop_size is not None:
            num_frames = self.crop_size
            info = torchaudio.info(str(self.paths["mixture"][idx]))[0]
            frame_offset = random.randrange(info.length // info.channels - num_frames)
        else:
            num_frames = -1
            frame_offset = 0

        mixture, sr = torchaudio.backend.sox_io_backend.load(
            self.paths["mixture"][idx], frame_offset=frame_offset, num_frames=num_frames
        )
        sources = [torchaudio.backend.sox_io_backend.load(
                       self.paths[source][idx], frame_offset=frame_offset, num_frames=num_frames
                   )[0]
                   for source in ["vocals"]]
        no_vocals = torch.stack(sources, dim=0).sum(dim=0)

        if self.use_cuda:
            mixture, no_vocals = mixture.cuda(), no_vocals.cuda()
        mixture = self.transforms(samples=mixture, sample_rate=sr)
        no_vocals = self.transforms(samples=no_vocals, sample_rate=sr)
        return mixture, no_vocals, sr

    def __len__(self) -> int:
        return len(self.paths["mixture"])
