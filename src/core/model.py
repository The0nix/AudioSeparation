
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

import core.transforms


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout=0, activation="relu"):
        super().__init__()
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "sigmoid":
            activation_layer = nn.Sigmoid()
        else:
            raise ValueError(f"Invalid activation in UNetDeconvBlock: {activation}")
        if dropout:
            dropout_layer = nn.Dropout2d()
        else:
            dropout_layer = nn.Identity()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2,
                               padding=kernel_size // 2, output_padding=1),
            nn.BatchNorm2d(out_channels),
            dropout_layer,
            activation_layer,
        )

    def forward(self, x, residual=None):
        """
        Concatenate x with residual and deconvolve
        :param x: torch.tensor of shape (1, n_fft, seq_len)
        :param residual: torch.tensor of shape (1, n_fft, seq_len) or None
        :return:
        """
        if residual is not None:
            x = torch.cat([x, residual], dim=1)
        return self.conv(x)


class UNet(pl.LightningModule):
    """
    UNet for source separation from https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
    :param conv_channels_list: List of number of channels in convolutional blocks
    :param spec_transform: Spectrogram transform with proper parameters from core.transforms
    :param inv_spec_transform: InverseSpectrogram transform with proper parameters from core.transforms
    :param optimizer_lr: Learning rate for Adam optimizer
    """
    def __init__(self, conv_channels_list: List[int], kernel_size: int,
                 spec_transform: core.transforms.Spectrogram, inv_spec_transform: core.transforms.InverseSpectrogram,
                 optimizer_lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.spec_transform = spec_transform
        self.inv_spec_transform = inv_spec_transform
        self.optimizer_lr = optimizer_lr

        conv_blocks = []
        for in_c, out_c in zip([1] + conv_channels_list[:-1], conv_channels_list):
            conv_blocks.append(UNetConvBlock(in_c, out_c, kernel_size))
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # First block does not have residual
        deconv_blocks = []
        for i, (in_c, out_c) in enumerate(zip(conv_channels_list[::-1], conv_channels_list[::-1][1:] + [1])):
            in_c = in_c * 2 if i > 0 else in_c
            dropout = 0.5 if i < 3 else 0
            activation = "relu" if i < len(conv_channels_list) - 1 else "sigmoid"
            deconv_blocks.append(UNetDeconvBlock(in_c, out_c, kernel_size, dropout=dropout, activation=activation))
        self.deconv_blocks = nn.ModuleList(deconv_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates forward pass.
        :param x: tensor of shape (bs, n_fft, spec_len) -- input spectrogram
        :return: tensor of shape (bs, n_fft, seq_len) -- output mask
        """
        # x = (x - x.min()) / x.max()
        x = x.unsqueeze(1)
        residuals = []
        for block in self.conv_blocks:
            x = block(x)
            residuals.append(x)
        residuals[-1] = None  # For first deconv block
        residuals = list(reversed(residuals))

        for block, residual in zip(self.deconv_blocks, residuals):
            x = block(x, residual)

        x = x.squeeze(1)
        return x
    
    # def inference(self, spectrogram):
    #     """
    #     Calculates waveform from spectrogram
    #     :param spectrogram:
    #     :return: tensor of shape (bs, 1, seq_len) -- predicted waveform
    #     """
    #     spectrogram = self.melspec_upsampler(spectrogram)
    #     bs, _, seq_len = spectrogram.shape
    #
    #     cur_waveform = torch.zeros([bs, 1, 1], device=self.device)
    #     while cur_waveform.shape[2] < spectrogram.shape[2]:
    #         print(f"{cur_waveform.shape[2]}/{spectrogram.shape[2]}")
    #         input_waveform = self.input_conv(cur_waveform[:, :, -self.receptive_field:])
    #         cur_result = torch.zeros_like(input_waveform, device=self.device)
    #         prev = input_waveform
    #         for block in self.residual_blocks:
    #             block_result = block(prev, spectrogram[:, :, max(0, cur_waveform.shape[2] -self.receptive_field):cur_waveform.shape[2]])
    #             cur_result += block_result
    #             block_result += prev
    #             prev = block_result
    #         cur_result = self.output_conv(cur_result)
    #         cur_result = cur_result[:, :, [-1]].argmax(dim=1, keepdim=True)
    #         cur_waveform = torch.cat([cur_waveform, cur_result], dim=2)
    #
    #     return cur_waveform

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, List[int]],
             batch_idx: int, inference: bool) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Pass batch to network, calculate losses and return total loss with gt and predicted spectrograms
        """
        mixture, no_vocals, sr = batch
        mixture_mag, mixture_phase = self.spec_transform(mixture)
        no_vocals_mag, no_vocals_phase = self.spec_transform(no_vocals)
        pred_mag = self(torch.log(mixture_mag.masked_fill((mixture_mag <= 0), 1e-5))) * mixture_mag

        loss = nn.L1Loss()(pred_mag, no_vocals_mag)

        return loss, pred_mag, mixture_mag, mixture_phase, no_vocals_mag, no_vocals_phase, sr

    def training_step(self, batch, batch_idx):
        loss, *_, = self.step(batch, batch_idx, inference=False)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Calculate losses and results in training and inference modes
        loss, pred_mag, mixture_mag, mixture_phase, no_vocals_mag, no_vocals_phase, sr = \
            self.step(batch, batch_idx, inference=False)
        pred_waveform = self.inv_spec_transform((pred_mag, mixture_phase))
        mixed_waveform = self.inv_spec_transform((mixture_mag, mixture_phase))
        instrument_waveform = self.inv_spec_transform((no_vocals_mag, no_vocals_phase))

        mixed_audio = [wandb.Audio(wav.detach().cpu(), sample_rate=sr[0]) for wav in mixed_waveform]
        pred_audio = [wandb.Audio(wav.detach().cpu(), sample_rate=sr[0]) for wav in pred_waveform]
        instrument_audio = [wandb.Audio(wav.detach().cpu(), sample_rate=sr[0]) for wav in instrument_waveform]

        self.logger.experiment.log({"Mixed audio": mixed_audio,
                                    "Predicted audio": pred_audio,
                                    "True audio": instrument_audio}, commit=False)
        self.log("val_loss", loss)

        return loss, pred_mag, mixture_mag, mixture_phase, no_vocals_mag, no_vocals_phase

    # def validation_epoch_end(self, outputs):
    #     val_loss, true_waveform, val_pred_waveform, spectrogram = outputs[-1]
    #     inf_waveform = self.inference(spectrogram[:4])
    #     inf_waveform = torchaudio.functional.mu_law_decoding(inf_waveform, self.n_mu_law).squeeze(1)
    #
    #     inf_audio = [wandb.Audio(wav.detach().cpu(), sample_rate=22050) for wav in inf_waveform]
    #     self.logger.experiment.log({"Inferenced audio": inf_audio}, commit=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "monitor": "val_loss",
        }
