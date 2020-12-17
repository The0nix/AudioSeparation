import hydra
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import DictConfig

import core


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    core.utils.fix_seeds(cfg.common.seed)

    # Define datasets and dataloaders:
    transforms = core.utils.get_transforms(cfg.train_transforms)
    inference_transforms = core.utils.get_transforms(cfg.inference_transforms)
    train_dataset = core.dataset.DSD100(root=hydra.utils.to_absolute_path(cfg.data.root),
                                        crop_size=cfg.preprocessing.crop_size,
                                        transforms=transforms)
    val_dataset = core.dataset.DSD100(root=hydra.utils.to_absolute_path(cfg.data.root),
                                      crop_size=cfg.preprocessing.val_crop_size,
                                      transforms=inference_transforms)

    # Split data with stratification
    train_idx, val_idx = core.utils.get_split(train_dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)
    train_dataset = torchdata.Subset(train_dataset, train_idx)
    val_dataset = torchdata.Subset(val_dataset, val_idx)

    # Create dataloaders
    train_sampler = torchdata.RandomSampler(train_dataset, replacement=True, num_samples=cfg.training.epoch_size)
    train_dataloader = torchdata.DataLoader(train_dataset,
                                            batch_size=cfg.training.batch_size,
                                            sampler=train_sampler,
                                            num_workers=cfg.training.num_workers)
    val_dataloader = torchdata.DataLoader(val_dataset,
                                          batch_size=cfg.training.batch_size,
                                          shuffle=False,
                                          num_workers=cfg.training.num_workers)

    # Define model
    spec_transform = core.transforms.Spectrogram(n_fft=cfg.preprocessing.n_fft,
                                                 win_length=cfg.preprocessing.win_length,
                                                 hop_length=cfg.preprocessing.hop_length)
    inv_spec_transform = core.transforms.InverseSpectrogram(n_fft=cfg.preprocessing.n_fft,
                                                            win_length=cfg.preprocessing.win_length,
                                                            hop_length=cfg.preprocessing.hop_length)
    if "checkpoint_path" in cfg.model:
        model = core.model.UNet(conv_channels_list=cfg.model.conv_channels_list,
                                kernel_size=cfg.model.kernel_size,
                                spec_transform=spec_transform,
                                inv_spec_transform=inv_spec_transform,
                                optimizer_lr=cfg.optimizer.lr)
        model.load_state_dict(torch.load(hydra.utils.to_absolute_path(cfg.model.checkpoint_path)))
    else:
        model = core.model.UNet(conv_channels_list=cfg.model.conv_channels_list,
                                kernel_size=cfg.model.kernel_size,
                                spec_transform=spec_transform,
                                inv_spec_transform=inv_spec_transform,
                                optimizer_lr=cfg.optimizer.lr)

    # Define logger and trainer
    wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb.project)
    wandb_logger.watch(model, log="gradients", log_freq=cfg.wandb.log_freq)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_weights_only=True,
                                                       save_last=True, save_top_k=1)
    trainer = pl.Trainer(max_epochs=cfg.training.n_epochs, gpus=cfg.training.gpus,
                         logger=wandb_logger, default_root_dir="checkpoints",
                         checkpoint_callback=checkpoint_callback,
                         val_check_interval=cfg.training.val_check_interval)

    # FIT IT!
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
