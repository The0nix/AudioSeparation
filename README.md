# WaveNet
Implementation of [U-Net based Audio Separation Model](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf) in PyTorch

<!---
## Usage

### Setup
To launch training and inference in nvidia-docker container follow these instructions:

0. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
1. Run `./docker-build.sh`

### Training
To launch training follow these instructions:

1. Set preferred configurations in `config/config.yaml` in particular you might want to set dataset path (it will be concatendated with data path in `docker-train.sh`)
2. In `docker-run.sh` change `memory`, `memory-swap`, `shm-size`, `cpuset-cpus`, `gpus`, and data `volume` to desired values
3. Set WANDB_API_KEY environment variable to your wandb key
4. Run `./docker-train.sh`

All outputs including models will be saved to `outputs` dir.

### Inference
To launch inference run the following command:
```
./docker-inference.sh checkpoint.ckpt spectrogram.pt device
```
Where:
* `checkpoint.ckpt` is a path to .ckpt model file
* `spectrogram.pt`is a path to .pt file with spectrogram. Can be produced with `./docker-audio-to-spec.sh`
* `device` is the device to inference on: either 'cpu', 'cuda' or cuda device number

Resulted audio will be located in `inferenced` folder

### Audio to spectrogram transformer
To transform audio to spectrogram you can use the following command:
```
./docker-audio-to-spec.sh audio_path
```

Where:
* `audio_path` is a path to audio file

Spectrogram will be saved to `inferenced` folder with `.pt` extension

## Pretrained models
Pretrained model can be downloaded [here](https://drive.google.com/drive/folders/1PttatYjedJ8Qv4c5kdzcPNXwD6c3zIdH?usp=sharing).
-->
