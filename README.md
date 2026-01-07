# Bridging Semantic Scale Gaps in Image Transmission Through Multi-scale Joint Perception and Generation

> PyTorch implementation of the [paper](https://ieeexplore.ieee.org/document/11096614) "Bridging Semantic Scale Gaps in Image Transmission Through Multi-scale Joint Perception and Generation".

## Installation

We implement BriGSC under python 3.10 and PyTorch 2.5.0.

## Usage

Pretrained BriGSC models are in [Google Drive](https://drive.google.com/drive/folders/1qP1an-Q1scBoldIJE2AJDh--uCLBbSZL?usp=sharing).

### Train

#### First stage training

```bash
e.g.
python train.py --trainset DIV2K --testset kodak --model BriGSC_W/O --channel-type awgn --distortion-metric MSE --comp_size 2048 --multiple-snr 1
```

#### Second stage training

```bash
e.g.
python train.py --trainset DIV2K --testset kodak --model BriGSC --channel-type awgn --distortion-metric MSE --comp_size 2048 --multiple-snr 1  --first_stage_ckpt history/512-1/models/train_best.model
```

### Inference

```bash
e.g.
python inference.py --datasets datasets/kodak --model BriGSC --channel-type awgn --distortion-metric MSE --comp_size 2048 --multiple-snr 10
```