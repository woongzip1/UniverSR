# UniverSR
 
### Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching
 
[![arXiv](https://img.shields.io/badge/arXiv-2510.00771-b31b1b.svg)](https://arxiv.org/abs/2510.00771)
[![Demo](https://img.shields.io/badge/Demo-Page-blue.svg)](https://woongzip1.github.io/universr-demo/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woongzip1/UniverSR/blob/main/UniverSR_GUI.ipynb)
 
This is the official PyTorch implementation of *UniverSR: Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching* **(ICASSP 2026)**: [paper](https://arxiv.org/abs/2510.00771), [demo page](https://woongzip1.github.io/universr-demo/).
 
**UniverSR** performs audio super-resolution directly in the complex STFT domain using flow matching, without requiring a separate neural vocoder. A single model handles **multiple input sample rates (8 / 12 / 16 / 24 kHz → 48 kHz)** across speech, music, and sound effects.
 
<p align="center">
  <img src="assets/overview.png" width="720" />
</p>
 
---

## ⚙️ Installation

```bash
pip install git+https://github.com/woongzip1/UniverSR.git
```

Or install from source:

```bash
git clone https://github.com/woongzip1/UniverSR.git
cd UniverSR
pip install -e .
```

For training, install extra dependencies:

```bash
pip install -e ".[train]"
# or: pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0, CUDA ≥ 11.8

---

## 🚀 Quick Start

```python
import torchaudio
from universr import UniverSR

model = UniverSR.from_pretrained("woongzip1/universr-audio", device="cuda")

# Enhance a low-resolution audio file to 48 kHz
output = model.enhance("low_res_8k.wav", input_sr=8000)
torchaudio.save("output_48k.wav", output.cpu(), 48000)
```

### Inference Options

```python
output = model.enhance(
    "low_res.wav",
    input_sr=16000,          # 8000 / 12000 / 16000 / 24000
    ode_method="midpoint",   # 'euler', 'midpoint', 'rk4'
    ode_steps=4,             # number of ODE integration steps
    guidance_scale=1.5,      # classifier-free guidance (None = disabled)
)
```

### Handling 48 kHz Input with Limited Bandwidth

If your audio is already at 48 kHz but has limited bandwidth, set `input_sr` to the **sample rate** that matches the content range (Nyquist frequency extends up to `input_sr / 2` Hz). For instance, if content exists only up to 8 kHz, use `input_sr=16000`. UniverSR will automatically apply low-pass filtering to match the training pipeline before performing super-resolution.

```python
# File is 48 kHz, but content only up to 8 kHz → use input_sr=16000
output = model.enhance("fullband_file.wav", input_sr=16000)
```

---

## 🤗 Pretrained Models

| Model | Domain | HuggingFace |
|---|---|---|
| `universr-audio` | General audio (recommended) | [`woongzip1/universr-audio`](https://huggingface.co/woongzip1/universr-audio) |
| `universr-speech` | Speech only | [`woongzip1/universr-speech`](https://huggingface.co/woongzip1/universr-speech) |

To download a pretrained model locally:

```bash
# via CLI (requires: pip install huggingface_hub)
huggingface-cli download woongzip1/universr-audio --local-dir ./ckpts/universr-audio

# or in Python
from huggingface_hub import snapshot_download
snapshot_download("woongzip1/universr-audio", local_dir="./ckpts/universr-audio")
```

### Loading from a Local Checkpoint

```python
model = UniverSR.from_local(
    ckpt_path="ckpts/best_model.pth",
    config_path="configs/config.yaml",
    device="cuda",
)
```

---

## 🔊 Batch Inference

Enhance all `.wav` files in a folder:

```bash
python scripts/inference.py \
    --input_dir [YOUR INPUT DIR] \
    --output_dir results/enhanced/ \
    --model woongzip1/universr-audio \
    --input-sr 8000 \
    --ode-method midpoint \
    --ode-steps 4
```

From a local checkpoint:

```bash
python scripts/inference.py \
    --input_dir [YOUR INPUT DIR] \
    --output_dir results/enhanced/ \
    --ckpt ckpts/best_model.pth \
    --config configs/config.yaml \
    --input-sr 8000 \
    --ode-method midpoint \
    --ode-steps 4
```

---

## 📊 Evaluation

### Compute Metrics (Generated vs. Ground Truth)

Compare enhanced outputs against ground-truth references using Log-Spectral Distance (LSD):

```bash
python scripts/compute_metrics.py \
    --reference_dir data/gt_48k/ \
    --output_dir results/enhanced/ \
    --cutoff-sr 8 \
    --output_json results/metrics.json
```

### Full Pipeline Evaluation

Run evaluation using the training data pipeline with on-the-fly LR generation:

```bash
python evaluate.py \
    -c configs/config.yaml \
    --ckpt ckpts/best_model.pth \
    --sampling-rate 8 \
    --wandb true
```

---

## 🏋️ Training

### 1. Prepare File Lists

UniverSR expects **48 kHz mono `.wav` files** listed in a text file (one absolute path per line). Use the provided script to generate these lists from your audio directories:

```bash
python scripts/prepare_file_list.py \
    --dirs /path/to/dataset1 /path/to/dataset2 \
    --output data/train.txt
```

Or use `find` directly in your terminal:

```bash
find /path/to/dataset -name "*.wav" > data/train.txt
```

### 2. Configure Training

Point the config to your file lists:

```yaml
# configs/config.yaml
dataset:
  common:
    sr: 48000
    num_samples: 32767
  train:
    file_list: "./data/train.txt"
  val:
    file_list: "./data/val.txt"
```

Low-resolution inputs are generated on-the-fly during training via low-pass filtering and downsampling. The input sample rate distribution is controlled by `collator.sampling_rates_probs`:

```yaml
collator:
  sampling_rates_probs:
    8: 0.7    # 70% of batches at 8 kHz
    12: 0.1   # 10% at 12 kHz
    16: 0.1   # 10% at 16 kHz
    24: 0.1   # 10% at 24 kHz
```

### 3. Start Training

```bash
python train.py -c configs/config.yaml --wandb true
```

> **Note**: Please refer to the paper for dataset details. The pipeline works with any 48 kHz audio collection.

---

## ⚠️ Known Limitations & Tips

### Performance across input rates

The default training distribution allocates 70% of batches to 8 kHz and 10% each to 12/16/24 kHz, so **8 kHz→48 kHz** results tend to be strongest. Higher input rates (especially 24 kHz) may show weaker high-frequency reconstruction due to spectral aliasing near the cutoff introduced by the naive downsample-upsample augmentation pipeline.

**To improve higher input rates**, consider:
- Retraining with a more balanced `sampling_rates_probs` distribution.
- Using slightly lower `sr_to_lr_bins` values for 12/16/24 kHz (e.g. 120/160/240 instead of 128/170/256) to give the model a small overlap margin at the spectral boundary.

### Guidance scale recommendations

The classifier-free guidance (CFG) scale trades off objective fidelity with perceptual richness:

| Domain | Recommended `guidance_scale` |
|---|---|
| Speech | 1.0 – 1.5 |
| Music | 1.5 – 2.0 |
| Sound effects | 1.5 |

Higher values produce denser high-frequency structures but deviate more from the ground-truth reference. Setting `guidance_scale=None` or `0` disables CFG entirely.

---

## 📖 Citation and License

We've released our code under the [MIT License](LICENSE). If you find UniverSR useful in your research, please consider citing:

```bibtex
@article{choi2025universr,
  title={{UniverSR}: Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching},
  author={Choi, Woongjib and Lee, Sangmin and Lim, Hyungseob and Kang, Hong-Goo},
  journal={arXiv preprint arXiv:2510.00771},
  year={2025}
}

@inproceedings{choi2026universr,
  title     = {{UniverSR}: Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching},
  author    = {Choi, Woongjib and Lee, Sangmin and Lim, Hyungseob and Kang, Hong-Goo},
  booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year      = {2026}
}

```

## 📝 Changelog
### v0.1.2
- Added Colab notebook (`UniverSR_GUI.ipynb`) with chunked processing support
- Fixes crash for audio files longer than 2 minutes on free Colab (T4)

### v0.1.1
- Fixed short audio crash: added minimum length guard in `enhance()`

## 🙏 Acknowledgments

This project was developed at [DSPAI Lab, Yonsei University](http://dsp.yonsei.ac.kr/).
We thank the following open-source projects that inspired parts of our codebase:

- [MIT 6.S184: Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)
- [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution)
- [FlashSR](https://github.com/jakeoneijk/FlashSR_Inference)
- [FLowHigh](https://github.com/jjunak-yun/FLowHigh_code)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)
- [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
