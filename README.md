# TinySR: Efficient Image Restoration via Post-Training Quantization of SwinIR

**TinySR** builds on [SwinIR](https://arxiv.org/abs/2108.10257) by applying Post-Training Quantization (PTQ) to compress and accelerate the model with negligible quality loss.
No retraining is required — quantization is applied directly to the pretrained SwinIR weights.

---

## What We Did

SwinIR is a strong Swin Transformer-based image restoration model, but its FP32 weights are large (~56 MB for SwinIR-M) and inference is compute-intensive.
TinySR applies **weight-only PTQ** to all `nn.Linear` layers in the Attention and MLP blocks:

| Component | Quantization Target |
|:----------|:--------------------|
| `WindowAttention` | `qkv` (Linear), `proj` (Linear) |
| `Mlp` | `fc1` (Linear), `fc2` (Linear) |

Using [torchao](https://github.com/pytorch/ao) v0.15+, we apply INT8 weight-only quantization:

| Mode | Config | Memory Reduction | Quality Loss |
|:-----|:-------|:----------------:|:------------:|
| **INT8** | `Int8WeightOnlyConfig` | ~2x | < 0.01 dB |

Runs on **CUDA**. Weights are stored as INT8 and dequantized on-the-fly to FP16/BF16 before matrix multiplication, reducing memory bandwidth pressure with no change to activation precision.

> **Why not INT4?** torchao `Int4WeightOnlyConfig` requires `weight.shape[-1] % group_size == 0`
> for `group_size ∈ {32, 64, 128, 256}`. SwinIR uses `embed_dim=180`, which satisfies none of
> these constraints (e.g. `180 % 128 = 52`). All 144 Linear layers are silently skipped,
> resulting in no compression. INT8 is the effective quantization choice for SwinIR.

---

## Results

Testset: **Set5** for classical/lightweight SR; **Set12** for gray denoising; **McMaster** for color denoising; **classic5** for grayscale JPEG CAR; **RealSRSet+5images** for real-world SR (no-reference, no PSNR reported).

---

### Classical SR (DIV2K, patch=48) INT8, Set5

| Scale | FP32 PSNR | INT8 PSNR | ΔPSNR | FP32 Mem | INT8 Mem | Speedup |
|:-----:|:---------:|:---------:|:-----:|:--------:|:--------:|:-------:|
| ×2 | 36.15 dB | 36.15 dB | −0.0008 dB | 56.3 MB | 30.2 MB | 1.82× |
| ×3 | 32.88 dB | 32.88 dB | +0.0008 dB | 57.0 MB | 30.9 MB | 1.97× |
| ×4 | 30.80 dB | 30.79 dB | −0.0018 dB | 56.8 MB | 30.8 MB | 2.12× |
| ×8 | 25.65 dB | 25.65 dB | −0.0010 dB | 57.4 MB | 31.3 MB | 1.93× |

---

### Classical SR (DF2K, patch=64) INT8, Set5

| Scale | FP32 PSNR | INT8 PSNR | ΔPSNR | FP32 Mem | INT8 Mem | Speedup |
|:-----:|:---------:|:---------:|:-----:|:--------:|:--------:|:-------:|
| ×2 | 36.21 dB | 36.21 dB | −0.0014 dB | 64.2 MB | 38.1 MB | 1.65× |
| ×3 | 32.95 dB | 32.95 dB | +0.0004 dB | 64.9 MB | 38.8 MB | 1.72× |
| ×4 | 30.99 dB | 30.99 dB | +0.0006 dB | 64.7 MB | 38.6 MB | 2.04× |
| ×8 | 25.82 dB | 25.83 dB | +0.0055 dB | 65.3 MB | 39.2 MB | 2.05× |

---

### Lightweight SR (DIV2K, SwinIR-S) INT8, Set5

| Scale | FP32 PSNR | INT8 PSNR | ΔPSNR | FP32 Mem | INT8 Mem | Speedup |
|:-----:|:---------:|:---------:|:-----:|:--------:|:--------:|:-------:|
| ×2 | 35.95 dB | 35.95 dB | +0.0003 dB | 16.3 MB | 14.5 MB | 1.80× |
| ×3 | 32.61 dB | 32.61 dB | −0.0009 dB | 16.4 MB | 14.6 MB | 2.21× |
| ×4 | 30.52 dB | 30.51 dB | −0.0056 dB | 16.4 MB | 14.6 MB | 1.97× |

---

### Real-World SR (SwinIR-M ×4) INT8, RealSRSet+5images

No-reference task — no PSNR. Model compression and runtime:

| | FP32 | INT8 |
|:-:|:----:|:----:|
| Model size | 64.0 MB | 37.9 MB (1.69×) |
| Runtime | 0.229 s/img | 0.193 s/img (1.19× faster) |

---

### Real-World SR (SwinIR-L ×4) INT8, RealSRSet+5images

No-reference task — no PSNR. Model compression and runtime:

| | FP32 | INT8 |
|:-:|:----:|:----:|
| Model size | 135.9 MB | 65.9 MB (2.06×) |
| Runtime | 0.365 s/img | 0.315 s/img (1.16× faster) |

---

### Grayscale Denoising (SwinIR-M) INT8, Set12

Model: 117.2 MB → 91.1 MB (1.29× reduction)

| Noise σ | FP32 PSNR | INT8 PSNR | ΔPSNR | FP32 Runtime | INT8 Runtime |
|:-------:|:---------:|:---------:|:-----:|:------------:|:------------:|
| 15 | 33.36 dB | 33.36 dB | −0.0001 dB | 0.635 s/img | 0.686 s/img |
| 25 | 31.01 dB | 31.01 dB | −0.0004 dB | 0.666 s/img | 0.669 s/img |
| 50 | 27.91 dB | 27.91 dB | −0.0005 dB | 0.714 s/img | 0.869 s/img |

> Note: For denoising tasks with large images (Set12, McMaster), INT8 does not yield a runtime speedup due to memory-bound bottlenecks on this GPU.

---

### Color Denoising (SwinIR-M) INT8, McMaster

Model: 117.2 MB → 91.1 MB (1.29× reduction)

| Noise σ | FP32 PSNR | INT8 PSNR | ΔPSNR | FP32 Runtime | INT8 Runtime |
|:-------:|:---------:|:---------:|:-----:|:------------:|:------------:|
| 15 | 35.61 dB | 35.61 dB | −0.0007 dB | 1.276 s/img | 1.310 s/img |
| 25 | 33.31 dB | 33.31 dB | −0.0005 dB | 1.227 s/img | 1.273 s/img |
| 50 | 30.20 dB | 30.20 dB | −0.0014 dB | 1.278 s/img | 1.248 s/img |

---

### Grayscale JPEG CAR (SwinIR-M) INT8, classic5

Model: 98.1 MB → 72.0 MB (1.36× reduction)

| JPEG Quality | FP32 PSNR | INT8 PSNR | ΔPSNR | FP32 Runtime | INT8 Runtime |
|:------------:|:---------:|:---------:|:-----:|:------------:|:------------:|
| 10 | 30.27 dB | 30.27 dB | −0.0007 dB | 1.098 s/img | 1.101 s/img |
| 20 | 32.52 dB | 32.52 dB | −0.0003 dB | 1.174 s/img | 1.115 s/img |
| 30 | 33.74 dB | 33.74 dB | −0.0005 dB | 1.155 s/img | 1.078 s/img |
| 40 | 34.52 dB | 34.52 dB | +0.0004 dB | 1.041 s/img | 1.071 s/img |

---

## Installation

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
pip install torchao==0.15.0
pip install -r requirement.txt
```

Clone and set up testsets/model weights following the original SwinIR instructions.

---

## Usage

### FP32 Baseline (original SwinIR)

```bash
# Run all tasks (mirrors original run_tests.sh)
bash run_tests.sh
```

### PTQ Quantization Evaluation

```bash
# INT8 (default) — runs on GPU
bash run_quantize_tests.sh

# INT4
BITS=4 bash run_quantize_tests.sh

# Single task example: INT8, Color Denoising σ=25
python main_quantize_swinir.py \
    --task color_dn --noise 25 \
    --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth \
    --folder_gt testsets/McMaster \
    --bits 8

# INT4, Classical SR x4
python main_quantize_swinir.py \
    --task classical_sr --scale 4 --training_patch_size 48 \
    --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth \
    --folder_lq testsets/Set5/LR_bicubic/X4 \
    --folder_gt testsets/Set5/HR \
    --bits 4

# Skip FP32 baseline, only evaluate quantized model
python main_quantize_swinir.py --bits 8 --skip_fp32_eval \
    --task color_dn --noise 15 \
    --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
    --folder_gt testsets/McMaster
```

Quantized models are saved to:
- `model_zoo/swinir_quantized_int8/<model_name>.pth`
- `model_zoo/swinir_quantized_int4/<model_name>.pth`

### Tile-based Inference (large images)

```bash
# Use --tile to process large images in patches
BITS=8 TILE=256 bash run_quantize_tests.sh

python main_quantize_swinir.py --bits 8 --tile 256 --tile_overlap 32 \
    --task classical_sr --scale 4 --training_patch_size 48 \
    --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth \
    --folder_lq testsets/Set5/LR_bicubic/X4 \
    --folder_gt testsets/Set5/HR
```

---

## Key Files

| File | Description |
|:-----|:------------|
| `main_quantize_swinir.py` | PTQ script: applies torchao INT8/INT4 weight-only quantization, evaluates FP32 vs quantized PSNR/SSIM/runtime/memory |
| `run_quantize_tests.sh` | Batch evaluation script mirroring `run_tests.sh`, generates summary table |
| `main_test_swinir.py` | Original SwinIR inference script (unchanged) |
| `models/network_swinir.py` | SwinIR model definition (unchanged) |

---

## Output Format

`main_quantize_swinir.py` prints machine-parseable lines for use by `run_quantize_tests.sh`:

```
[FP32] Avg  PSNR: 40.12 dB  SSIM: 0.9611
[FP32] Avg  Runtime: 0.042 s/image
[FP32] Memory: 56.23 MB
[INT8] Avg  PSNR: 40.11 dB  SSIM: 0.9611
[INT8] Avg  Runtime: 0.038 s/image
[INT8] Memory: 30.14 MB
```

The shell script collects these across all tasks and prints a summary table with FP32 vs INT{N} PSNR, ΔPSNR, memory, and speedup.

---

## Acknowledgement

TinySR is built on top of [SwinIR](https://github.com/JingyunLiang/SwinIR) and uses [torchao](https://github.com/pytorch/ao) for quantization.
The original SwinIR model architecture and pretrained weights remain unchanged.

```
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
```
