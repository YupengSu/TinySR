# TinySR: Efficient Image Restoration via Compression of Swin Transformers

**TinySR** builds on [SwinIR](https://arxiv.org/abs/2108.10257) with two post-training compression methods — quantization and structured pruning — applied directly to pretrained weights. No retraining required.

---

## Methods

Both methods target the same `nn.Linear` layers inside each Swin Transformer block:

| Component | Layers |
|:----------|:-------|
| `WindowAttention` | `qkv`, `proj` |
| `Mlp` | `fc1`, `fc2` |

### Quantization (PTQ)

Weight-only INT8 quantization via [torchao](https://github.com/pytorch/ao) `Int8WeightOnlyConfig`. Weights are stored as INT8 and dequantized on-the-fly to FP16/BF16 at runtime — activations stay full precision. Requires **CUDA**.

> **Why not INT4?** `Int4WeightOnlyConfig` requires `weight.shape[-1] % group_size == 0` for
> `group_size ∈ {32, 64, 128, 256}`. SwinIR's `embed_dim=180` is divisible by none of these
> (`180 % 128 = 52`), so all 144 Linear layers are silently skipped. Use INT8.

### Structured Pruning

One-shot L1-magnitude pruning, no fine-tuning. Two optional targets:

| Target | What is removed | Scoring |
|:-------|:----------------|:--------|
| MLP hidden channels | `fc1` output rows + matching `fc2` columns | L1 norm of `fc1` rows |
| Attention heads | Full head slices from `qkv` + `proj` columns | Sum of L1 norms of Q/K/V + proj slices |

Input/output dimensions are unchanged (residual connections preserved). Attention head pruning patches each block's `forward()` with the original fixed `head_dim` to avoid the `C // num_heads` reshape mismatch.

> Post-training pruning causes measurable PSNR loss without fine-tuning, especially for SR.
> JPEG CAR is the most robust task (~0.75 dB drop). Fine-tuning after pruning recovers most of the gap.

### At a Glance

| Method | Config | Param reduction | Mem reduction | Quality loss |
|:-------|:-------|:---------------:|:-------------:|:------------:|
| INT8 | `Int8WeightOnlyConfig` | none | ~2× | < 0.01 dB |
| Pruned | MLP 75% + Attn 4 heads | ~1.30× | ~1.19× | 0.7 – 11 dB† |

†Varies by task; JPEG CAR ≈ 0.75 dB, SR without fine-tuning up to −11 dB.

---

## Results

**Testsets:** Set5 (classical/lightweight SR) · Set12 (grayscale denoising) · McMaster (color denoising) · classic5 (grayscale JPEG CAR) · RealSRSet+5images (real-world SR, no-reference)

Pruning config: `--mlp_keep_ratio 0.75 --attn_keep_heads 4` (360→270 hidden channels, 6→4 heads per block, ~11.5M→8.8M params).

---

### Classical SR (DIV2K, patch=48) — Set5

**INT8** · FP32 mem ~56–57 MB → INT8 ~30–31 MB (~1.85×)

| Scale | FP32 PSNR | INT8 PSNR | ΔPSNR | Speedup |
|:-----:|:---------:|:---------:|:-----:|:-------:|
| ×2 | 36.15 dB | 36.15 dB | −0.001 dB | 1.82× |
| ×3 | 32.88 dB | 32.88 dB | +0.001 dB | 1.97× |
| ×4 | 30.80 dB | 30.79 dB | −0.002 dB | 2.12× |
| ×8 | 25.65 dB | 25.65 dB | −0.001 dB | 1.93× |

**Pruned** · FP32 mem ~56–57 MB → Pruned ~46–47 MB (~1.24×) · Params: ~11.9M → ~9.1M (1.30×)

| Scale | FP32 PSNR | Pruned PSNR | ΔPSNR | Speedup |
|:-----:|:---------:|:-----------:|:-----:|:-------:|
| ×2 | 36.15 dB | 33.50 dB | −2.65 dB | 1.18× |
| ×3 | 32.88 dB | 27.98 dB | −4.90 dB | 2.16× |
| ×4 | 30.80 dB | 25.42 dB | −5.38 dB | 2.39× |
| ×8 | 25.65 dB | 21.88 dB | −3.77 dB | 2.70× |

---

### Classical SR (DF2K, patch=64) — Set5

**INT8** · FP32 mem ~64–65 MB → INT8 ~38–39 MB (~1.68×)

| Scale | FP32 PSNR | INT8 PSNR | ΔPSNR | Speedup |
|:-----:|:---------:|:---------:|:-----:|:-------:|
| ×2 | 36.21 dB | 36.21 dB | −0.001 dB | 1.65× |
| ×3 | 32.95 dB | 32.95 dB | +0.000 dB | 1.72× |
| ×4 | 30.99 dB | 30.99 dB | +0.001 dB | 2.04× |
| ×8 | 25.82 dB | 25.83 dB | +0.006 dB | 2.05× |

**Pruned** · FP32 mem ~64–65 MB → Pruned ~54–55 MB (~1.19×) · Params: ~11.9M → ~9.1M (1.30×)

| Scale | FP32 PSNR | Pruned PSNR | ΔPSNR | Speedup |
|:-----:|:---------:|:-----------:|:-----:|:-------:|
| ×2 | 36.21 dB | 35.31 dB | −0.90 dB | 1.75× |
| ×3 | 32.95 dB | 26.84 dB | −6.11 dB | 2.21× |
| ×4 | 30.99 dB | 23.10 dB | −7.89 dB | 1.74× |
| ×8 | 25.82 dB | 14.48 dB | −11.34 dB | 2.74× |

---

### Lightweight SR (DIV2K, SwinIR-S) — Set5

**INT8** · FP32 mem ~16.4 MB → INT8 ~14.6 MB (~1.12×)

| Scale | FP32 PSNR | INT8 PSNR | ΔPSNR | Speedup |
|:-----:|:---------:|:---------:|:-----:|:-------:|
| ×2 | 35.95 dB | 35.95 dB | +0.000 dB | 1.80× |
| ×3 | 32.61 dB | 32.61 dB | −0.001 dB | 2.21× |
| ×4 | 30.52 dB | 30.51 dB | −0.006 dB | 1.97× |

**Pruned** · FP32 mem ~16.4 MB → Pruned ~15.6 MB (~1.05×) · Params: ~0.92M → ~0.71M (1.30×)

| Scale | FP32 PSNR | Pruned PSNR | ΔPSNR | Speedup |
|:-----:|:---------:|:-----------:|:-----:|:-------:|
| ×2 | 35.95 dB | 33.41 dB | −2.54 dB | 2.24× |
| ×3 | 32.61 dB | 31.13 dB | −1.48 dB | 2.69× |
| ×4 | 30.52 dB | 26.94 dB | −3.58 dB | 2.73× |

---

### Real-World SR ×4 — RealSRSet+5images (no-reference)

| | FP32 | INT8 | Pruned |
|:-|:----:|:----:|:------:|
| **SwinIR-M** mem | 64.0 MB | 37.9 MB (1.69×) | 53.5 MB (1.20×) |
| **SwinIR-M** params | 11.72M | 11.72M | 8.97M (1.31×) |
| **SwinIR-M** runtime | 0.233 s/img | 0.193 s/img (1.21×) | 0.195 s/img (1.19×) |
| **SwinIR-L** mem | 135.9 MB | 65.9 MB (2.06×) | 100.0 MB (1.36×) |
| **SwinIR-L** params | 28.01M | 28.01M | 18.61M (1.50×) |
| **SwinIR-L** runtime | 0.361 s/img | 0.315 s/img (1.15×) | 0.287 s/img (1.26×) |

---

### Grayscale Denoising (SwinIR-M) — Set12

Model: 117.2 MB → INT8 91.1 MB (1.29×) · Pruned 106.7 MB (1.10×) · Params: 11.50M → 8.75M (1.31×)

> Note: denoising on large images (Set12, McMaster) is memory-bound on this GPU — neither INT8 nor Pruned yields a consistent runtime speedup.

| Noise σ | FP32 PSNR | INT8 PSNR | ΔINT8 | Pruned PSNR | ΔPruned |
|:-------:|:---------:|:---------:|:-----:|:-----------:|:-------:|
| 15 | 33.36 dB | 33.36 dB | −0.000 dB | 31.01 dB | −2.35 dB |
| 25 | 31.01 dB | 31.01 dB | −0.000 dB | 27.52 dB | −3.49 dB |
| 50 | 27.91 dB | 27.91 dB | −0.001 dB | 21.42 dB | −6.49 dB |

---

### Color Denoising (SwinIR-M) — McMaster

Model: 117.2 MB → INT8 91.1 MB (1.29×) · Pruned 106.7 MB (1.10×) · Params: 11.50M → 8.76M (1.31×)

| Noise σ | FP32 PSNR | INT8 PSNR | ΔINT8 | Pruned PSNR | ΔPruned |
|:-------:|:---------:|:---------:|:-----:|:-----------:|:-------:|
| 15 | 35.61 dB | 35.61 dB | −0.001 dB | 30.46 dB | −5.15 dB |
| 25 | 33.31 dB | 33.31 dB | −0.001 dB | 26.87 dB | −6.44 dB |
| 50 | 30.20 dB | 30.20 dB | −0.001 dB | 23.35 dB | −6.85 dB |

---

### Grayscale JPEG CAR (SwinIR-M) — classic5

Model: 98.1 MB → INT8 72.0 MB (1.36×) · Pruned 87.6 MB (1.12×) · Params: 11.49M → 8.74M (1.31×)

| JPEG Quality | FP32 PSNR | INT8 PSNR | ΔINT8 | Pruned PSNR | ΔPruned |
|:------------:|:---------:|:---------:|:-----:|:-----------:|:-------:|
| 10 | 30.27 dB | 30.27 dB | −0.001 dB | 29.45 dB | −0.82 dB |
| 20 | 32.52 dB | 32.52 dB | −0.000 dB | 31.73 dB | −0.79 dB |
| 30 | 33.74 dB | 33.74 dB | −0.001 dB | 32.96 dB | −0.78 dB |
| 40 | 34.52 dB | 34.52 dB | +0.000 dB | 33.81 dB | −0.71 dB |

---

## Installation

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
pip install torchao==0.15.0
pip install -r requirement.txt
```

Clone and set up testsets/model weights following the original [SwinIR instructions](https://github.com/JingyunLiang/SwinIR).

---

## Usage

### FP32 Baseline

```bash
bash run_tests.sh
```

### Quantization

```bash
# All tasks, INT8 (default)
bash run_quantize_tests.sh

# All tasks, INT8 with tile inference (for large images)
BITS=8 TILE=256 bash run_quantize_tests.sh

# Single task
python main_quantize_swinir.py \
    --task color_dn --noise 25 \
    --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth \
    --folder_gt testsets/McMaster \
    --bits 8

# Skip FP32 baseline
python main_quantize_swinir.py --bits 8 --skip_fp32_eval \
    --task color_dn --noise 15 \
    --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
    --folder_gt testsets/McMaster
```

Quantized models saved to `model_zoo/swinir_quantized_int8/<model_name>.pth`.

### Pruning

```bash
# All tasks, MLP 75% only (default)
bash run_prune_tests.sh

# All tasks, MLP 75% + 4 attention heads (replicates results above)
MLP=0.75 ATTN=4 bash run_prune_tests.sh

# All tasks with tile inference
MLP=0.75 ATTN=4 TILE=256 bash run_prune_tests.sh

# Single task
python main_prune_swinir.py \
    --task jpeg_car --jpeg 40 \
    --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth \
    --folder_gt testsets/classic5 \
    --mlp_keep_ratio 0.75 --attn_keep_heads 4

# MLP pruning only (no attention head pruning)
python main_prune_swinir.py \
    --task color_dn --noise 15 \
    --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
    --folder_gt testsets/McMaster \
    --mlp_keep_ratio 0.75

# Skip FP32 baseline, custom save path
python main_prune_swinir.py --skip_fp32_eval \
    --task color_dn --noise 25 \
    --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth \
    --mlp_keep_ratio 0.75 --attn_keep_heads 4 \
    --save_path model_zoo/swinir_pruned/my_model.pth
```

Pruned models saved to `model_zoo/swinir_pruned/<model_name>_mlp{N}_attn{K}h.pth`.

---

## Key Files

| File | Description |
|:-----|:------------|
| `main_quantize_swinir.py` | INT8/INT4 PTQ via torchao; evaluates FP32 vs quantized PSNR/SSIM/runtime/memory |
| `run_quantize_tests.sh` | Batch quantization runner; prints summary table |
| `main_prune_swinir.py` | One-shot L1 magnitude pruning of MLP channels and/or attention heads; evaluates FP32 vs pruned PSNR/params/runtime |
| `run_prune_tests.sh` | Batch pruning runner; prints summary table with param counts and speedup |
| `main_test_swinir.py` | Original SwinIR inference (unchanged) |
| `models/network_swinir.py` | SwinIR model definition (unchanged) |

Both scripts emit machine-parseable tagged lines (`[FP32]`, `[INT8]`, `[Pruned]`) that the shell scripts grep to build the summary table.

---

## Acknowledgement

Built on [SwinIR](https://github.com/JingyunLiang/SwinIR). Quantization uses [torchao](https://github.com/pytorch/ao); pruning uses native PyTorch. Pretrained weights and model architecture are unchanged.

```
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
```
