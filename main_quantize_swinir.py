#!/usr/bin/env python3
"""
main_quantize_swinir.py  --  Post-Training Quantization (PTQ) for SwinIR

Quantization targets (all nn.Linear layers in the network):
  MLP      : Mlp.fc1, Mlp.fc2
  Attention: WindowAttention.qkv, WindowAttention.proj

Uses torchao dynamic (weight-only) quantization.  No calibration data needed.

  --bits 8  : Int8WeightOnlyConfig  (default)
  --bits 4  : Int4WeightOnlyConfig  (smaller, CUDA-optimized)

Usage examples
  # INT8 weight-only PTQ
  python main_quantize_swinir.py \
      --task color_dn --noise 15 \
      --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
      --folder_gt testsets/McMaster

  # INT4 weight-only PTQ
  python main_quantize_swinir.py --bits 4 \
      --task color_dn --noise 15 \
      --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
      --folder_gt testsets/McMaster

  # Also quantize Conv2d, skip FP32 baseline
  python main_quantize_swinir.py --bits 8 --quant_conv --skip_fp32_eval \
      --task color_dn --noise 15 \
      --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth
"""

import argparse
import copy
import cv2
import glob
import io
import numpy as np
from collections import OrderedDict
import os
import time
import torch
import torch.nn as nn

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util
from main_test_swinir import define_model, setup, get_image_pair, test


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    """Estimate serialized model size in MB via in-memory torch.save."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 / 1024


def count_linear_layers(model: nn.Module):
    """Return (total Linear layers, names list) in the model."""
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    return len(names), names


# ─────────────────────────────────────────────────────────────────────────────
# Quantization
# ─────────────────────────────────────────────────────────────────────────────

def _count_quantized_linear(model: nn.Module) -> int:
    """Count Linear layers whose weight is no longer a plain nn.Parameter (i.e. quantized)."""
    return sum(
        1 for _, m in model.named_modules()
        if isinstance(m, nn.Linear) and type(m.weight).__name__ != 'Parameter'
    )


def apply_dynamic_ptq(model: nn.Module, bits: int = 8, quant_conv: bool = False) -> nn.Module:
    """
    Dynamic (weight-only) quantization via torchao.  No calibration data needed.

    bits=8 : Int8WeightOnlyConfig  -- W8, activations stay float
    bits=4 : Int4WeightOnlyConfig  -- W4 (requires weight last-dim divisible by group_size=128)

    Conv2d is included only when quant_conv=True (Linear only by default).

    Note on INT4: torchao silently skips layers whose weight shape is incompatible with
    group_size=128.  SwinIR-M uses embed_dim=180 (180 % 128 = 52), so ALL layers are
    skipped and the model is NOT actually quantized.  Use --bits 8 for SwinIR.
    """
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig

    config_map = {8: Int8WeightOnlyConfig(), 4: Int4WeightOnlyConfig()}
    if bits not in config_map:
        raise ValueError(f'--bits must be 4 or 8, got {bits}')

    model = copy.deepcopy(model)
    target_types = (nn.Linear, nn.Conv2d) if quant_conv else (nn.Linear,)
    n_linear_before = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))

    def _filter(module, fqn):
        return isinstance(module, target_types)

    quantize_(model, config_map[bits], filter_fn=_filter)

    n_quantized = _count_quantized_linear(model)
    target_str = 'nn.Linear + nn.Conv2d' if quant_conv else 'nn.Linear'

    if n_quantized == 0 and n_linear_before > 0:
        if bits == 4:
            # Diagnose the incompatibility
            sample = next(
                (m for _, m in model.named_modules() if isinstance(m, nn.Linear)), None
            )
            hint = ''
            if sample is not None:
                group_size = Int4WeightOnlyConfig().group_size
                last_dim = sample.weight.shape[-1]
                hint = (f' (weight last-dim={last_dim}, group_size={group_size}, '
                        f'{last_dim}%{group_size}={last_dim % group_size} ≠ 0)')
            raise RuntimeError(
                f'INT4 quantization skipped ALL {n_linear_before} Linear layers{hint}.\n'
                f'torchao Int4WeightOnlyConfig requires weight.shape[-1] % group_size == 0.\n'
                f'SwinIR embed_dim=180 is not divisible by any supported group_size '
                f'(32, 64, 128, 256).  Use --bits 8 instead.'
            )
        raise RuntimeError(
            f'INT{bits} quantization applied to 0 out of {n_linear_before} Linear layers. '
            'Check model compatibility.'
        )

    print(f'[torchao] INT{bits} weight-only: {n_quantized}/{n_linear_before} {target_str} quantized')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_folder(model: nn.Module, folder: str, args, window_size: int,
                    device: torch.device, tag: str = '', save_dir: str = None) -> dict:
    """
    Run inference on every image in folder and return average metrics + timing.
    Returns dict with keys: psnr, ssim, psnr_y, ssim_y, avg_time_s
    """
    results = {'psnr': [], 'ssim': [], 'psnr_y': [], 'ssim_y': [], 'time_s': []}
    paths = sorted(glob.glob(os.path.join(folder, '*')))
    if not paths:
        print(f'[WARN] No images found in {folder}')
        return results

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for idx, path in enumerate(paths):
        imgname, img_lq, img_gt = get_image_pair(args, path)
        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
        )
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, h_old, w_old = img_lq.shape
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq_pad = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq_pad = torch.cat([img_lq_pad, torch.flip(img_lq_pad, [3])], 3)[:, :, :, :w_old + w_pad]
            t0 = time.perf_counter()
            output = test(img_lq_pad, model, args, window_size)
            results['time_s'].append(time.perf_counter() - t0)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f'{imgname}_SwinIR.png'), output)

        if img_gt is None:
            print(f'[{tag}] {idx:3d} {imgname:20s}')
            continue

        img_gt = (img_gt * 255.0).round().astype(np.uint8)
        img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]
        img_gt = np.squeeze(img_gt)

        border = args.scale if args.task in ['classical_sr', 'lightweight_sr'] else 0
        psnr = util.calculate_psnr(output, img_gt, crop_border=border)
        ssim = util.calculate_ssim(output, img_gt, crop_border=border)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

        row = f'[{tag}] {idx:3d} {imgname:20s}  PSNR: {psnr:.2f} dB  SSIM: {ssim:.4f}'

        if img_gt.ndim == 3:
            psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
            ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
            results['psnr_y'].append(psnr_y)
            results['ssim_y'].append(ssim_y)
            row += f'  PSNR_Y: {psnr_y:.2f} dB  SSIM_Y: {ssim_y:.4f}'

        print(row)

    avg_t = float(np.mean(results['time_s'])) if results['time_s'] else 0.0

    if results['psnr']:
        print(f'\n[{tag}] Avg  PSNR: {np.mean(results["psnr"]):.2f} dB'
              f'  SSIM: {np.mean(results["ssim"]):.4f}')
        if results['psnr_y']:
            print(f'[{tag}] Avg  PSNR_Y: {np.mean(results["psnr_y"]):.2f} dB'
                  f'  SSIM_Y: {np.mean(results["ssim_y"]):.4f}')

    # Machine-parseable summary line (grep target for shell script)
    print(f'[{tag}] Avg  Runtime: {avg_t:.3f} s/image')

    out = {k: (float(np.mean(v)) if v else None) for k, v in results.items() if k != 'time_s'}
    out['avg_time_s'] = avg_t
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_quantized_model(model: nn.Module, save_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({'params': model.state_dict()}, save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f'Saved quantized model → {save_path}  ({size_mb:.1f} MB on disk)')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Post-Training Quantization (PTQ) for SwinIR',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Task / model args (mirrors main_test_swinir.py) ──────────────────────
    parser.add_argument('--task', type=str, default='color_dn',
                        help='classical_sr | lightweight_sr | real_sr | '
                             'gray_dn | color_dn | jpeg_car | color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1,
                        help='Scale factor: 1, 2, 3, 4, 8  (1 for dn/jpeg_car)')
    parser.add_argument('--noise', type=int, default=15,
                        help='Noise level for denoising tasks: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40,
                        help='JPEG quality for jpeg_car tasks: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128,
                        help='Patch size used in training (see main_test_swinir.py)')
    parser.add_argument('--large_model', action='store_true',
                        help='Use large model (real_sr only)')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth')
    parser.add_argument('--folder_lq', type=str, default=None,
                        help='Input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='Input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size for patch-wise inference (None = whole image).\n'
                             'Recommended for quantized CPU inference on large images.')
    parser.add_argument('--tile_overlap', type=int, default=32)

    # ── Quantization args ─────────────────────────────────────────────────────
    parser.add_argument('--bits', type=int, default=8, choices=[4, 8],
                        help='Weight quantization bit-width: 8 (INT8) or 4 (INT4).\n'
                             'INT4 is CUDA-optimized; CPU inference may be slower.')
    parser.add_argument('--quant_conv', action='store_true',
                        help='Also quantize Conv2d layers (default: Linear only)')

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument('--save_path', type=str, default=None,
                        help='Override save path for the quantized model.\n'
                             'Default: model_zoo/swinir_quantized_int{bits}/<model_name>.pth')
    parser.add_argument('--skip_fp32_eval', action='store_true',
                        help='Skip FP32 baseline evaluation (saves time)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load FP32 model ───────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  SwinIR PTQ  |  task={args.task}  |  INT{args.bits} weight-only')
    print(f'{"="*60}')
    print(f'Loading FP32 model from {args.model_path}')
    model_fp32 = define_model(args)
    model_fp32.eval()

    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    img_save_dir = os.path.join(save_dir, 'quantize')

    n_linear, linear_names = count_linear_layers(model_fp32)
    print(f'FP32 model: {n_linear} nn.Linear layers to be quantized')
    print(f'FP32 size : {model_size_mb(model_fp32):.1f} MB')
    sample = [n for n in linear_names if 'attn' in n or 'mlp' in n][:6]
    if sample:
        print('  e.g. ' + '\n       '.join(sample))

    # ── FP32 baseline evaluation ──────────────────────────────────────────────
    metrics_fp32 = {}
    if not args.skip_fp32_eval and folder and os.path.exists(folder):
        print(f'\n--- FP32 Baseline on {folder} ---')
        model_fp32.to(device)
        metrics_fp32 = evaluate_folder(
            model_fp32, folder, args, window_size, device, tag='FP32',
        )
        model_fp32.cpu()
    else:
        print('\n[INFO] Skipping FP32 baseline evaluation.')

    # ── Apply PTQ ─────────────────────────────────────────────────────────────
    print(f'\n--- Applying INT{args.bits} PTQ ---')
    model_quant = apply_dynamic_ptq(model_fp32, bits=args.bits, quant_conv=args.quant_conv)

    # ── Size comparison ───────────────────────────────────────────────────────
    fp32_size = model_size_mb(model_fp32)
    quant_size = model_size_mb(model_quant)
    print(f'\nModel size:  FP32 {fp32_size:.1f} MB  →  INT{args.bits} {quant_size:.1f} MB'
          f'  ({fp32_size / quant_size:.2f}x reduction)')
    # Machine-parseable size lines
    print(f'[FP32] Memory: {fp32_size:.2f} MB')
    print(f'[INT{args.bits}] Memory: {quant_size:.2f} MB')

    # ── Quantized evaluation ──────────────────────────────────────────────────
    metrics_quant = {}
    if folder and os.path.exists(folder):
        print(f'\n--- INT{args.bits} evaluation on {folder} ({device}) ---')
        model_quant.to(device)
        metrics_quant = evaluate_folder(
            model_quant, folder, args, window_size, device, tag=f'INT{args.bits}',
            save_dir=img_save_dir,
        )

        if metrics_fp32.get('psnr') is not None and metrics_quant.get('psnr') is not None:
            print(f'\n--- Quality delta (FP32 → INT{args.bits}) ---')
            for key in ['psnr', 'ssim', 'psnr_y', 'ssim_y']:
                v_fp32 = metrics_fp32.get(key)
                v_q    = metrics_quant.get(key)
                if v_fp32 is not None and v_q is not None:
                    unit = ' dB' if 'psnr' in key else '    '
                    print(f'  {key.upper():7s}: {v_fp32:.4f} → {v_q:.4f}'
                          f'  (Δ = {v_q - v_fp32:+.4f}{unit})')

        t_fp32 = metrics_fp32.get('avg_time_s')
        t_q    = metrics_quant.get('avg_time_s')
        if t_fp32 is not None and t_q is not None and t_fp32 > 0:
            speedup = t_fp32 / t_q
            print(f'  RUNTIME: {t_fp32:.3f} s/img → {t_q:.3f} s/img'
                  f'  ({speedup:.2f}x {"faster" if speedup >= 1 else "slower"})')

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.model_path))[0]
        conv_tag = '_conv' if args.quant_conv else ''
        quant_dir = os.path.join('model_zoo', f'swinir_quantized_int{args.bits}')
        args.save_path = os.path.join(quant_dir, f'{base}{conv_tag}.pth')

    save_quantized_model(model_quant, args.save_path)
    print('\nDone.')


if __name__ == '__main__':
    main()
