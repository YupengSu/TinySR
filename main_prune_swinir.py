#!/usr/bin/env python3
"""
main_prune_swinir.py -- Structured Pruning for SwinIR (no training required)

Pruning targets (one-shot magnitude pruning by L1 importance):
  MLP  (default) : hidden channels of Mlp.fc1 / Mlp.fc2
  Attn (optional): attention heads of WindowAttention.qkv / proj

SwinIR-M stats: embed_dim=180, mlp_ratio=2 → hidden_dim=360, 6 heads @ head_dim=30

Usage examples
  # MLP channel pruning, keep 75% of hidden neurons (default)
  python main_prune_swinir.py \
      --task color_dn --noise 15 \
      --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
      --folder_gt testsets/McMaster

  # Aggressively prune MLP to 50%
  python main_prune_swinir.py --mlp_keep_ratio 0.5 \
      --task color_dn --noise 15 \
      --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
      --folder_gt testsets/McMaster

  # Prune MLP + attention heads (keep 4 of 6 heads)
  python main_prune_swinir.py --mlp_keep_ratio 0.75 --attn_keep_heads 4 \
      --task color_dn --noise 15 \
      --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
      --folder_gt testsets/McMaster
"""

import argparse
import copy
import io
import types
import glob
import numpy as np
import os
import time
import torch
import torch.nn as nn

from models.network_swinir import SwinIR as net, Mlp, WindowAttention
from utils import util_calculate_psnr_ssim as util
from main_test_swinir import define_model, setup, get_image_pair, test


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 / 1024


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# MLP channel pruning
# ─────────────────────────────────────────────────────────────────────────────

def prune_mlp_block(mlp: Mlp, keep_ratio: float) -> int:
    """
    Prune hidden channels of a single Mlp block in-place.

    Scoring: L1 norm of fc1 output rows (one row = one hidden neuron).
    Rebuild fc1 [in → keep_n] and fc2 [keep_n → out] with selected channels.

    Returns the number of channels kept.
    """
    fc1, fc2 = mlp.fc1, mlp.fc2
    hidden_dim = fc1.out_features
    keep_n = max(1, round(hidden_dim * keep_ratio))

    importance = fc1.weight.data.abs().sum(dim=1)          # [hidden_dim]
    keep_idx   = importance.topk(keep_n).indices.sort().values  # sorted ascending

    # New fc1: [in_features → keep_n]
    new_fc1 = nn.Linear(fc1.in_features, keep_n, bias=fc1.bias is not None,
                         device=fc1.weight.device, dtype=fc1.weight.dtype)
    new_fc1.weight.data = fc1.weight.data[keep_idx]
    if fc1.bias is not None:
        new_fc1.bias.data = fc1.bias.data[keep_idx]

    # New fc2: [keep_n → out_features]
    new_fc2 = nn.Linear(keep_n, fc2.out_features, bias=fc2.bias is not None,
                         device=fc2.weight.device, dtype=fc2.weight.dtype)
    new_fc2.weight.data = fc2.weight.data[:, keep_idx]
    if fc2.bias is not None:
        new_fc2.bias.data = fc2.bias.data.clone()

    mlp.fc1 = new_fc1
    mlp.fc2 = new_fc2
    return keep_n


# ─────────────────────────────────────────────────────────────────────────────
# Attention head pruning
# ─────────────────────────────────────────────────────────────────────────────

def _make_pruned_attn_forward(keep_n_heads: int, head_dim: int):
    """
    Return a patched forward function for WindowAttention that uses the pruned
    num_heads and the original (fixed) head_dim, bypassing the original
    `C // self.num_heads` reshape which would break after head removal.
    """
    def forward(self, x, mask=None):
        B_, N, _ = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, keep_n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q   = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)
        attn = attn + rpb.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, keep_n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, keep_n_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, keep_n_heads * head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return forward


def prune_attn_block(attn: WindowAttention, keep_n_heads: int) -> int:
    """
    Prune attention heads of a single WindowAttention block in-place.

    Scoring: sum of L1 norms of each head's Q/K/V weight slices + proj columns.
    Patches the forward() method so the reshape uses the fixed original head_dim.

    Returns the number of heads kept.
    """
    dim       = attn.dim
    num_heads = attn.num_heads
    head_dim  = dim // num_heads
    keep_n_heads = min(keep_n_heads, num_heads)

    if keep_n_heads == num_heads:
        return num_heads

    qkv_w  = attn.qkv.weight.data   # [3*dim, dim]
    proj_w = attn.proj.weight.data  # [dim, dim]

    # Score each head
    scores = []
    for i in range(num_heads):
        qs = qkv_w[i*head_dim           : (i+1)*head_dim,            :]
        ks = qkv_w[dim + i*head_dim     : dim + (i+1)*head_dim,      :]
        vs = qkv_w[2*dim + i*head_dim   : 2*dim + (i+1)*head_dim,    :]
        ps = proj_w[:, i*head_dim       : (i+1)*head_dim              ]
        scores.append((qs.abs().sum() + ks.abs().sum()
                       + vs.abs().sum() + ps.abs().sum()).item())

    keep_idx = torch.tensor(scores).topk(keep_n_heads).indices.sort().values.tolist()

    new_dim = keep_n_heads * head_dim

    # Rebuild qkv weight: stack selected head rows for Q, K, V
    def _gather_rows(mat, offset):
        return torch.cat([mat[offset + i*head_dim : offset + (i+1)*head_dim] for i in keep_idx])

    new_qkv_w = torch.cat([_gather_rows(qkv_w, 0),
                            _gather_rows(qkv_w, dim),
                            _gather_rows(qkv_w, 2*dim)], dim=0)

    new_qkv = nn.Linear(dim, 3 * new_dim, bias=attn.qkv.bias is not None,
                         device=qkv_w.device, dtype=qkv_w.dtype)
    new_qkv.weight.data = new_qkv_w
    if attn.qkv.bias is not None:
        b = attn.qkv.bias.data
        new_qkv.bias.data = torch.cat([
            torch.cat([b[i*head_dim       : (i+1)*head_dim      ] for i in keep_idx]),
            torch.cat([b[dim + i*head_dim : dim+(i+1)*head_dim  ] for i in keep_idx]),
            torch.cat([b[2*dim+i*head_dim : 2*dim+(i+1)*head_dim] for i in keep_idx]),
        ])

    # Rebuild proj: select columns corresponding to kept V heads
    new_proj_w = torch.cat([proj_w[:, i*head_dim : (i+1)*head_dim] for i in keep_idx], dim=1)
    new_proj = nn.Linear(new_dim, dim, bias=attn.proj.bias is not None,
                          device=proj_w.device, dtype=proj_w.dtype)
    new_proj.weight.data = new_proj_w
    if attn.proj.bias is not None:
        new_proj.bias.data = attn.proj.bias.data.clone()

    # Prune relative_position_bias_table columns
    new_rpbt = attn.relative_position_bias_table.data[:, keep_idx]
    attn.relative_position_bias_table = nn.Parameter(new_rpbt)

    attn.qkv       = new_qkv
    attn.proj      = new_proj
    attn.num_heads = keep_n_heads
    attn.scale     = head_dim ** -0.5  # head_dim unchanged

    # Patch forward to use fixed head_dim (avoids C // num_heads mismatch)
    attn.forward = types.MethodType(_make_pruned_attn_forward(keep_n_heads, head_dim), attn)

    return keep_n_heads


# ─────────────────────────────────────────────────────────────────────────────
# Apply pruning to full model
# ─────────────────────────────────────────────────────────────────────────────

def apply_structured_pruning(model: nn.Module,
                              mlp_keep_ratio: float = 1.0,
                              attn_keep_heads: int = -1) -> nn.Module:
    """
    Walk all SwinTransformerBlock modules and prune their Mlp and/or attention heads.

    mlp_keep_ratio : fraction of hidden MLP channels to keep (0 < r ≤ 1). 1.0 = skip.
    attn_keep_heads: number of heads to keep. -1 = skip attn pruning.
    """
    from models.network_swinir import SwinTransformerBlock

    model = copy.deepcopy(model)
    n_mlp_pruned  = 0
    n_attn_pruned = 0

    for name, module in model.named_modules():
        if not isinstance(module, SwinTransformerBlock):
            continue

        if mlp_keep_ratio < 1.0:
            old_h = module.mlp.fc1.out_features
            new_h = prune_mlp_block(module.mlp, mlp_keep_ratio)
            n_mlp_pruned += 1
            if n_mlp_pruned == 1:
                print(f'  [MLP]  hidden {old_h} → {new_h}  (keep {mlp_keep_ratio:.0%})')

        if attn_keep_heads > 0:
            old_nh = module.attn.num_heads
            new_nh = prune_attn_block(module.attn, attn_keep_heads)
            n_attn_pruned += 1
            if n_attn_pruned == 1:
                print(f'  [Attn] heads  {old_nh} → {new_nh}')

    print(f'  Pruned {n_mlp_pruned} MLP blocks, {n_attn_pruned} Attn blocks')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (mirrors main_quantize_swinir.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_folder(model: nn.Module, folder: str, args,
                    window_size: int, device: torch.device, tag: str = '') -> dict:
    results = {'psnr': [], 'ssim': [], 'psnr_y': [], 'ssim_y': [], 'time_s': []}
    paths = sorted(glob.glob(os.path.join(folder, '*')))
    if not paths:
        print(f'[WARN] No images found in {folder}')
        return results

    for idx, path in enumerate(paths):
        imgname, img_lq, img_gt = get_image_pair(args, path)
        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
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

    print(f'[{tag}] Avg  Runtime: {avg_t:.3f} s/image')

    out = {k: (float(np.mean(v)) if v else None) for k, v in results.items() if k != 'time_s'}
    out['avg_time_s'] = avg_t
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_pruned_model(model: nn.Module, save_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({'params': model.state_dict()}, save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f'Saved pruned model → {save_path}  ({size_mb:.1f} MB on disk)')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Structured Channel Pruning for SwinIR',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Task / model args (mirrors main_test_swinir.py) ──────────────────────
    parser.add_argument('--task', type=str, default='color_dn',
                        help='classical_sr | lightweight_sr | real_sr | '
                             'gray_dn | color_dn | jpeg_car | color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--noise', type=int, default=15)
    parser.add_argument('--jpeg',  type=int, default=40)
    parser.add_argument('--training_patch_size', type=int, default=128)
    parser.add_argument('--large_model', action='store_true')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth')
    parser.add_argument('--folder_lq', type=str, default=None)
    parser.add_argument('--folder_gt', type=str, default=None)
    parser.add_argument('--tile', type=int, default=None)
    parser.add_argument('--tile_overlap', type=int, default=32)

    # ── Pruning args ──────────────────────────────────────────────────────────
    parser.add_argument('--mlp_keep_ratio', type=float, default=0.75,
                        help='Fraction of MLP hidden channels to keep (0 < r ≤ 1).\n'
                             '0.75 → 25%% of hidden neurons removed.  1.0 = no MLP pruning.')
    parser.add_argument('--attn_keep_heads', type=int, default=-1,
                        help='Number of attention heads to keep (e.g. 4 out of 6).\n'
                             '-1 = skip attention head pruning (default).')

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument('--save_path', type=str, default=None,
                        help='Override save path for the pruned model.\n'
                             'Default: model_zoo/swinir_pruned/<model_name>_pruned.pth')
    parser.add_argument('--skip_fp32_eval', action='store_true',
                        help='Skip FP32 baseline evaluation (saves time)')

    args = parser.parse_args()

    if args.mlp_keep_ratio <= 0 or args.mlp_keep_ratio > 1:
        raise ValueError('--mlp_keep_ratio must be in (0, 1]')

    do_mlp  = args.mlp_keep_ratio < 1.0
    do_attn = args.attn_keep_heads > 0

    if not do_mlp and not do_attn:
        print('[WARN] Nothing to prune: mlp_keep_ratio=1.0 and attn_keep_heads=-1.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load FP32 model ───────────────────────────────────────────────────────
    tag_parts = []
    if do_mlp:
        tag_parts.append(f'MLP={args.mlp_keep_ratio:.0%}')
    if do_attn:
        tag_parts.append(f'Attn={args.attn_keep_heads}heads')
    prune_tag = '+'.join(tag_parts)

    print(f'\n{"="*60}')
    print(f'  SwinIR Structured Pruning  |  task={args.task}  |  {prune_tag}')
    print(f'{"="*60}')
    print(f'Loading FP32 model from {args.model_path}')
    model_fp32 = define_model(args)
    model_fp32.eval()

    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    fp32_params = count_params(model_fp32)
    fp32_size   = model_size_mb(model_fp32)
    print(f'FP32 model: {fp32_params:,} params  |  {fp32_size:.1f} MB')

    # ── FP32 baseline ─────────────────────────────────────────────────────────
    metrics_fp32 = {}
    if not args.skip_fp32_eval and folder and os.path.exists(folder):
        print(f'\n--- FP32 Baseline on {folder} ---')
        model_fp32.to(device)
        metrics_fp32 = evaluate_folder(model_fp32, folder, args, window_size, device, tag='FP32')
        model_fp32.cpu()
    else:
        print('\n[INFO] Skipping FP32 baseline evaluation.')

    # ── Apply pruning ─────────────────────────────────────────────────────────
    print(f'\n--- Applying structured pruning ({prune_tag}) ---')
    model_pruned = apply_structured_pruning(
        model_fp32,
        mlp_keep_ratio=args.mlp_keep_ratio if do_mlp else 1.0,
        attn_keep_heads=args.attn_keep_heads,
    )
    model_pruned.eval()

    # ── Size / param comparison ───────────────────────────────────────────────
    pruned_params = count_params(model_pruned)
    pruned_size   = model_size_mb(model_pruned)
    param_ratio   = fp32_params / pruned_params
    size_ratio    = fp32_size   / pruned_size

    print(f'\nParams:  {fp32_params:>12,} → {pruned_params:>12,}  ({param_ratio:.2f}x reduction)')
    print(f'Size:    {fp32_size:>8.1f} MB  → {pruned_size:>8.1f} MB  ({size_ratio:.2f}x reduction)')
    # Machine-parseable lines
    print(f'[FP32]   Memory: {fp32_size:.2f} MB   Params: {fp32_params}')
    print(f'[Pruned] Memory: {pruned_size:.2f} MB   Params: {pruned_params}')

    # ── Pruned model evaluation ───────────────────────────────────────────────
    metrics_pruned = {}
    if folder and os.path.exists(folder):
        print(f'\n--- Pruned evaluation on {folder} ({device}) ---')
        model_pruned.to(device)
        metrics_pruned = evaluate_folder(
            model_pruned, folder, args, window_size, device, tag='Pruned')

        if metrics_fp32.get('psnr') is not None and metrics_pruned.get('psnr') is not None:
            print(f'\n--- Quality delta (FP32 → Pruned) ---')
            for key in ['psnr', 'ssim', 'psnr_y', 'ssim_y']:
                v_fp32   = metrics_fp32.get(key)
                v_pruned = metrics_pruned.get(key)
                if v_fp32 is not None and v_pruned is not None:
                    unit = ' dB' if 'psnr' in key else '    '
                    print(f'  {key.upper():7s}: {v_fp32:.4f} → {v_pruned:.4f}'
                          f'  (Δ = {v_pruned - v_fp32:+.4f}{unit})')

        t_fp32   = metrics_fp32.get('avg_time_s')
        t_pruned = metrics_pruned.get('avg_time_s')
        if t_fp32 and t_pruned and t_fp32 > 0:
            speedup = t_fp32 / t_pruned
            print(f'  RUNTIME: {t_fp32:.3f} s/img → {t_pruned:.3f} s/img'
                  f'  ({speedup:.2f}x {"faster" if speedup >= 1 else "slower"})')

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save_path is None:
        base      = os.path.splitext(os.path.basename(args.model_path))[0]
        mlp_tag   = f'_mlp{int(args.mlp_keep_ratio*100)}' if do_mlp  else ''
        attn_tag  = f'_attn{args.attn_keep_heads}h'       if do_attn else ''
        prune_dir = os.path.join('model_zoo', 'swinir_pruned')
        args.save_path = os.path.join(prune_dir, f'{base}{mlp_tag}{attn_tag}.pth')

    save_pruned_model(model_pruned, args.save_path)
    print('\nDone.')


if __name__ == '__main__':
    main()
