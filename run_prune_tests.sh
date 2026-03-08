#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# =============================================================================
# SwinIR Structured Pruning Test Script
# Mirrors run_quantize_tests.sh task-by-task.
# Evaluates FP32 baseline and pruned model in one pass, reports delta.
#
# Usage:
#   bash run_prune_tests.sh                              # MLP 75% (default)
#   MLP=0.5 bash run_prune_tests.sh                     # MLP 50%
#   MLP=0.75 ATTN=4 bash run_prune_tests.sh             # MLP 75% + 4 heads
#   MLP=0.75 TILE=256 bash run_prune_tests.sh           # with tile inference
#
# Pruning targets:
#   MLP  : hidden channels of Mlp.fc1/fc2  (SwinIR-M: 360 → keep MLP*360)
#   Attn : attention heads (SwinIR-M: 6 heads → keep ATTN heads, -1 = skip)
# =============================================================================

LOG_DIR="logs/prune"
mkdir -p "${LOG_DIR}"

PYTHON="python main_prune_swinir.py"
MODEL_DIR="model_zoo/swinir"
TESTSETS="testsets"

MLP="${MLP:-0.75}"
ATTN="${ATTN:--1}"

ATTN_ARG=""
if [ "${ATTN}" != "-1" ]; then
    ATTN_ARG="--attn_keep_heads ${ATTN}"
fi

TILE_ARG=""
if [ -n "${TILE}" ]; then
    TILE_ARG="--tile ${TILE} --tile_overlap 32"
fi

# Prune tag for log file naming  (e.g. mlp75  or  mlp75_attn4)
MLP_PCT=$(awk "BEGIN{printf \"%d\", ${MLP}*100}")
PRUNE_TAG="mlp${MLP_PCT}"
if [ "${ATTN}" != "-1" ]; then
    PRUNE_TAG="${PRUNE_TAG}_attn${ATTN}"
fi

echo "============================================================"
echo " SwinIR Structured Pruning  |  MLP=${MLP}  Attn=${ATTN}  |  $(date)"
echo " Logs -> ${LOG_DIR}/"
echo "============================================================"

# =============================================================================
# 001 Classical Image Super-Resolution (Middle Size, SwinIR-M)
# =============================================================================

{
    echo "===== Classical SR (DIV2K, patch=48) ====="
    for SCALE in 2 3 4 8; do
        echo "[$(date)] Scale: x${SCALE}"
        ${PYTHON} --task classical_sr --scale ${SCALE} --training_patch_size 48 \
            --model_path ${MODEL_DIR}/001_classicalSR_DIV2K_s48w8_SwinIR-M_x${SCALE}.pth \
            --folder_lq ${TESTSETS}/Set5/LR_bicubic/X${SCALE} \
            --folder_gt ${TESTSETS}/Set5/HR \
            --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/001_classicalSR_DIV2K_s48_${PRUNE_TAG}.log"

{
    echo "===== Classical SR (DF2K, patch=64) ====="
    for SCALE in 2 3 4 8; do
        echo "[$(date)] Scale: x${SCALE}"
        ${PYTHON} --task classical_sr --scale ${SCALE} --training_patch_size 64 \
            --model_path ${MODEL_DIR}/001_classicalSR_DF2K_s64w8_SwinIR-M_x${SCALE}.pth \
            --folder_lq ${TESTSETS}/Set5/LR_bicubic/X${SCALE} \
            --folder_gt ${TESTSETS}/Set5/HR \
            --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/001_classicalSR_DF2K_s64_${PRUNE_TAG}.log"

# =============================================================================
# 002 Lightweight Image Super-Resolution (Small Size, SwinIR-S)
# =============================================================================
{
    echo "===== Lightweight SR (DIV2K) ====="
    for SCALE in 2 3 4; do
        echo "[$(date)] Scale: x${SCALE}"
        ${PYTHON} --task lightweight_sr --scale ${SCALE} \
            --model_path ${MODEL_DIR}/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x${SCALE}.pth \
            --folder_lq ${TESTSETS}/Set5/LR_bicubic/X${SCALE} \
            --folder_gt ${TESTSETS}/Set5/HR \
            --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/002_lightweightSR_${PRUNE_TAG}.log"

# =============================================================================
# 003 Real-World Image Super-Resolution (no GT, quality delta not shown)
# =============================================================================

{
    echo "===== Real-World SR (SwinIR-M) ====="
    ${PYTHON} --task real_sr --scale 4 \
        --model_path ${MODEL_DIR}/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
        --folder_lq ${TESTSETS}/RealSRSet+5images \
        --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
} 2>&1 | tee "${LOG_DIR}/003_realSR_medium_${PRUNE_TAG}.log"

{
    echo "===== Real-World SR (SwinIR-L) ====="
    ${PYTHON} --task real_sr --scale 4 --large_model \
        --model_path ${MODEL_DIR}/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
        --folder_lq ${TESTSETS}/RealSRSet+5images \
        --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
} 2>&1 | tee "${LOG_DIR}/003_realSR_large_${PRUNE_TAG}.log"

# =============================================================================
# 004 Grayscale Image Denoising (Middle Size, SwinIR-M)
# =============================================================================
{
    echo "===== Grayscale Denoising ====="
    for NOISE in 15 25 50; do
        echo "[$(date)] Noise level: ${NOISE}"
        ${PYTHON} --task gray_dn --noise ${NOISE} \
            --model_path ${MODEL_DIR}/004_grayDN_DFWB_s128w8_SwinIR-M_noise${NOISE}.pth \
            --folder_gt ${TESTSETS}/Set12 \
            --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/004_grayDN_${PRUNE_TAG}.log"

# =============================================================================
# 005 Color Image Denoising (Middle Size, SwinIR-M)
# =============================================================================
{
    echo "===== Color Denoising ====="
    for NOISE in 15 25 50; do
        echo "[$(date)] Noise level: ${NOISE}"
        ${PYTHON} --task color_dn --noise ${NOISE} \
            --model_path ${MODEL_DIR}/005_colorDN_DFWB_s128w8_SwinIR-M_noise${NOISE}.pth \
            --folder_gt ${TESTSETS}/McMaster \
            --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/005_colorDN_${PRUNE_TAG}.log"

# =============================================================================
# 006 JPEG Compression Artifact Reduction (Middle Size, window_size=7)
# =============================================================================

{
    echo "===== Grayscale JPEG CAR ====="
    for JPEG in 10 20 30 40; do
        echo "[$(date)] JPEG quality: ${JPEG}"
        ${PYTHON} --task jpeg_car --jpeg ${JPEG} \
            --model_path ${MODEL_DIR}/006_CAR_DFWB_s126w7_SwinIR-M_jpeg${JPEG}.pth \
            --folder_gt ${TESTSETS}/classic5 \
            --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/006_jpegCAR_gray_${PRUNE_TAG}.log"

{
    echo "===== Color JPEG CAR ====="
    for JPEG in 10 20 30 40; do
        echo "[$(date)] JPEG quality: ${JPEG}"
        ${PYTHON} --task color_jpeg_car --jpeg ${JPEG} \
            --model_path ${MODEL_DIR}/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg${JPEG}.pth \
            --folder_gt ${TESTSETS}/LIVE1 \
            --mlp_keep_ratio ${MLP} ${ATTN_ARG} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/006_jpegCAR_color_${PRUNE_TAG}.log"

# =============================================================================
# Summary: FP32 vs Pruned  --  PSNR / Params / Memory / Runtime
# =============================================================================
echo ""
echo "============================================================"
echo " Summary: FP32 vs Pruned (${PRUNE_TAG})"
echo "============================================================"
printf "%-42s  %7s  %7s  %7s  %10s  %10s  %8s  %8s\n" \
    "Log" "FP32" "Pruned" "ΔPSNR" "FP32_param" "Prnd_param" "Ratio" "Speedup"
echo "--------------------------------------------------------------------------------------"

for LOG in "${LOG_DIR}"/*_${PRUNE_TAG}.log; do
    NAME=$(basename "${LOG}" _${PRUNE_TAG}.log)

    FP32_PSNR=$(grep -oP '\[FP32\] Avg\s+PSNR:\s*\K[0-9.]+' "${LOG}" | tail -1)
    PRN_PSNR=$(grep -oP '\[Pruned\] Avg\s+PSNR:\s*\K[0-9.]+' "${LOG}" | tail -1)

    FP32_MB=$(grep -oP '\[FP32\] Memory:\s*\K[0-9.]+' "${LOG}" | tail -1)
    PRN_MB=$(grep -oP '\[Pruned\] Memory:\s*\K[0-9.]+' "${LOG}" | tail -1)

    FP32_PAR=$(grep -oP '\[FP32\]\s+Memory:[^\n]+Params:\s*\K[0-9]+' "${LOG}" | tail -1)
    PRN_PAR=$(grep -oP '\[Pruned\]\s+Memory:[^\n]+Params:\s*\K[0-9]+' "${LOG}" | tail -1)

    FP32_T=$(grep -oP '\[FP32\] Avg\s+Runtime:\s*\K[0-9.]+' "${LOG}" | tail -1)
    PRN_T=$(grep -oP '\[Pruned\] Avg\s+Runtime:\s*\K[0-9.]+' "${LOG}" | tail -1)

    # ΔPSNR
    if [ -n "${FP32_PSNR}" ] && [ -n "${PRN_PSNR}" ]; then
        DELTA=$(awk "BEGIN{printf \"%+.3f\", ${PRN_PSNR}-${FP32_PSNR}}")
    else
        DELTA="N/A"
    fi

    # Param compression ratio
    if [ -n "${FP32_PAR}" ] && [ -n "${PRN_PAR}" ] && awk "BEGIN{exit!(${PRN_PAR}>0)}"; then
        PARAM_RATIO=$(awk "BEGIN{printf \"%.2fx\", ${FP32_PAR}/${PRN_PAR}}")
    else
        PARAM_RATIO="N/A"
    fi

    # Speedup
    if [ -n "${FP32_T}" ] && [ -n "${PRN_T}" ] && awk "BEGIN{exit!(${PRN_T}>0)}"; then
        SPEEDUP=$(awk "BEGIN{printf \"%.2fx\", ${FP32_T}/${PRN_T}}")
    else
        SPEEDUP="N/A"
    fi

    # Format large param counts as "M"
    FP32_PAR_FMT=$([ -n "${FP32_PAR}" ] && awk "BEGIN{printf \"%.2fM\", ${FP32_PAR}/1e6}" || echo "-")
    PRN_PAR_FMT=$([ -n "${PRN_PAR}"  ] && awk "BEGIN{printf \"%.2fM\", ${PRN_PAR}/1e6}"  || echo "-")

    printf "%-42s  %7s  %7s  %7s  %10s  %10s  %8s  %8s\n" \
        "${NAME}" \
        "${FP32_PSNR:--}" "${PRN_PSNR:--}" "${DELTA}" \
        "${FP32_PAR_FMT}" "${PRN_PAR_FMT}" "${PARAM_RATIO}" "${SPEEDUP}"
done

echo ""
echo "===== All pruning tests completed! (${PRUNE_TAG}) ====="
echo "Logs   -> ${LOG_DIR}/"
echo "Models -> model_zoo/swinir_pruned/"
ls -lh "${LOG_DIR}"/*_${PRUNE_TAG}.log 2>/dev/null
