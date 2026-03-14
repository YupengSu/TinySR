#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# =============================================================================
# SwinIR PTQ Quantization Test Script
# Mirrors run_tests.sh task-by-task.
# Evaluates FP32 baseline and quantized model in one pass, reports delta.
#
# Usage:
#   bash run_quantize_tests.sh              # INT8 (default, recommended)
#   BITS=8 TILE=256 bash run_quantize_tests.sh  # INT8 with tile inference
#
# NOTE: INT4 (BITS=4) is NOT compatible with SwinIR.
#   torchao Int4WeightOnlyConfig requires weight.shape[-1] % group_size == 0
#   for group_size in {32, 64, 128, 256}.  SwinIR uses embed_dim=180 which is
#   not divisible by any of these values, so all layers are silently skipped.
#   Use INT8 (--bits 8) for actual quantization of SwinIR models.
# =============================================================================

LOG_DIR="logs/quantize"
mkdir -p "${LOG_DIR}"

PYTHON="python main_quantize_swinir.py"
MODEL_DIR="model_zoo/swinir"
TESTSETS="testsets"

BITS="${BITS:-8}"

TILE_ARG=""
if [ -n "${TILE}" ]; then
    TILE_ARG="--tile ${TILE} --tile_overlap 32"
fi

echo "============================================================"
echo " SwinIR PTQ  |  INT${BITS}  |  $(date)"
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
            --bits ${BITS} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/001_classicalSR_DIV2K_s48_int${BITS}.log"

{
    echo "===== Classical SR (DF2K, patch=64) ====="
    for SCALE in 2 3 4 8; do
        echo "[$(date)] Scale: x${SCALE}"
        ${PYTHON} --task classical_sr --scale ${SCALE} --training_patch_size 64 \
            --model_path ${MODEL_DIR}/001_classicalSR_DF2K_s64w8_SwinIR-M_x${SCALE}.pth \
            --folder_lq ${TESTSETS}/Set5/LR_bicubic/X${SCALE} \
            --folder_gt ${TESTSETS}/Set5/HR \
            --bits ${BITS} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/001_classicalSR_DF2K_s64_int${BITS}.log"

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
            --bits ${BITS} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/002_lightweightSR_int${BITS}.log"

# =============================================================================
# 003 Real-World Image Super-Resolution (no GT, FP32 vs INT delta not shown)
# =============================================================================

{
    echo "===== Real-World SR (SwinIR-M) ====="
    ${PYTHON} --task real_sr --scale 4 \
        --model_path ${MODEL_DIR}/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
        --folder_lq ${TESTSETS}/RealSRSet+5images \
        --bits ${BITS} ${TILE_ARG}
} 2>&1 | tee "${LOG_DIR}/003_realSR_medium_int${BITS}.log"

{
    echo "===== Real-World SR (SwinIR-L) ====="
    ${PYTHON} --task real_sr --scale 4 --large_model \
        --model_path ${MODEL_DIR}/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
        --folder_lq ${TESTSETS}/RealSRSet+5images \
        --bits ${BITS} ${TILE_ARG}
} 2>&1 | tee "${LOG_DIR}/003_realSR_large_int${BITS}.log"

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
            --bits ${BITS} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/004_grayDN_int${BITS}.log"

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
            --bits ${BITS} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/005_colorDN_int${BITS}.log"

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
            --bits ${BITS} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/006_jpegCAR_gray_int${BITS}.log"

{
    echo "===== Color JPEG CAR ====="
    for JPEG in 10 20 30 40; do
        echo "[$(date)] JPEG quality: ${JPEG}"
        ${PYTHON} --task color_jpeg_car --jpeg ${JPEG} \
            --model_path ${MODEL_DIR}/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg${JPEG}.pth \
            --folder_gt ${TESTSETS}/LIVE1 \
            --bits ${BITS} ${TILE_ARG}
    done
} 2>&1 | tee "${LOG_DIR}/006_jpegCAR_color_int${BITS}.log"

# =============================================================================
# Summary: FP32 vs INT${BITS}  --  PSNR / Memory / Runtime
# =============================================================================
echo ""
echo "============================================================"
echo " Summary: FP32 vs INT${BITS}"
echo "============================================================"
printf "%-42s  %7s  %7s  %7s  %8s  %8s  %8s\n" \
    "Log" "FP32" "INT${BITS}" "ΔPSNR" "FP32_MB" "INT${BITS}_MB" "Speedup"
echo "--------------------------------------------------------------------"
for LOG in "${LOG_DIR}"/*_int${BITS}.log; do
    NAME=$(basename "${LOG}" _int${BITS}.log)

    FP32_PSNR=$(grep -oP '\[FP32\] Avg\s+PSNR:\s*\K[0-9.]+' "${LOG}" | tail -1)
    INT_PSNR=$(grep -oP "\[INT${BITS}\] Avg\s+PSNR:\s*\K[0-9.]+" "${LOG}" | tail -1)

    FP32_MB=$(grep -oP '\[FP32\] Memory:\s*\K[0-9.]+' "${LOG}" | tail -1)
    INT_MB=$(grep -oP "\[INT${BITS}\] Memory:\s*\K[0-9.]+" "${LOG}" | tail -1)

    FP32_T=$(grep -oP '\[FP32\] Avg\s+Runtime:\s*\K[0-9.]+' "${LOG}" | tail -1)
    INT_T=$(grep -oP "\[INT${BITS}\] Avg\s+Runtime:\s*\K[0-9.]+" "${LOG}" | tail -1)

    # ΔPSNR
    if [ -n "${FP32_PSNR}" ] && [ -n "${INT_PSNR}" ]; then
        DELTA=$(awk "BEGIN{printf \"%+.3f\", ${INT_PSNR}-${FP32_PSNR}}")
    else
        DELTA="N/A"
    fi

    # Speedup
    if [ -n "${FP32_T}" ] && [ -n "${INT_T}" ] && awk "BEGIN{exit!(${INT_T}>0)}"; then
        SPEEDUP=$(awk "BEGIN{printf \"%.2fx\", ${FP32_T}/${INT_T}}")
    else
        SPEEDUP="N/A"
    fi

    printf "%-42s  %7s  %7s  %7s  %8s  %8s  %8s\n" \
        "${NAME}" \
        "${FP32_PSNR:--}" "${INT_PSNR:--}" "${DELTA}" \
        "${FP32_MB:--}" "${INT_MB:--}" "${SPEEDUP}"
done

echo ""
echo "===== All INT${BITS} quantize tests completed! ====="
echo "Logs  -> ${LOG_DIR}/"
echo "Models -> model_zoo/swinir_quantized_int${BITS}/"
ls -lh "${LOG_DIR}/"*_int${BITS}.log 2>/dev/null
