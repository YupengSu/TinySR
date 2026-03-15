#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

# =============================================================================
# SwinIR Test Script
# Each task outputs to its own log file under the logs/ directory
# =============================================================================

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

PYTHON="python main_test_swinir.py"
MODEL_DIR="model_zoo/swinir"
TESTSETS="testsets"

# =============================================================================
# 001 Classical Image Super-Resolution (Middle Size, SwinIR-M)
# =============================================================================

# Setting 1: Trained on DIV2K, training_patch_size=48
{
    echo "===== Classical SR (DIV2K, patch=48) ====="
    for SCALE in 2 3 4 8; do
        echo "[$(date)] Scale: x${SCALE}"
        ${PYTHON} --task classical_sr --scale ${SCALE} --training_patch_size 48 \
            --model_path ${MODEL_DIR}/001_classicalSR_DIV2K_s48w8_SwinIR-M_x${SCALE}.pth \
            --folder_lq ${TESTSETS}/Set5/LR_bicubic/X${SCALE} \
            --folder_gt ${TESTSETS}/Set5/HR
    done
} 2>&1 | tee "${LOG_DIR}/001_classicalSR_DIV2K_s48.log"

# Setting 2: Trained on DIV2K+Flickr2K, training_patch_size=64
{
    echo "===== Classical SR (DF2K, patch=64) ====="
    for SCALE in 2 3 4 8; do
        echo "[$(date)] Scale: x${SCALE}"
        ${PYTHON} --task classical_sr --scale ${SCALE} --training_patch_size 64 \
            --model_path ${MODEL_DIR}/001_classicalSR_DF2K_s64w8_SwinIR-M_x${SCALE}.pth \
            --folder_lq ${TESTSETS}/Set5/LR_bicubic/X${SCALE} \
            --folder_gt ${TESTSETS}/Set5/HR
    done
} 2>&1 | tee "${LOG_DIR}/001_classicalSR_DF2K_s64.log"

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
            --folder_gt ${TESTSETS}/Set5/HR
    done
} 2>&1 | tee "${LOG_DIR}/002_lightweightSR.log"

# =============================================================================
# 003 Real-World Image Super-Resolution
# =============================================================================

# Middle size (SwinIR-M)
{
    echo "===== Real-World SR (SwinIR-M) ====="
    echo "[$(date)] Scale: x4"
    ${PYTHON} --task real_sr --scale 4 \
        --model_path ${MODEL_DIR}/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
        --folder_lq ${TESTSETS}/RealSRSet+5images
} 2>&1 | tee "${LOG_DIR}/003_realSR_medium.log"

# Larger size (SwinIR-L)
{
    echo "===== Real-World SR (SwinIR-L) ====="
    echo "[$(date)] Scale: x4 (large model)"
    ${PYTHON} --task real_sr --scale 4 --large_model \
        --model_path ${MODEL_DIR}/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
        --folder_lq ${TESTSETS}/RealSRSet+5images
} 2>&1 | tee "${LOG_DIR}/003_realSR_large.log"

# =============================================================================
# 004 Grayscale Image Denoising (Middle Size, SwinIR-M)
# =============================================================================
{
    echo "===== Grayscale Denoising ====="
    for NOISE in 15 25 50; do
        echo "[$(date)] Noise level: ${NOISE}"
        ${PYTHON} --task gray_dn --noise ${NOISE} \
            --model_path ${MODEL_DIR}/004_grayDN_DFWB_s128w8_SwinIR-M_noise${NOISE}.pth \
            --folder_gt ${TESTSETS}/Set12
    done
} 2>&1 | tee "${LOG_DIR}/004_grayDN.log"

# =============================================================================
# 005 Color Image Denoising (Middle Size, SwinIR-M)
# =============================================================================
{
    echo "===== Color Denoising ====="
    for NOISE in 15 25 50; do
        echo "[$(date)] Noise level: ${NOISE}"
        ${PYTHON} --task color_dn --noise ${NOISE} \
            --model_path ${MODEL_DIR}/005_colorDN_DFWB_s128w8_SwinIR-M_noise${NOISE}.pth \
            --folder_gt ${TESTSETS}/McMaster
    done
} 2>&1 | tee "${LOG_DIR}/005_colorDN.log"

# =============================================================================
# 006 JPEG Compression Artifact Reduction (Middle Size, window_size=7)
# =============================================================================

# Grayscale JPEG CAR
{
    echo "===== Grayscale JPEG CAR ====="
    for JPEG in 10 20 30 40; do
        echo "[$(date)] JPEG quality: ${JPEG}"
        ${PYTHON} --task jpeg_car --jpeg ${JPEG} \
            --model_path ${MODEL_DIR}/006_CAR_DFWB_s126w7_SwinIR-M_jpeg${JPEG}.pth \
            --folder_gt ${TESTSETS}/classic5
    done
} 2>&1 | tee "${LOG_DIR}/006_jpegCAR_gray.log"

# Color JPEG CAR
{
    echo "===== Color JPEG CAR ====="
    for JPEG in 10 20 30 40; do
        echo "[$(date)] JPEG quality: ${JPEG}"
        ${PYTHON} --task color_jpeg_car --jpeg ${JPEG} \
            --model_path ${MODEL_DIR}/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg${JPEG}.pth \
            --folder_gt ${TESTSETS}/LIVE1
    done
} 2>&1 | tee "${LOG_DIR}/006_jpegCAR_color.log"

echo ""
echo "===== All tests completed! ====="
echo "Logs saved to ${LOG_DIR}/"
ls -lh ${LOG_DIR}/