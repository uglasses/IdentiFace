#!/usr/bin/env bash
# Preprocess ID-FFHQ (VAE / T5 / edges / LQ features)
set -euo pipefail
cd "$(dirname "$0")"

DATASET_ROOT="${DATASET_ROOT:-dataset/ID-FFHQ}"
MODELS_DIR="${MODELS_DIR:-models}"
RESOLUTION="${RESOLUTION:-1024}"
INPUT_METADATA="${INPUT_METADATA:-${DATASET_ROOT}/data_info_FFHQ.json}"
IMAGE_DIR="${IMAGE_DIR:-${DATASET_ROOT}}"
GPU_ID="${1:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

python preprocess_dataset.py \
    --dataset_root "$DATASET_ROOT" \
    --models_dir "$MODELS_DIR" \
    --resolution "$RESOLUTION" \
    --input_metadata "$INPUT_METADATA" \
    --image_dir "$IMAGE_DIR" \
    --use_long_prompt \
    --use_hard_lq \
    --hard_lq_camera_target_size 32
