#!/usr/bin/env bash
# Batch inference. Override paths via env vars if needed.
# Usage: ./run_inference.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")"

GPU_ID="${1:-0}"
DATA_ROOT="${DATA_ROOT:-dataset/ID-CelebA}"
MODEL_DIR="${MODEL_DIR:-models/controlnet_checkpoint}"
CKPT="${CKPT:-${MODEL_DIR}/checkpoints/latest.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-output/batch_test$(date +%Y%m%d%H%M%S)/result.png}"
INFER_RANGE_START="${INFER_RANGE_START:-0}"
INFER_RANGE_END="${INFER_RANGE_END:-20}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

python inference_controlnet.py \
    --model_path "$CKPT" \
    --model_dir "$MODEL_DIR" \
    --metadata_json "$DATA_ROOT/data_info_celebA_test.json" \
    --dataset_base_path "$DATA_ROOT" \
    --identity_file "$DATA_ROOT/identity_celebA.txt" \
    --output_path "$OUTPUT_PATH" \
    --use_long_prompt True \
    --adaface_faiss_index_type flat \
    --adaface_model_path models/adaface_ir101_webface12m.ckpt \
    --adaface_top_k 10 \
    --sface_model_path models/SFace_backbone.pth \
    --sface_top_k 10 \
    --lq_degrade_low_res 8 \
    --use_hd_lq False \
    --hd_lq_strength 1 \
    --adaface_gallery_dir "$DATA_ROOT/CelebA-HQ-img" \
    --sface_gallery_dir "$DATA_ROOT/CelebA-HQ-img" \
    --infer_range_start "$INFER_RANGE_START" \
    --infer_range_end "$INFER_RANGE_END"
