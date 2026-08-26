#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${1:-"$ROOT/output"}

CUDA_VISIBLE_DEVICES=0 python "$ROOT/other_tools/compute_batch_metrics_offline.py" \
  "$DATA_ROOT" \
    --AdaFace 1 \
    --SFace 1 \
    --LPIPS 1 \
    --FG-CLIP_Score 1 \
    --MS-SSIM 1 \
    --FID 1 \
    --IS 1 \
    --SR_SIM 1 \
    --FSIM 1 \
    --VIF 1 \
    --early_stopped_both_matches_mean 1 \
    --only_one_shot \
    --write-csv
