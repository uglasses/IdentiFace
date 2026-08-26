#!/usr/bin/env bash
# ControlNet training launcher for ID-FFHQ (single or multi-GPU)
set -euo pipefail
cd "$(dirname "$0")"

CONFIG_FILE="${CONFIG_FILE:-configs/train_controlnet_ffhq_1024.py}"
WORK_DIR="${WORK_DIR:-work_dirs/controlnet_ffhq}"
DATA_ROOT="${DATA_ROOT:-dataset/ID-FFHQ}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-12345}"
LONG_PROMPT_RATIO="${LONG_PROMPT_RATIO:-0.3}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$DATA_ROOT/InternData" ]; then
    echo "Error: Data directory not found: $DATA_ROOT/InternData"
    echo "Please run ./preprocessing_ffhq.sh first."
    exit 1
fi

if [ ! -f "models/PixArt-XL-2-1024-MS.pth" ]; then
    echo "Warning: Pre-trained model not found: models/PixArt-XL-2-1024-MS.pth"
    echo "Download from:"
    echo "  https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth"
    echo "Continuing anyway; training will fail if load_from is missing."
fi

mkdir -p "$WORK_DIR"

echo "=========================================="
echo "Starting ControlNet Training (FFHQ)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Work dir: $WORK_DIR"
echo "Data root: $DATA_ROOT"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single GPU training..."
    python PixArt-alpha/train_scripts/train_controlnet.py \
        "$CONFIG_FILE" \
        --work-dir "$WORK_DIR" \
        --data_root "$DATA_ROOT" \
        --long-prompt-ratio "$LONG_PROMPT_RATIO"
else
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    PORT=$MASTER_PORT
    MAX_ATTEMPTS=10
    ATTEMPT=0
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if ! python3 -c "import socket; s=socket.socket(); s.bind(('', $PORT)); s.close()" 2>/dev/null; then
            echo "Port $PORT is in use, trying next port..."
            PORT=$((PORT + 1))
            ATTEMPT=$((ATTEMPT + 1))
        else
            break
        fi
    done
    echo "Using master port: $PORT"
    python -m torch.distributed.launch \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$PORT" \
        PixArt-alpha/train_scripts/train_controlnet.py \
        "$CONFIG_FILE" \
        --work-dir "$WORK_DIR" \
        --data_root "$DATA_ROOT" \
        --long-prompt-ratio "$LONG_PROMPT_RATIO"
fi
