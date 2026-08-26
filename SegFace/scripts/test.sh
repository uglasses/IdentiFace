NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python /data/knaraya4/SegFace/test.py \
    --ckpt_path ckpts \
    --expt_name <expt_name> \
    --dataset <dataset_name> \
    --backbone <backbone_name> \
    --model <model_name> \
    --input_resolution 512 \
    --test_bs 1 \
    --model_path [LOG_PATH]/<ckpt_path>/<expt_name>/model_299.pt


# --dataset celebamask_hq
# --dataset lapa
# --dataset helen

# --backbone segface_celeb
# --backbone segface_lapa
# --backbone segface_helen

# --model swin_base, swinv2_base, swinv2_small, swinv2_tiny
# --model convnext_base, convnext_small, convnext_tiny
# --model mobilenet
# --model efficientnet