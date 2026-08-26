NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29440 /data/knaraya4/SegFace/train.py \
    --ckpt_path ckpts \
    --expt_name swin_base_helen_512 \
    --dataset helen \
    --backbone segface_helen \
    --model swin_base \
    --lr 1e-4 \
    --lr_schedule 80,200 \
    --input_resolution 512 \
    --train_bs 2 \
    --val_bs 1 \
    --test_bs 1 \
    --num_workers 4 \
    --epochs 300

# --model swin_base, swinv2_base, swinv2_small, swinv2_tiny
# --model convnext_base, convnext_small, convnext_tiny
# --model mobilenet
# --model efficientnet