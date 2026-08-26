import os
import re
import torch

from diffusion.utils.logger import get_root_logger


def save_checkpoint(work_dir,
                    epoch,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    keep_last=False,
                    step=None,
                    overwrite=True,
                    ):
    """
    Save a checkpoint.

    Args:
        work_dir: Directory to save into
        epoch: Epoch number
        model: Model
        model_ema: EMA model (optional)
        optimizer: Optimizer (optional)
        lr_scheduler: Learning-rate scheduler (optional)
        keep_last: Whether to keep only the latest checkpoint (deprecated; use overwrite instead)
        step: Step number (optional)
        overwrite: Whether to overwrite old checkpoints (True: fixed name latest.pth; False: epoch_step naming)
    """
    os.makedirs(work_dir, exist_ok=True)
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict['state_dict_ema'] = model_ema.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['scheduler'] = lr_scheduler.state_dict()
    if epoch is not None:
        state_dict['epoch'] = epoch
    if step is not None:
        state_dict['step'] = step
    
    logger = get_root_logger()
    
    if overwrite:
        # Use a fixed filename and overwrite the old checkpoint
        file_path = os.path.join(work_dir, "latest.pth")
        torch.save(state_dict, file_path)
        logger.info(f'Saved checkpoint (epoch {epoch}' + (f', step {step}' if step is not None else '') + f') to {file_path} (overwrite mode).')
    else:
        # Name by epoch and step; keep historical checkpoints
        if epoch is not None:
            file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"
        else:
            file_path = os.path.join(work_dir, "checkpoint.pth")
    torch.save(state_dict, file_path)
    logger.info(f'Saved checkpoint of epoch {epoch}' + (f', step {step}' if step is not None else '') + f' to {file_path}.')
    if keep_last:
        for i in range(epoch):
            previous_ckgt = os.path.join(work_dir, f"epoch_{i}.pth")
            if os.path.exists(previous_ckgt):
                os.remove(previous_ckgt)


def load_checkpoint(checkpoint,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    load_ema=False,
                    resume_optimizer=True,
                    resume_lr_scheduler=True
                    ):
    assert isinstance(checkpoint, str)
    ckpt_file = checkpoint
    checkpoint = torch.load(ckpt_file, map_location="cpu")

    state_dict_keys = ['pos_embed', 'base_model.pos_embed', 'model.pos_embed']
    for key in state_dict_keys:
        if key in checkpoint['state_dict']:
            del checkpoint['state_dict'][key]
            if 'state_dict_ema' in checkpoint and key in checkpoint['state_dict_ema']:
                del checkpoint['state_dict_ema'][key]
            break

    if load_ema:
        state_dict = checkpoint['state_dict_ema']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)  # to be compatible with the official checkpoint
    # model.load_state_dict(state_dict)
    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    if model_ema is not None:
        model_ema.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger = get_root_logger()
    if optimizer is not None:
        # Try to get epoch from checkpoint; if missing, try extracting from filename
        epoch = checkpoint.get('epoch', None)
        if epoch is None:
            match = re.match(r'.*epoch_(\d*).*.pth', ckpt_file)
            if match:
                epoch = int(match.group(1))
            else:
                epoch = checkpoint.get('step', 0)  # If neither exists, use step or default 0
        logger.info(f'Resume checkpoint of epoch {epoch}' + (f', step {checkpoint.get("step", "N/A")}' if 'step' in checkpoint else '') + f' from {ckpt_file}. Load ema: {load_ema}, '
                    f'resume optimizer： {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}.')
        return epoch, missing, unexpect
    logger.info(f'Load checkpoint from {ckpt_file}. Load ema: {load_ema}.')
    return missing, unexpect
