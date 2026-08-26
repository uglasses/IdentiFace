import argparse
import datetime
import os
import sys
import time
import types
import warnings
import hashlib
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType, DistributedDataParallelKwargs
from mmcv.runner import LogBuffer
from torch.utils.data import RandomSampler
import numpy as np

from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.model.nets import PixArtMS, ControlPixArtHalf, ControlPixArtMSHalf
from diffusion.model.gaussian_diffusion import ModelMeanType
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

# Import AdaFace and VAE
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))
try:
    from AdaFace import AdaFace
    ADAFACE_AVAILABLE = True
except ImportError:
    ADAFACE_AVAILABLE = False
    warnings.warn("AdaFace not available. AdaFace loss will be disabled.")

try:
    from diffusers.models import AutoencoderKL
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    warnings.warn("VAE not available. AdaFace loss will be disabled.")

warnings.filterwarnings("ignore")  # ignore warning


def stack_control_conditions_from_data_info(data_info):
    """Stack per-sample ``condition`` / ``condition2`` from batch ``data_info`` (list of dicts)."""
    condition = None
    condition2 = None
    if isinstance(data_info, (list, tuple)) and len(data_info) > 0 and isinstance(data_info[0], dict):
        if 'condition' in data_info[0]:
            condition = torch.stack([info['condition'] for info in data_info])
        if 'condition2' in data_info[0]:
            condition2 = torch.stack([info['condition2'] for info in data_info])
    elif isinstance(data_info, dict):
        condition = data_info.get('condition', None)
        condition2 = data_info.get('condition2', None)
    return condition, condition2


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


def _split_data_info_to_per_sample(data_info, bs):
    """Normalize collated ``data_info`` into a list of length ``bs``."""
    if isinstance(data_info, (list, tuple)):
        out = []
        for i in range(bs):
            item = data_info[i] if i < len(data_info) and isinstance(data_info[i], dict) else {}
            out.append(item)
        return out

    if isinstance(data_info, dict):
        out = []
        keys = list(data_info.keys())
        for i in range(bs):
            sample = {}
            for k in keys:
                v = data_info.get(k)
                if isinstance(v, torch.Tensor):
                    if v.dim() > 0 and v.shape[0] == bs:
                        sample[k] = v[i]
                    else:
                        sample[k] = v
                elif isinstance(v, (list, tuple)):
                    sample[k] = v[i] if i < len(v) else None
                else:
                    sample[k] = v
            out.append(sample)
        return out

    return [{} for _ in range(bs)]


def _resolve_gt_path(data_info, data_root):
    """
    Resolve GT image path from current data format.
    Prefer ``data_info['path']``; fallback to ``data_info['img_path']``.
    """
    path_field = data_info.get('path', None)
    if isinstance(path_field, str) and path_field.strip():
        p = path_field.strip()
        if os.path.isabs(p):
            return p
        return os.path.join(data_root, p)
    
    img_path = data_info.get('img_path', None)
    # INSERT_YOUR_CODE
    if isinstance(img_path, str) and "10620.jpg" in img_path:
        raise RuntimeError("Forbidden image path: 10620.jpg found in img_path!")
        return None 
    if isinstance(img_path, str) and img_path.strip():
        p = img_path.strip()
        if os.path.isabs(p):
            return p
        return os.path.join(data_root, p)
    return None


def _predict_x0_from_model_output(train_diffusion, x_t, timesteps, model_output):
    """Convert model output to x0 prediction according to diffusion mean type."""
    if train_diffusion.model_mean_type == ModelMeanType.START_X:
        return model_output
    if train_diffusion.model_mean_type == ModelMeanType.EPSILON:
        return train_diffusion._predict_xstart_from_eps(x_t=x_t, t=timesteps, eps=model_output)
    raise NotImplementedError(
        f"AdaFace loss currently supports ModelMeanType.START_X/EPSILON, got {train_diffusion.model_mean_type}."
    )


def compute_adaface_loss(
    pred_x0_latents,
    vae,
    adaface_model,
    device,
    scale_factor=0.18215,
    reference_features=None,
    quality_gate=True,
    use_sigmoid_weight=True,
    same_person_threshold=0.3603,
    sigmoid_alpha=10.0,
    sigmoid_w_min=0.0,
    sigmoid_w_max=1.0,
):
    """
    AdaFace identity-consistency loss on predicted x0:
    1) Decode pred x0 latents.
    2) Optional teacher detect/align quality gate; skip failed samples.
    3) On valid samples, 1 - cos(f(pred), f(ref)).
       Optional sigmoid weight: w_i = w_min + (w_max - w_min) * sigmoid(alpha * (T - s_i))

    Args:
        pred_x0_latents: predicted x0 latent (B, C, H, W)
        reference_features: precomputed GT AdaFace features (B, D); None falls back to in-batch self-target
        vae: VAE decoder
        adaface_model: AdaFace model
        device: device
        scale_factor: VAE scale
        quality_gate: skip samples where teacher detect/align fails
        use_sigmoid_weight: weight by similarity vs same-person threshold
        same_person_threshold: same-identity threshold T
        sigmoid_alpha: sigmoid slope
        sigmoid_w_min: weight lower bound
        sigmoid_w_max: weight upper bound

    Returns:
        adaface_loss: 1 - mean cosine similarity, or None if no valid samples
        avg_similarity: mean cosine similarity, or None if no valid samples
        valid_ratio: fraction of samples that passed the quality gate
    """
    if not ADAFACE_AVAILABLE or not VAE_AVAILABLE or adaface_model is None or vae is None:
        return None, None, 0.0
    
    try:
        # decode predicted x0 latents
        pred_decoded = vae.decode(pred_x0_latents / scale_factor).sample
        pred_decoded = torch.clamp((pred_decoded + 1.0) / 2.0, 0.0, 1.0)
        bs = pred_decoded.shape[0]
        valid_mask = torch.ones(bs, device=device, dtype=torch.bool)

        # Quality gate: only keep samples where teacher face detection/alignment succeeds.
        if quality_gate:
            valid_flags = []
            for i in range(bs):
                try:
                    pred_pil = TF.to_pil_image(pred_decoded[i].detach().cpu())
                    _ = adaface_model.extract_feature_from_pil(pred_pil)
                    valid_flags.append(True)
                except Exception:
                    valid_flags.append(False)
            valid_mask = torch.tensor(valid_flags, device=device, dtype=torch.bool)

        if reference_features is not None:
            ref_feat_all = reference_features.to(device=device, dtype=torch.float32)
            ref_valid = torch.isfinite(ref_feat_all).all(dim=1)
            valid_mask = valid_mask & ref_valid

        if not bool(valid_mask.any().item()):
            return None, None, 0.0

        # Differentiable teacher input prep (BGR, [-1, 1], 112x112)
        pred_in = F.interpolate(pred_decoded, size=(112, 112), mode='bilinear', align_corners=False)
        pred_in = (pred_in[:, [2, 1, 0], :, :] - 0.5) / 0.5

        pred_feat, _ = adaface_model.model(pred_in)
        pred_feat = F.normalize(pred_feat[valid_mask], p=2, dim=1)
        if reference_features is not None:
            ref_feat = F.normalize(ref_feat_all[valid_mask], p=2, dim=1)
        else:
            # Fallback: self-target (not ideal). Kept for backward compatibility.
            ref_feat = pred_feat.detach()
        similarities = (pred_feat * ref_feat).sum(dim=1)

        per_sample_loss = (1.0 - similarities)
        if use_sigmoid_weight:
            w_min = float(sigmoid_w_min)
            w_max = float(sigmoid_w_max)
            if w_max < w_min:
                w_min, w_max = w_max, w_min
            # Below threshold -> larger weight; above threshold -> smaller weight.
            weights = torch.sigmoid(float(sigmoid_alpha) * (float(same_person_threshold) - similarities))
            weights = w_min + (w_max - w_min) * weights
            weight_sum = torch.clamp(weights.sum(), min=1e-8)
            adaface_loss = (per_sample_loss * weights).sum() / weight_sum
        else:
            adaface_loss = per_sample_loss.mean()
        avg_similarity = similarities.mean().item()
        valid_ratio = float(valid_mask.float().mean().item())

        return adaface_loss, avg_similarity, valid_ratio
    except Exception as e:
        # On failure, skip this AdaFace term.
        return None, None, 0.0


def _resolve_abs_gt_path_for_cache(path_value, data_root):
    if path_value is None:
        return None
    p = str(path_value).strip()
    if not p:
        return None
    if os.path.isabs(p):
        return os.path.realpath(p)
    # Prefer the path itself if it is already a valid workspace-relative GT path
    # (e.g. "dataset_train/celebA/xxx.png"), to avoid double-joining data_root.
    if os.path.exists(p):
        return os.path.realpath(p)
    return os.path.realpath(os.path.join(data_root, p))


def build_or_load_adaface_gt_cache(
    adaface_model,
    candidate_paths,
    data_root,
    cache_root,
    adaface_model_path,
):
    """
    Precompute AdaFace GT features before training and store/load from cache.
    Returns path->feature map for fast lookup in training loop.
    """
    os.makedirs(cache_root, exist_ok=True)
    model_real = os.path.realpath(adaface_model_path)
    st = os.stat(model_real)
    model_fp = f"{model_real}|size={st.st_size}"
    key = f"{model_fp}|count={len(candidate_paths)}"
    cache_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    cache_file = os.path.join(cache_root, f"adaface_gt_feat_{cache_hash}.npz")

    path_to_feat = {}
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            feats = data["feats"].astype(np.float32)
            paths = [str(p) for p in data["paths"].tolist()]
            path_to_feat = {paths[i]: feats[i] for i in range(len(paths))}
            print(f"[INFO] Loaded AdaFace GT cache: {cache_file} ({len(path_to_feat)} items)")
        except Exception as e:
            print(f"[WARN] Failed to load AdaFace GT cache {cache_file}: {e}")
            path_to_feat = {}

    missing = [p for p in candidate_paths if p not in path_to_feat]
    if missing:
        print(f"[INFO] AdaFace GT cache missing {len(missing)} features, extracting...")
        report_interval = 200
        success_count = 0
        fail_count = 0
        for idx, p in enumerate(missing, start=1):
            try:
                feat = adaface_model.extract_feature(p)
                path_to_feat[p] = np.asarray(feat, dtype=np.float32)
                success_count += 1
            except Exception:
                fail_count += 1
                continue
            if idx % report_interval == 0 or idx == len(missing):
                print(
                    "[INFO] AdaFace GT cache extracting progress: "
                    f"{idx}/{len(missing)} "
                    f"(success={success_count}, failed={fail_count})"
                )
        if path_to_feat:
            paths_sorted = sorted(path_to_feat.keys())
            feats_stacked = np.stack([path_to_feat[p] for p in paths_sorted], axis=0).astype(np.float32)
            np.savez_compressed(cache_file, paths=np.array(paths_sorted, dtype=object), feats=feats_stacked)
            print(f"[INFO] Saved AdaFace GT cache: {cache_file} ({len(paths_sorted)} items)")

    return path_to_feat, cache_file


def validate(val_dataloader, model, train_diffusion, config, accelerator, logger, epoch, global_step):
    """
    Compute diffusion loss on the validation set.

    Args:
        val_dataloader: validation dataloader
        model: model
        train_diffusion: diffusion trainer
        config: config
        accelerator: Accelerator
        logger: logger
        epoch: current epoch
        global_step: current global step

    Returns:
        val_loss: mean validation loss
    """
    model.eval()
    val_log_buffer = LogBuffer()
    total_val_loss = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            z = batch[0]
            clean_images = z * config.scale_factor
            y = batch[1]
            y_mask = batch[2]
            data_info = batch[3]
            
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            
            condition, condition2 = stack_control_conditions_from_data_info(data_info)
            model_kwargs = dict(y=y, mask=y_mask, data_info=data_info)
            if condition is not None:
                model_kwargs['c'] = condition * config.scale_factor
            if condition2 is not None:
                model_kwargs['c2'] = condition2 * config.scale_factor
            
            # Standard diffusion loss only (no AdaFace on val)
            loss_term = train_diffusion.training_losses(model, clean_images, timesteps, model_kwargs=model_kwargs)
            loss = loss_term['loss'].mean()
            
            total_val_loss += loss.item()
            val_steps += 1
            
            val_log_buffer.update({"val_loss": loss.item()})
    
    avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0.0
    val_log_buffer.average()
    
    if accelerator.is_main_process:
        logger.info(f"Validation - Epoch [{epoch}], Step [{global_step}]: " + 
                   ', '.join([f"{k}:{v:.4f}" for k, v in val_log_buffer.output.items()]))
        
        # Log to tensorboard
        accelerator.log({"val_loss": avg_val_loss}, step=global_step)
    
    model.train()
    return avg_val_loss


def train(model, train_diffusion, config, accelerator, logger, train_dataloader, val_dataloader, optimizer, lr_scheduler, start_epoch,
          adaface_gt_feat_map=None):
    # Access global variables for AdaFace and VAE
    global adaface_model, vae_decoder
    
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs
    freq_unfreeze_step = int(getattr(config, 'freq_fusion_unfreeze_step', -1))
    if freq_unfreeze_step < 0:
        freq_unfreeze_step = max(1, int(0.3 * total_steps))
    freq_unfrozen_logged = False
    
    val_interval = getattr(config, 'val_interval', 1)  # validate every N epochs
    val_steps = getattr(config, 'val_steps', None)  # also every N steps; None = epoch end only
    has_val_dataloader = val_dataloader is not None
    # AdaFace loss: only from this epoch onward (default 1 = unchanged legacy behavior).
    adaface_start_epoch = int(getattr(config, 'adaface_start_epoch', 1))
    freq_fusion_param_warmup_steps = int(getattr(config, 'freq_fusion_param_warmup_steps', 0))
    freq_fusion_param_warmup_start_factor = float(getattr(config, 'freq_fusion_param_warmup_start_factor', 0.1))
    freq_fusion_enabled_global_step = (
        0 if getattr(config, 'use_frequency_control_fusion', False) else None
    )

    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    if not load_vae_feat:
        raise ValueError("Only support load vae features for now.")
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        if accelerator.is_main_process and getattr(config, 'use_adaface_loss', False):
            if epoch < adaface_start_epoch:
                logger.info(
                    f"[INFO] Epoch {epoch}: AdaFace loss disabled (adaface_start_epoch={adaface_start_epoch})."
                )
            elif epoch == adaface_start_epoch:
                logger.info(
                    f"[INFO] Epoch {epoch}: AdaFace loss enabled (adaface_start_epoch={adaface_start_epoch})."
                )

        data_time_start = time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # Early training: keep frequency cutoff params (r0/k) fixed, then unfreeze later.
            if getattr(config, 'use_frequency_control_fusion', False):
                raw_model = accelerator.unwrap_model(model)
                freq_fusion = getattr(raw_model, 'freq_fusion', None)
                if (
                    freq_fusion is not None
                    and hasattr(freq_fusion, 'frequency_params_frozen')
                    and freq_fusion.frequency_params_frozen()
                    and global_step >= freq_unfreeze_step
                ):
                    freq_fusion.unfreeze_frequency_params()
                    if not freq_unfrozen_logged and accelerator.is_main_process:
                        logger.info(
                            f"[INFO] Unfroze FrequencyControlFusion r0/k at global_step={global_step} "
                            f"(freq_fusion_unfreeze_step={freq_unfreeze_step})."
                        )
                        freq_unfrozen_logged = True

            data_time_all += time.time() - data_time_start
            z = batch[0]  # 4 x 4 x 128 x 128 z:vae output, 3x1024x1024->vae->4x128x128
            clean_images = z * config.scale_factor  # vae needed scale factor
            y = batch[1]  # 4 x 1 x 120 x 4096 # T5 extracted feature of caption, 120 token, 4096
            y_mask = batch[2]  # 4 x 1 x 1 x 120 # caption indicate whether valid
            data_info = batch[3]

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            
            # AdaFace: low noise + step frequency + after warmup, then ramp
            adaface_warmup_steps = int(getattr(config, 'adaface_warmup_steps', 0))
            adaface_ramp_steps = int(getattr(config, 'adaface_ramp_steps', 1000))
            use_adaface_loss = (
                getattr(config, 'use_adaface_loss', False) and
                adaface_model is not None and
                vae_decoder is not None and
                (epoch >= adaface_start_epoch) and
                (getattr(config, 'adaface_loss_freq', 1) == 1 or 
                 global_step % getattr(config, 'adaface_loss_freq', 100) == 0) and
                # Only near the end of denoising (small timesteps)
                (timesteps.max().item() < getattr(config, 'adaface_max_timestep', 200)) and
                (global_step >= adaface_warmup_steps)
            )
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                x0_pred_for_adaface = None

                condition, condition2 = stack_control_conditions_from_data_info(data_info)
                model_kwargs = dict(y=y, mask=y_mask, data_info=data_info)
                if condition is not None:
                    model_kwargs['c'] = condition * config.scale_factor
                if (
                    condition2 is not None
                    and not bool(getattr(accelerator.unwrap_model(model), 'disable_freq_fusion_runtime', False))
                ):
                    model_kwargs['c2'] = condition2 * config.scale_factor

                loss_term = train_diffusion.training_losses(model, clean_images, timesteps, model_kwargs=model_kwargs)
                loss = loss_term['loss'].mean()

                # For AdaFace-on-x0_pred: run one extra forward only when needed.
                if use_adaface_loss:
                    noise_id = torch.randn_like(clean_images)
                    x_t_id = train_diffusion.q_sample(clean_images, timesteps, noise=noise_id)
                    model_output_id = model(x_t_id, timesteps, **model_kwargs)
                    if isinstance(model_output_id, dict) and model_output_id.get('x', None) is not None:
                        model_output_id = model_output_id['x']
                    c_id = x_t_id.shape[1]
                    if model_output_id.shape[1] == c_id * 2:
                        model_output_id, _ = torch.split(model_output_id, c_id, dim=1)
                    x0_pred_for_adaface = _predict_x0_from_model_output(
                        train_diffusion=train_diffusion,
                        x_t=x_t_id,
                        timesteps=timesteps,
                        model_output=model_output_id,
                    )

                # AdaFace loss (if enabled)
                adaface_loss_value = None
                adaface_similarity = None
                adaface_valid_ratio = None
                if use_adaface_loss:
                    # Fetch precomputed GT features by path for this batch.
                    ref_feats_batch = None
                    if adaface_gt_feat_map:
                        per_sample_infos = _split_data_info_to_per_sample(data_info, bs)
                        feat_list = []
                        feat_dim = None
                        for info in per_sample_infos:
                            p = None
                            if isinstance(info, dict):
                                p = _resolve_gt_path(info, config.data_root)
                            p = os.path.realpath(p) if isinstance(p, str) and p else None
                            feat = adaface_gt_feat_map.get(p) if p is not None else None
                            if feat is None:
                                feat_list.append(None)
                            else:
                                feat_arr = np.asarray(feat, dtype=np.float32)
                                feat_dim = feat_arr.shape[0] if feat_dim is None else feat_dim
                                feat_list.append(feat_arr)
                        if feat_dim is not None:
                            ref_feats_np = np.full((bs, feat_dim), np.nan, dtype=np.float32)
                            for i, feat in enumerate(feat_list):
                                if feat is not None:
                                    ref_feats_np[i] = feat
                            ref_feats_batch = torch.from_numpy(ref_feats_np).to(clean_images.device)

                    adaface_loss_value, adaface_similarity, adaface_valid_ratio = compute_adaface_loss(
                        x0_pred_for_adaface,
                        vae_decoder, 
                        adaface_model, 
                        clean_images.device,
                        scale_factor=config.scale_factor,
                        reference_features=ref_feats_batch,
                        quality_gate=bool(getattr(config, 'adaface_quality_gate', True)),
                        use_sigmoid_weight=bool(getattr(config, 'adaface_use_sigmoid_weight', True)),
                        same_person_threshold=float(getattr(config, 'adaface_same_person_threshold', 0.3603)),
                        sigmoid_alpha=float(getattr(config, 'adaface_sigmoid_alpha', 10.0)),
                        sigmoid_w_min=float(getattr(config, 'adaface_sigmoid_w_min', 0.0)),
                        sigmoid_w_max=float(getattr(config, 'adaface_sigmoid_w_max', 1.0)),
                    )
                    if adaface_loss_value is not None:
                        adaface_weight = float(getattr(config, 'adaface_loss_weight', 0.1))
                        ramp_progress = 1.0
                        if adaface_ramp_steps > 0:
                            ramp_progress = min(
                                max((global_step - adaface_warmup_steps) / float(adaface_ramp_steps), 0.0),
                                1.0,
                            )
                        loss = loss + (adaface_weight * ramp_progress) * adaface_loss_value
                
                accelerator.backward(loss)
                # Warmup newly-enabled freq_fusion params in the latter half:
                # scale only freq_fusion grads from start_factor -> 1.0 over warmup steps.
                if getattr(config, 'use_frequency_control_fusion', False):
                    raw_model = accelerator.unwrap_model(model)
                    freq_fusion = getattr(raw_model, 'freq_fusion', None)
                    if (
                        freq_fusion is not None
                        and not bool(getattr(raw_model, 'disable_freq_fusion_runtime', False))
                        and freq_fusion_enabled_global_step is not None
                        and freq_fusion_param_warmup_steps > 0
                    ):
                        elapsed = max(global_step - freq_fusion_enabled_global_step, 0)
                        progress = min(elapsed / float(freq_fusion_param_warmup_steps), 1.0)
                        grad_scale = freq_fusion_param_warmup_start_factor + (
                            1.0 - freq_fusion_param_warmup_start_factor
                        ) * progress
                        for p in freq_fusion.parameters():
                            if p.grad is not None:
                                p.grad.mul_(grad_scale)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {"loss": accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            if adaface_similarity is not None:
                logs.update(adaface_similarity=adaface_similarity)
            if adaface_loss_value is not None:
                logs.update(adaface_loss=accelerator.gather(adaface_loss_value).mean().item())
            if adaface_valid_ratio is not None:
                logs.update(adaface_valid_ratio=adaface_valid_ratio)
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                # avg_loss = sum(loss_buffer) / len(loss_buffer)
                log_buffer.average()
                info = f"Step/Epoch [{(epoch - 1) * len(train_dataloader) + step + 1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({data_info['img_hw'][0][0].item()}, {data_info['img_hw'][0][1].item()}), "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            if (global_step + 1) % 1000 == 0 and config.s3_work_dir is not None:
                logger.info(f"s3_work_dir: {config.s3_work_dir}")

            # Periodic validation if val_steps is set
            # IMPORTANT: All ranks must participate in validation, not just main process
            # because the validation dataloader is distributed
            if has_val_dataloader and val_steps is not None and (global_step + 1) % val_steps == 0:
                if accelerator.is_main_process:
                    logger.info(f"Running validation at step {global_step + 1}...")
                # All ranks participate in validation
                val_loss = validate(val_dataloader, model, train_diffusion, config, accelerator, logger, epoch, global_step + 1)
                if accelerator.is_main_process:
                    logger.info(f"Validation completed. Average val_loss: {val_loss:.4f}")

            global_step += 1
            data_time_start = time.time()

            synchronize()
            if accelerator.is_main_process:
                if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                    os.umask(0o000)  # file permission: 666; dir permission: 777
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
            synchronize()

        synchronize()
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        # End-of-epoch validation if a val set is configured
        # IMPORTANT: All ranks must participate in validation, not just main process
        # because the validation dataloader is distributed
        if has_val_dataloader and (epoch % val_interval == 0 or epoch == config.num_epochs):
            if accelerator.is_main_process:
                logger.info(f"Running validation at end of epoch {epoch}...")
            # All ranks participate in validation
            val_loss = validate(val_dataloader, model, train_diffusion, config, accelerator, logger, epoch, global_step)
            if accelerator.is_main_process:
                logger.info(f"Validation completed. Average val_loss: {val_loss:.4f}")
        
        synchronize()
        if accelerator.is_main_process:
            if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
                os.umask(0o000)  # file permission: 666; dir permission: 777
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=(epoch - 1) * len(train_dataloader) + step + 1,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
        synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the dir to save logs and models')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true')
    parser.add_argument('--resume_lr_scheduler', action='store_true')
    parser.add_argument(
        '--long-prompt-ratio',
        type=float,
        default=None,
        dest='long_prompt_ratio',
        help='Probability in [0,1] to use long_prompt branch (T5 from prompt_feature_long); '
             '1-long_prompt_ratio uses short prompt_feature. Overrides config when set.',
    )
    parser.add_argument(
        '--use_long_prompt',
        action='store_true',
        help='Deprecated: equivalent to --long-prompt-ratio 1.0',
    )
    parser.add_argument('--use_sketch_blurred', action='store_true',
                        help='Use sketch_feature directory for condition images (from InternData/sketch_feature_1024/)')
    parser.add_argument(
        '--use-frequency-control-fusion',
        action='store_true',
        help='Use FrequencyControlFusion (high-frequency edges + low-frequency LQ + small residual)',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.data_root:
        config.data_root = args.data_root
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=args.resume_optimizer,
            resume_lr_scheduler=args.resume_lr_scheduler)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 6
        config.optimizer.update({'lr': args.lr})
    
    # long_prompt_ratio: CLI overrides config (mix short/long T5 features)
    if isinstance(config.data, dict):
        d = config.data
    else:
        d = dict(config.data) if hasattr(config.data, '__dict__') else {}
        config.data = d
    if args.long_prompt_ratio is not None:
        r = float(args.long_prompt_ratio)
        if not (0.0 <= r <= 1.0):
            raise ValueError(f'--long-prompt-ratio must be in [0, 1], got {r}')
        d['long_prompt_ratio'] = r
        d.pop('use_long_prompt', None)
    elif args.use_long_prompt:
        d['long_prompt_ratio'] = 1.0
        d.pop('use_long_prompt', None)

    # --use_sketch_blurred overrides controlnet_feat_dir in config.
    # mmcv build_from_cfg reads cfg fields, so mutate config.data directly.
    if args.use_sketch_blurred:
        feat_dir_name = f"sketch_feature_{config.image_size}"
        if isinstance(config.data, dict):
            config.data['controlnet_feat_dir'] = feat_dir_name
        else:
            data_dict = dict(config.data) if hasattr(config.data, '__dict__') else {}
            data_dict['controlnet_feat_dir'] = feat_dir_name
            config.data = data_dict
        # Logger is not ready yet; print so the path also appears later in logs.
        print(f"[INFO] Overriding config.data.controlnet_feat_dir to: {feat_dir_name} because --use_sketch_blurred is set")

    if args.use_frequency_control_fusion:
        config.use_frequency_control_fusion = True
        if isinstance(config.data, dict):
            if not config.data.get('controlnet_feat_dir_2'):
                print(f"[WARN] --use-frequency-control-fusion set but config.data.controlnet_feat_dir_2 is empty; set second feature dir in config.")

    # Without frequency fusion: a single ControlNet condition (edges or lq), not both.
    use_freq_fusion = bool(getattr(config, 'use_frequency_control_fusion', False))
    if not use_freq_fusion:
        single = str(getattr(config, 'single_control_source', 'edges')).strip().lower()
        if single not in ('edges', 'lq'):
            raise ValueError(
                f"single_control_source must be one of 'edges' | 'lq' when frequency fusion is disabled, got {single!r}"
            )
        if isinstance(config.data, dict):
            d = config.data
        else:
            d = dict(config.data) if hasattr(config.data, '__dict__') else {}
            config.data = d
        res = int(getattr(config, 'image_size', 1024))
        edges_dir = d.get('controlnet_feat_dir') or f'edges_feature_{res}'
        lq_dir = d.get('controlnet_feat_dir_2') or f'lq_feature_{res}'
        if single == 'edges':
            d['controlnet_feat_dir'] = edges_dir
        else:
            d['controlnet_feat_dir'] = lq_dir
            d['controlnet_modality'] = 'lq'
        d.pop('controlnet_feat_dir_2', None)
        print(
            f"[INFO] Single-control mode (no fusion): single_control_source={single!r}, "
            f"using InternData/{d['controlnet_feat_dir']}/ as the only condition."
        )

    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=9600)  # change timeout to avoid a strange NCCL bug
    # Default False: avoids extra autograd traversal + reducer warning when all params participate each step.
    # Set config.ddp_find_unused_parameters=True if you toggle e.g. disable_freq_fusion_runtime so some
    # submodules skip the forward graph on some iterations (otherwise DDP may error on unused grads).
    ddp_find_unused = bool(getattr(config, 'ddp_find_unused_parameters', False))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    # even_batches parameter is not supported in this version of accelerate
    # even_batches = True
    # if config.multi_scale:
    #     even_batches = False

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[init_handler, ddp_kwargs]
    )

    logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

    _data = config.data if isinstance(config.data, dict) else {}
    long_prompt_ratio_cfg = _data.get('long_prompt_ratio', 0.0)
    logger.info(
        f"long_prompt_ratio={long_prompt_ratio_cfg} (short: InternData/{_data.get('prompt_feature_dir', 'prompt_feature')}, "
        f"long: InternData/{_data.get('prompt_feature_long_dir', 'prompt_feature_long')})"
    )

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [512, 1024]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps))
    model: PixArtMS = build_model(config.model,
                                  config.grad_checkpointing,
                                  config.get('fp32_attention', False),
                                  input_size=latent_size,
                                  learn_sigma=learn_sigma,
                                  pred_sigma=pred_sigma,
                                  **model_kwargs)

    if config.load_from is not None and args.resume_from is None:
        # load from PixArt model
        missing, unexpected = load_checkpoint(config.load_from, model)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    use_frequency_control_fusion = bool(getattr(config, 'use_frequency_control_fusion', False))
    if image_size == 1024:
        model: ControlPixArtMSHalf = ControlPixArtMSHalf(
            model,
            copy_blocks_num=config.copy_blocks_num,
            use_frequency_control_fusion=use_frequency_control_fusion,
        ).train()
    else:
        model: ControlPixArtHalf = ControlPixArtHalf(
            model,
            copy_blocks_num=config.copy_blocks_num,
            use_frequency_control_fusion=use_frequency_control_fusion,
        ).train()
    if use_frequency_control_fusion:
        logger.info(
            "FrequencyControlFusion enabled: high-frequency edges + low-frequency LQ + small residual."
        )
        if getattr(model, 'freq_fusion', None) is not None:
            model.freq_fusion.freeze_frequency_params()
            logger.info("[INFO] FrequencyControlFusion r0/k are frozen at startup (will unfreeze later).")

    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"T5 max token length: {config.model_max_length}")

    # if args.local_rank == 0:
    #     for name, params in model.named_parameters():
    #         if params.requires_grad == False: logger.info(f"freeze param: {name}")
    #
    #     for name, params in model.named_parameters():
    #         if params.requires_grad == True: logger.info(f"trainable param: {name}")

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    train_ratio = getattr(config, 'train_ratio', 1.0)
    val_ratio = getattr(config, 'val_ratio', 0.0)
    
    # --use_sketch_blurred overrides controlnet_feat_dir; otherwise None lets the
    # dataset use the config value or its default naming.
    controlnet_feat_dir = None
    if args.use_sketch_blurred:
        controlnet_feat_dir = f'sketch_feature_{image_size}'
        logger.info(f"Using sketch_blurred mode: controlnet_feat_dir={controlnet_feat_dir}")

    # Build train dataset
    train_dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type, 
                                  train_ratio=train_ratio, mode='train', controlnet_feat_dir=controlnet_feat_dir)
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(train_dataset), dataset=train_dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=train_dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=train_dataset.ratio_nums, config=config, valid_num=1)
        # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
        #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
        #                                                 ratio_nums=dataset.ratio_nums)
        train_dataloader = build_dataloader(train_dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
    else:
        train_dataloader = build_dataloader(train_dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)
    
    # Build val dataset if configured
    val_dataloader = None
    if val_ratio > 0:
        logger.info(f"Building validation dataset with ratio: {val_ratio}")
        val_dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
                                    train_ratio=val_ratio, mode='val', controlnet_feat_dir=controlnet_feat_dir)
        val_batch_size = getattr(config, 'val_batch_size', config.train_batch_size)
        val_dataloader = build_dataloader(val_dataset, num_workers=config.num_workers, batch_size=val_batch_size, shuffle=False)
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    else:
        logger.info("No validation dataset configured (val_ratio=0)")

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    _opt_parts = [model.controlnet]
    if getattr(model, 'freq_fusion', None) is not None:
        _opt_parts.append(model.freq_fusion)
    optimizer = build_optimizer(nn.ModuleList(_opt_parts), config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        if args.resume_optimizer == False or args.resume_lr_scheduler == False:
            missing, unexpected = load_checkpoint(args.resume_from, model)
        else:
            start_epoch, missing, unexpected = load_checkpoint(**config.resume_from,
                                                               model=model,
                                                               optimizer=optimizer,
                                                               lr_scheduler=lr_scheduler,
                                                               )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # Initialize AdaFace and VAE for face similarity loss (if enabled)
    global adaface_model, vae_decoder
    adaface_model = None
    vae_decoder = None
    use_adaface_loss = getattr(config, 'use_adaface_loss', False)
    
    if use_adaface_loss:
        if ADAFACE_AVAILABLE and VAE_AVAILABLE:
            adaface_model_path = getattr(config, 'adaface_model_path', 'models/adaface_ir101_webface12m.ckpt')
            if os.path.exists(adaface_model_path):
                try:
                    logger.info(f"Loading AdaFace model from: {adaface_model_path}")
                    adaface_model = AdaFace(adaface_model_path, device=accelerator.device)
                    adaface_model.eval()
                    for param in adaface_model.parameters():
                        param.requires_grad = False
                    logger.info("AdaFace model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load AdaFace model: {e}. AdaFace loss will be disabled.")
                    adaface_model = None
            else:
                logger.warning(f"AdaFace model not found at: {adaface_model_path}. AdaFace loss will be disabled.")
            
            # Load VAE decoder
            vae_pretrained = getattr(config, 'vae_pretrained', 'models/sd-vae-ft-ema')
            try:
                logger.info(f"Loading VAE decoder from: {vae_pretrained}")
                vae_decoder = AutoencoderKL.from_pretrained(vae_pretrained)
                vae_decoder.to(accelerator.device)
                vae_decoder.eval()
                for param in vae_decoder.parameters():
                    param.requires_grad = False
                logger.info("VAE decoder loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load VAE decoder: {e}. AdaFace loss will be disabled.")
                vae_decoder = None
        else:
            logger.warning("AdaFace or VAE not available. AdaFace loss will be disabled.")
            use_adaface_loss = False
    
    if use_adaface_loss and (adaface_model is None or vae_decoder is None):
        logger.warning("AdaFace loss is enabled but models are not loaded. Disabling AdaFace loss.")
        use_adaface_loss = False

    # Precompute/load GT AdaFace features before the whole training loop.
    # Only the main process extracts missing features and writes the cache file; other ranks
    # reload after a barrier to avoid triple work and npz write races under multi-GPU.
    adaface_gt_feat_map = None
    if use_adaface_loss and adaface_model is not None and train_dataset is not None:
        try:
            candidate_paths_raw = getattr(train_dataset, 'img_samples', [])
            candidate_paths = []
            for p in candidate_paths_raw:
                abs_p = _resolve_abs_gt_path_for_cache(p, config.data_root)
                if abs_p is not None and os.path.exists(abs_p):
                    candidate_paths.append(abs_p)
            candidate_paths = sorted(set(candidate_paths))
            cache_root = os.path.join(config.data_root, ".adaface_cache")
            adaface_model_path_resolved = getattr(
                config, "adaface_model_path", "models/adaface_ir101_webface12m.ckpt"
            )
            if accelerator.is_main_process:
                adaface_gt_feat_map, cache_file = build_or_load_adaface_gt_cache(
                    adaface_model=adaface_model,
                    candidate_paths=candidate_paths,
                    data_root=config.data_root,
                    cache_root=cache_root,
                    adaface_model_path=adaface_model_path_resolved,
                )
                logger.info(
                    f"[INFO] AdaFace GT cache ready: {len(adaface_gt_feat_map)}/{len(candidate_paths)} "
                    f"features from {cache_file}"
                )
            accelerator.wait_for_everyone()
            if not accelerator.is_main_process:
                adaface_gt_feat_map, cache_file = build_or_load_adaface_gt_cache(
                    adaface_model=adaface_model,
                    candidate_paths=candidate_paths,
                    data_root=config.data_root,
                    cache_root=cache_root,
                    adaface_model_path=adaface_model_path_resolved,
                )
                logger.info(
                    f"[INFO] AdaFace GT cache loaded (rank {accelerator.process_index}): "
                    f"{len(adaface_gt_feat_map)}/{len(candidate_paths)} from {cache_file}"
                )
        except Exception as e:
            logger.warning(f"[WARNING] Failed to build/load AdaFace GT cache: {e}")
            adaface_gt_feat_map = None

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model,)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    
    # Prepare validation dataloader if it exists
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    train(
        model,
        train_diffusion,
        config,
        accelerator,
        logger,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        start_epoch,
        adaface_gt_feat_map=adaface_gt_feat_map,
    )
