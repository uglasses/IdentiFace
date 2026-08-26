#!/usr/bin/env python3
"""
Full ControlNet dataset preprocessing script.
Converts raw data in dataset_train into the format required by the training code.

Steps: 1 GT VAE features → 2 T5 → 3 edges/sketch condition features (control image encoded directly with VAE) → 4 LQ degraded-image VAE features (lq_feature_{resolution}, downsample then upsample)
→ 5 hard LQ (camera_lq_pipeline complex camera degradation → native low-res PNG + upsampled VAE .npz).
"""
import os
import json
import sys
import importlib.util
import random
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from diffusers.models import AutoencoderKL
import cv2

# Add project paths
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "PixArt-alpha"))  # Prefer PixArt-alpha/diffusion

try:
    from diffusion.model.t5 import T5Embedder
except ImportError as e:
    print(f"Warning: Could not import T5Embedder: {e}")
    print("Make sure you're running from the project root directory")

from long_prompt_segmentation import (
    process_long_prompt_segmented,
    format_random_list_line,
)

try:
    from AdaFace import _parse_low_res, _get_pil_resample
except ImportError:
    from typing import Tuple as _Tuple

    def _parse_low_res(low_res: str) -> _Tuple[int, int]:
        low_res = str(low_res).strip().lower()
        if "x" in low_res:
            w_str, h_str = low_res.split("x", 1)
            return int(w_str), int(h_str)
        v = int(low_res)
        return v, v

    def _get_pil_resample(method: str) -> int:
        method = str(method).strip().lower()
        mapping = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX,
        }
        if method not in mapping:
            raise ValueError(f"Unknown resample method '{method}'. Supported: {', '.join(mapping.keys())}")
        return mapping[method]


def _tensor_minus1_to_pil_rgb(t: torch.Tensor) -> Image.Image:
    """[1,3,H,W] in [-1,1] -> PIL RGB."""
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    arr = ((t + 1.0) * 0.5 * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def _pil_rgb_to_tensor_minus1(pil: Image.Image, device, dtype) -> torch.Tensor:
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device=device, dtype=dtype)


def _load_camera_lq_pipeline_module():
    """Load other_tools/camera_lq_pipeline.py without requiring other_tools to be a package."""
    path = current_file_path.parent / "other_tools" / "camera_lq_pipeline.py"
    if not path.is_file():
        raise FileNotFoundError(f"camera_lq_pipeline not found: {path}")
    spec = importlib.util.spec_from_file_location("relactrl_camera_lq_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses etc. look up sys.modules[cls.__module__]; must register before exec or AttributeError
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pil_square_preprocess(
    img: Image.Image,
    resolution: int,
    preserve_aspect_ratio: bool,
) -> Image.Image:
    """Same geometry preprocessing as step1/step4; returns a square RGB PIL image."""
    img = img.convert("RGB")
    if preserve_aspect_ratio:
        w, h = img.size
        scale = resolution / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        try:
            bicubic = Image.Resampling.BICUBIC
        except AttributeError:
            bicubic = Image.BICUBIC
        img = img.resize((new_w, new_h), bicubic)
        canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
        offset_x = (resolution - new_w) // 2
        offset_y = (resolution - new_h) // 2
        canvas.paste(img, (offset_x, offset_y))
        return canvas
    geom = T.Compose(
        [
            T.Resize(resolution),
            T.CenterCrop(resolution),
        ]
    )
    return geom(img)


def degrade_image_resolution(
    gt_image_normalized: torch.Tensor,
    low_res: str,
    downsample_method: str,
    upsample_method: str,
) -> torch.Tensor:
    """
    Downsample to low resolution, then interpolate back to the original size.
    gt_image_normalized: [1,3,H,W], [-1,1]
    """
    _, _, H, W = gt_image_normalized.shape
    low_w, low_h = _parse_low_res(low_res)
    low_w = max(1, min(int(low_w), int(W)))
    low_h = max(1, min(int(low_h), int(H)))
    pil = _tensor_minus1_to_pil_rgb(gt_image_normalized)
    pil_low = pil.resize((low_w, low_h), resample=_get_pil_resample(downsample_method))
    pil_up = pil_low.resize((W, H), resample=_get_pil_resample(upsample_method))
    return _pil_rgb_to_tensor_minus1(pil_up, gt_image_normalized.device, gt_image_normalized.dtype)


def step1_extract_vae_features(metadata_path, image_dir, output_dir, vae_path, resolution=1024, device='cuda', preserve_aspect_ratio=False):
    """Step 1: Extract VAE features.
    
    Args:
        preserve_aspect_ratio: If True, keep aspect ratio and pad to square; if False, resize+crop to square
    """
    print("=" * 50)
    print("Step 1: Extracting VAE features...")
    
    if not torch.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA not available, using CPU (will be very slow)")
    
    print(f"Loading VAE from {vae_path}...")
    # use_safetensors=False to avoid warnings when only .bin files exist
    vae = AutoencoderKL.from_pretrained(vae_path, use_safetensors=False).to(device)
    vae.eval()
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    if preserve_aspect_ratio:
        # Keep aspect ratio: resize short side to resolution, then pad to square
        def resize_pad(img):
            img = img.convert('RGB')
            w, h = img.size
            # Compute scale factor
            scale = resolution / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            # Compatible with different PIL versions of BICUBIC
            try:
                bicubic = Image.Resampling.BICUBIC
            except AttributeError:
                bicubic = Image.BICUBIC
            img = img.resize((new_w, new_h), bicubic)
            # Create square canvas and paste centered
            canvas = Image.new('RGB', (resolution, resolution), (0, 0, 0))
            offset_x = (resolution - new_w) // 2
            offset_y = (resolution - new_h) // 2
            canvas.paste(img, (offset_x, offset_y))
            return canvas
        
        transform = T.Compose([
            T.Lambda(resize_pad),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])
    else:
        # Default: resize + center crop (changes aspect ratio)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, item in enumerate(tqdm(data, desc="Extracting VAE features")):
        # Path handling: item['path'] may include a celebA/ prefix; join correctly
        path = item['path']
        if path.startswith('celebA/'):
            path = path[7:]  # Strip 'celebA/' prefix
        img_path = os.path.join(image_dir, path)
        save_path = os.path.join(
            output_dir,
            Path(path).stem + '.npy'
        )
        
        if os.path.exists(save_path):
            skipped += 1
            continue
        
        try:
            # If file is missing, try alternate extensions (.jpg / .png)
            if not os.path.exists(img_path):
                if img_path.endswith('.png'):
                    img_path_alt = img_path.replace('.png', '.jpg')
                    if os.path.exists(img_path_alt):
                        img_path = img_path_alt
                elif img_path.endswith('.jpg'):
                    img_path_alt = img_path.replace('.jpg', '.png')
                    if os.path.exists(img_path_alt):
                        img_path = img_path_alt
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                errors += 1
                continue
            
            if idx % 10 == 0:
                print(f"[Step1-debug] idx={idx} item_path={item.get('path','')} img_path={img_path} save_path={save_path}")

            img = Image.open(img_path)
            img = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                posterior = vae.encode(img).latent_dist
                z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy().squeeze()
            
            np.save(save_path, z)
            processed += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            errors += 1
    
    print(f"✓ VAE features: {processed} processed, {skipped} skipped, {errors} errors")
    print(f"  Saved to: {output_dir}")


def step4_extract_lq_vae_features(
    metadata_path,
    image_dir,
    output_dir,
    vae_path,
    resolution=1024,
    preserve_aspect_ratio=False,
    lq_low_res='8',
    lq_downsample_method='bicubic',
    lq_upsample_method='bicubic',
):
    """
    Step 4: Same-size LQ image as GT (downsample then upsample) → VAE encode → .npz (mean+std), same as sketch/edges.

    Default output dir: InternData/lq_feature_{resolution} (e.g. lq_feature_1024).
    """
    print("=" * 50)
    print("Step 4: Extracting LQ (degraded) VAE features (same pipeline as inference mask LQ)...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU (will be slow)")

    print(f"  LQ: low_res={lq_low_res}, down={lq_downsample_method}, up={lq_upsample_method}")
    print(f"Loading VAE from {vae_path}...")
    vae = AutoencoderKL.from_pretrained(vae_path, use_safetensors=False).to(device)
    vae.eval()

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    if preserve_aspect_ratio:

        def resize_pad(img):
            img = img.convert('RGB')
            w, h = img.size
            scale = resolution / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            try:
                bicubic = Image.Resampling.BICUBIC
            except AttributeError:
                bicubic = Image.BICUBIC
            img = img.resize((new_w, new_h), bicubic)
            canvas = Image.new('RGB', (resolution, resolution), (0, 0, 0))
            offset_x = (resolution - new_w) // 2
            offset_y = (resolution - new_h) // 2
            canvas.paste(img, (offset_x, offset_y))
            return canvas

        transform = T.Compose([
            T.Lambda(resize_pad),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])

    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    skipped = 0
    errors = 0

    for idx, item in enumerate(tqdm(data, desc="LQ → VAE features")):
        path = item['path']
        if path.startswith('celebA/'):
            path = path[7:]
        img_path = os.path.join(image_dir, path)
        save_path = os.path.join(output_dir, Path(path).stem + '.npz')

        if os.path.exists(save_path):
            skipped += 1
            continue

        try:
            if not os.path.exists(img_path):
                if img_path.endswith('.png'):
                    img_path_alt = img_path.replace('.png', '.jpg')
                    if os.path.exists(img_path_alt):
                        img_path = img_path_alt
                elif img_path.endswith('.jpg'):
                    img_path_alt = img_path.replace('.jpg', '.png')
                    if os.path.exists(img_path_alt):
                        img_path = img_path_alt

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                errors += 1
                continue
            
            if idx % 10 == 0:
                print(f"[Step4-debug] idx={idx} item_path={item.get('path','')} img_path={img_path} save_path={save_path}")

            img = Image.open(img_path)
            x = transform(img).unsqueeze(0).to(device)
            x_lq = degrade_image_resolution(
                x, lq_low_res, lq_downsample_method, lq_upsample_method
            )

            with torch.no_grad():
                posterior = vae.encode(x_lq).latent_dist
                condition_feat = torch.cat([posterior.mean, posterior.std], dim=1)
                condition_feat = condition_feat.cpu().numpy()
                if condition_feat.ndim == 4 and condition_feat.shape[0] == 1:
                    condition_feat = condition_feat.squeeze(0)

            np.savez_compressed(save_path, arr_0=condition_feat)
            processed += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            errors += 1

    print(f"✓ LQ VAE features: {processed} processed, {skipped} skipped, {errors} errors")
    print(f"  Saved to: {output_dir}")


def step5_hard_lq_generation(
    metadata_path,
    image_dir,
    vae_npz_output_dir,
    native_lq_png_dir,
    vae_path,
    resolution=1024,
    preserve_aspect_ratio=False,
    camera_target_size=16,
    hard_lq_seed=20260409,
    hard_lq_strength=1.0,
    hard_lq_isp_gamma=0.92,
    hard_lq_surveillance_grade=True,
):
    """
    Step 5: camera_lq_pipeline two-stage camera degradation → native-size LQ PNG;
    then upsample LQ to resolution, VAE encode, save .npz in the same format as step4 (arr_0 = mean∥std).
    Skip sample if .npz already exists (same as step4); to regenerate, delete the corresponding .npz (PNG is overwritten too).
    """
    print("=" * 50)
    print("Step 5: hard LQ (camera pipeline) → native PNG + VAE on upscaled LQ...")

    cam = _load_camera_lq_pipeline_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, using CPU (will be slow)")

    print(f"  camera_target_size={camera_target_size}, encode_resolution={resolution}")
    print(f"  native LQ PNG dir: {native_lq_png_dir}")
    print(f"  VAE .npz dir: {vae_npz_output_dir}")
    print(f"Loading VAE from {vae_path}...")
    vae = AutoencoderKL.from_pretrained(vae_path, use_safetensors=False).to(device)
    vae.eval()

    rng = np.random.default_rng(int(hard_lq_seed))
    encode_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    try:
        up_resample = Image.Resampling.LANCZOS
    except AttributeError:
        up_resample = Image.LANCZOS

    with open(metadata_path, "r") as f:
        data = json.load(f)

    os.makedirs(vae_npz_output_dir, exist_ok=True)
    os.makedirs(native_lq_png_dir, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0

    for idx, item in enumerate(tqdm(data, desc="hard LQ → PNG + VAE")):
        path = item["path"]
        if path.startswith("celebA/"):
            path = path[7:]
        img_path = os.path.join(image_dir, path)
        stem = Path(path).stem
        save_npz = os.path.join(vae_npz_output_dir, stem + ".npz")
        save_png = os.path.join(native_lq_png_dir, stem + ".png")

        if os.path.exists(save_npz):
            skipped += 1
            continue

        try:
            if not os.path.exists(img_path):
                if img_path.endswith(".png"):
                    img_path_alt = img_path.replace(".png", ".jpg")
                    if os.path.exists(img_path_alt):
                        img_path = img_path_alt
                elif img_path.endswith(".jpg"):
                    img_path_alt = img_path.replace(".jpg", ".png")
                    if os.path.exists(img_path_alt):
                        img_path = img_path_alt

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                errors += 1
                continue

            if idx % 10 == 0:
                print(
                    f"[hard_LQ-debug] idx={idx} item_path={item.get('path','')} "
                    f"img_path={img_path} npz={save_npz}"
                )

            img = Image.open(img_path)
            pil_sq = _pil_square_preprocess(img, resolution, preserve_aspect_ratio)

            params = cam.sample_params(rng, int(camera_target_size), float(hard_lq_strength))
            _oh, _sc, _snq, lq_native = cam.degrade_image(
                pil_sq,
                rng,
                params,
                float(hard_lq_isp_gamma),
                surveillance_grade=bool(hard_lq_surveillance_grade),
                surveillance_strength=float(hard_lq_strength),
            )

            if lq_native.size[0] != int(camera_target_size) or lq_native.size[1] != int(camera_target_size):
                raise RuntimeError(
                    f"Expected LQ {camera_target_size}x{camera_target_size}, got {lq_native.size}"
                )

            lq_native.save(save_png)

            lq_up = lq_native.resize((resolution, resolution), up_resample)
            x = encode_transform(lq_up).unsqueeze(0).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                posterior = vae.encode(x).latent_dist
                condition_feat = torch.cat([posterior.mean, posterior.std], dim=1)
                condition_feat = condition_feat.cpu().numpy()
                if condition_feat.ndim == 4 and condition_feat.shape[0] == 1:
                    condition_feat = condition_feat.squeeze(0)

            np.savez_compressed(save_npz, arr_0=condition_feat)
            processed += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            errors += 1

    print(
        f"✓ Step 5 hard LQ: {processed} processed, {skipped} skipped, {errors} errors"
    )
    print(f"  Native LQ PNG: {native_lq_png_dir}")
    print(f"  VAE npz: {vae_npz_output_dir}")


def step2_extract_t5_features(
    metadata_path,
    output_dir,
    t5_model_path=None,
    t5_cache_dir=None,
    max_length=120,
    device='cuda',
    use_long_prompt=False,
):
    """Step 2: Extract T5 text features.
    
    Args:
        metadata_path: Path to metadata JSON file
        output_dir: Output directory
        t5_model_path: Path to T5 model
        t5_cache_dir: T5 cache directory
        max_length: Maximum text length
        device: Device
        use_long_prompt: If True, extract features from long_prompt; if False, use original prompt
    """
    print("=" * 50)
    prompt_type = "long_prompt" if use_long_prompt else "prompt"
    print(f"Step 2: Extracting T5 text features (using {prompt_type})...")
    
    if not torch.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA not available, using CPU")
    
    print(f"Loading T5 model...")
    # If model dir exists and contains model files, use it directly
    if t5_model_path and os.path.exists(t5_model_path) and os.path.exists(os.path.join(t5_model_path, "config.json")):
        print(f"  Using local model from: {t5_model_path}")
        # Ensure absolute path
        t5_model_path_abs = os.path.abspath(t5_model_path)
        # Check whether tokenizer files exist
        tokenizer_files = ['tokenizer_config.json', 'spiece.model', 'special_tokens_map.json']
        has_tokenizer = all(os.path.exists(os.path.join(t5_model_path_abs, f)) for f in tokenizer_files)
        
        if has_tokenizer:
            # Tokenizer files present; use local path directly
            # Note: when local_cache=False and dir_or_name is a path, T5Embedder uses that path as the model path
            # but sets the tokenizer path to the cache dir, which causes problems
            # Fix: use local_cache=True, set cache_dir to the model dir's parent, dir_or_name to the dir name
            print(f"  Tokenizer files found in model directory")
            # Create a temporary T5Embedder subclass to set paths correctly
            # Due to T5Embedder logic limits, we manually set tokenizer and model paths
            class CustomT5Embedder(T5Embedder):
                def __init__(self, device, model_path, model_max_length=120):
                    # Use model path directly, bypassing T5Embedder's complex logic
                    # torch is already imported at module level; no need to re-import
                    from transformers import T5EncoderModel, AutoTokenizer
                    self.device = torch.device(device)
                    self.torch_dtype = torch.bfloat16
                    self.use_text_preprocessing = True
                    self.model_max_length = model_max_length
                    print(f"  Loading tokenizer from: {model_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    print(f"  Loading model from: {model_path}")
                    self.model = T5EncoderModel.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=self.torch_dtype,
                        device_map={'shared': self.device, 'encoder': self.device}
                    ).eval()
            
            t5 = CustomT5Embedder(device, t5_model_path_abs, max_length)
        else:
            # Tokenizer files missing; download tokenizer via cache mode
            print(f"  Tokenizer files not found, will download to cache")
            t5 = T5Embedder(
                device=device,
                dir_or_name='t5-v1_1-xxl',  # Use model name so code downloads tokenizer
                local_cache=True,  # Use cache mode
                cache_dir=t5_cache_dir,  # Use cache directory
                model_max_length=max_length
            )
            # Manually load model weights
            # torch is already imported at module level; no need to re-import
            from transformers import T5EncoderModel
            print(f"  Loading model weights from: {t5_model_path_abs}")
            t5.model = T5EncoderModel.from_pretrained(t5_model_path_abs, device_map={'shared': device, 'encoder': device}).eval()
    else:
        print(f"  Model not found locally, using cache directory: {t5_cache_dir}")
        print(f"  Note: This will attempt to download from HuggingFace if model not in cache")
        # Cache mode: T5Embedder creates t5-v1_1-xxl under cache_dir
        t5 = T5Embedder(
            device=device,
            local_cache=True,
            cache_dir=t5_cache_dir,
            model_max_length=max_length
        )
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    errors = 0
    
    long_prompt_count = 0
    processed_prompt_count = 0
    fallback_prompt_count = 0
    
    # Samples successfully written as npz this run: (path, tags), tags = selected [] label list
    random_idx_entries = []
    
    for idx, item in enumerate(tqdm(data, desc="Extracting T5 features")):
        save_path = os.path.join(
            output_dir,
            Path(item['path']).stem + '.npz'
        )
        
        if os.path.exists(save_path):
            skipped += 1
            continue
        
        try:
            path = item['path']
            caption_source = prompt_type
            if use_long_prompt:
                long_prompt = item.get('long_prompt', None)
                if long_prompt:
                    long_prompt_count += 1
                    processed_prompt, tags = process_long_prompt_segmented(
                        long_prompt, path
                    )
                    random_idx_entries.append((path, tags))
                    if processed_prompt:
                        caption = processed_prompt.strip()
                        processed_prompt_count += 1
                        caption_source = "long_prompt_processed"
                    else:
                        caption = item['prompt'].strip()
                        fallback_prompt_count += 1
                        caption_source = "long_prompt_fallback_to_prompt"
                else:
                    caption = item['prompt'].strip()
                    fallback_prompt_count += 1
                    random_idx_entries.append((path, []))
                    caption_source = "long_prompt_missing_fallback_to_prompt"
            else:
                caption = item['prompt'].strip()
                fallback_prompt_count += 1
                caption_source = "prompt"
            
            if idx % 10 == 0:
                caption_preview = (caption or "").replace("\n", " ")
                caption_preview = caption_preview[:120]
                print(f"[Step2-debug] idx={idx} item_path={path} caption_source={caption_source} caption_preview={caption_preview} tags={tags}")

            caption_emb, emb_mask = t5.get_text_embeddings([caption])
            
            emb_dict = {
                'caption_feature': caption_emb.float().cpu().data.numpy(),
                'attention_mask': emb_mask.cpu().data.numpy(),
            }
            np.savez_compressed(save_path, **emb_dict)
            processed += 1
        except Exception as e:
            print(f"Error processing {item['path']}: {e}")
            errors += 1
    
    print(f"✓ T5 features: {processed} processed, {skipped} skipped, {errors} errors")
    if use_long_prompt:
        print(f"  Long prompt usage: {long_prompt_count} items had long_prompt")
        print(f"  Processed prompt: {processed_prompt_count} items used processed long_prompt")
        print(f"  Fallback to prompt: {fallback_prompt_count} items used original prompt")
    print(f"  Saved to: {output_dir}")
    
    if not use_long_prompt:
        return

    random_idx_output_file = os.path.join(output_dir, "random_idx_list.txt")
    try:
        import fcntl
        use_lock = True
    except ImportError:
        use_lock = False
    
    existing_entries = set()
    if os.path.exists(random_idx_output_file):
        try:
            with open(random_idx_output_file, 'r') as f:
                if use_lock:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for line in f:
                        if ':' in line:
                            item_path = line.split(':', 1)[0].strip()
                            existing_entries.add(item_path)
                finally:
                    if use_lock:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"  Warning: Could not read existing random_idx_list.txt: {e}")
    
    with open(random_idx_output_file, 'a') as f:
        if use_lock:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            new_count = 0
            for path, tags in random_idx_entries:
                if path not in existing_entries:
                    f.write(format_random_list_line(path, tags) + "\n")
                    existing_entries.add(path)
                    new_count += 1
            f.flush()
            if new_count > 0:
                print(f"  Added {new_count} new entries to random_idx_list.txt (bracket tags, not numeric idx)")
            else:
                print(f"  All entries already exist in random_idx_list.txt")
        finally:
            if use_lock:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    print(f"  Random selection list saved to: {random_idx_output_file}")

def step3_extract_edges_features(
    metadata_path,
    image_dir,
    output_dir,
    vae_path,
    resolution=1024,
    device='cuda',
    preserve_aspect_ratio=False,
    gaussian_blur_radius=0,
    use_sketch_blurred=False,
):
    """Step 3: Encode control images (edges / sketch) directly with VAE into condition latents.

    Args:
        preserve_aspect_ratio: True to keep aspect ratio and pad; False to resize+crop
        gaussian_blur_radius: Gaussian blur radius (0=off; even values are auto-bumped +1)
        use_sketch_blurred: True reads control_image2, otherwise control_image
    """
    print("=" * 50)
    print("Step 3: Encoding control images with VAE (direct edges)...")
    if gaussian_blur_radius > 0:
        blur_radius = gaussian_blur_radius if gaussian_blur_radius % 2 == 1 else gaussian_blur_radius + 1
        print(f"  Applying Gaussian blur with radius: {blur_radius} (requested: {gaussian_blur_radius})")

    if not torch.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA not available, using CPU (will be very slow)")

    print(f"Loading VAE from {vae_path}...")
    try:
        vae_model = AutoencoderKL.from_pretrained(vae_path, use_safetensors=False)
        vae_model = vae_model.to(device)
        vae_model.eval()
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    if preserve_aspect_ratio:
        def resize_pad(img):
            img = img.convert('RGB')
            w, h = img.size
            scale = resolution / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            try:
                bicubic = Image.Resampling.BICUBIC
            except AttributeError:
                bicubic = Image.BICUBIC
            img = img.resize((new_w, new_h), bicubic)
            canvas = Image.new('RGB', (resolution, resolution), (0, 0, 0))
            offset_x = (resolution - new_w) // 2
            offset_y = (resolution - new_h) // 2
            canvas.paste(img, (offset_x, offset_y))
            return canvas

        transform = T.Compose([
            T.Lambda(resize_pad),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
        ])

    os.makedirs(output_dir, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0

    for idx, item in enumerate(tqdm(data, desc="Extracting edges features")):
        path = item['path']
        if path.startswith('celebA/'):
            path = path[7:]

        if use_sketch_blurred:
            control_path = item.get('control_image2') or item.get('control_image')
        else:
            control_path = item.get('control_image')

        if control_path:
            control_img_path = os.path.join(image_dir, control_path)
        else:
            control_path = path.replace('_original.png', '_edges.jpg').replace('_original.jpg', '_edges.jpg')
            control_img_path = os.path.join(image_dir, control_path)
            if not os.path.exists(control_img_path):
                control_path = path.replace('_original.png', '_edges.png').replace('_original.jpg', '_edges.png')
                control_img_path = os.path.join(image_dir, control_path)

        if idx % 10 == 0:
            print(f"[Step3-debug] idx={idx} item_path={item.get('path','')} path={path} control_img_path={control_img_path}")

        if not os.path.exists(control_img_path):
            if control_img_path.endswith('.jpg'):
                control_img_path_alt = control_img_path.replace('.jpg', '.png')
                if os.path.exists(control_img_path_alt):
                    control_img_path = control_img_path_alt
            elif control_img_path.endswith('.png'):
                control_img_path_alt = control_img_path.replace('.png', '.jpg')
                if os.path.exists(control_img_path_alt):
                    control_img_path = control_img_path_alt

        if not os.path.exists(control_img_path):
            errors += 1
            print(f"Warning: Control image not found: {control_img_path}")
            continue

        save_path = os.path.join(output_dir, Path(path).stem + '.npz')
        if os.path.exists(save_path):
            skipped += 1
            continue

        try:
            img = Image.open(control_img_path)

            if gaussian_blur_radius > 0:
                img_np = np.array(img)
                blur_radius = gaussian_blur_radius if gaussian_blur_radius % 2 == 1 else gaussian_blur_radius + 1
                img_blurred = cv2.GaussianBlur(img_np, (blur_radius, blur_radius), 0)
                img = Image.fromarray(img_blurred)

            img_tensor = transform(img).unsqueeze(0).to(device)
            img_normalized = img_tensor * 2.0 - 1.0

            with torch.no_grad():
                posterior = vae_model.encode(img_normalized).latent_dist
                condition_feat = torch.cat([posterior.mean, posterior.std], dim=1)
                condition_feat = condition_feat.cpu().numpy()
                if condition_feat.ndim == 4 and condition_feat.shape[0] == 1:
                    condition_feat = condition_feat.squeeze(0)

            np.savez_compressed(save_path, arr_0=condition_feat)
            processed += 1
        except Exception as e:
            print(f"Error processing {control_img_path}: {e}")
            errors += 1

    print(f"✓ edges features: {processed} processed, {skipped} skipped, {errors} errors")
    print(f"  Saved to: {output_dir}")

def main():
    """Main entry: run all preprocessing steps."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess dataset for ControlNet training')
    parser.add_argument('--dataset_root', type=str, default='dataset_train',
                        help='Root directory of the dataset')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing pretrained models')
    parser.add_argument('--resolution', type=int, default=1024,
                        help='Image resolution (512 or 1024)')
    parser.add_argument('--preserve_aspect_ratio', action='store_true',
                        help='Preserve aspect ratio by padding instead of cropping')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start index for processing (for batch processing)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='End index for processing (for batch processing)')
    parser.add_argument('--start', type=int, default=None,
                        help='Start index for processing (alias for --start_index, e.g., --start 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='End index for processing (alias for --end_index, e.g., --end 1000 processes 0-999)')
    parser.add_argument('--skip_steps', type=str, nargs='+', default=[],
                        choices=['1', '2', '3', '4', '5'],
                        help='Skip steps: 1=GT VAE, 2=T5, 3=edges VAE, 4=LQ VAE, 5=hard LQ')
    parser.add_argument('--use_long_prompt', action='store_true',
                        help='Use long_prompt instead of prompt for T5 feature extraction')
    parser.add_argument('--long_prompt_output_dir', type=str, default=None,
                        help='Output directory for long_prompt features (default: prompt_feature_long)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Image directory (default: dataset_root/celebA, or dataset_root if metadata paths include subdirectories)')
    parser.add_argument('--input_metadata', type=str, default=None,
                        help='Input metadata JSON file (already in final format, default: dataset_root/InternData/partition_filter/data_info.json)')
    parser.add_argument('--gaussian_blur_radius', type=int, default=0,
                        help='Gaussian blur radius for control images (0=disabled, must be odd number, default: 0)')
    parser.add_argument('--use_sketch_blurred', action='store_true',
                        help='Use control_image2 from metadata and save features to sketch_feature_1024 folder')
    # LQ features (Step 4): same as inference in-mask degradation — downsample then upsample to original size, then VAE encode
    parser.add_argument(
        '--lq_low_res',
        type=str,
        default='8',
        help='Step4 LQ target low resolution (same as inference --mask_face_degrade_low_res, e.g. 8, 64, 128x96)',
    )
    parser.add_argument(
        '--lq_downsample_method',
        type=str,
        default='bicubic',
        choices=['nearest', 'bilinear', 'bicubic', 'lanczos', 'box'],
        help='Step4 downsample interpolation method',
    )
    parser.add_argument(
        '--lq_upsample_method',
        type=str,
        default='bicubic',
        choices=['nearest', 'bilinear', 'bicubic', 'lanczos', 'box'],
        help='Step4 upsample-back-to-original-resolution interpolation method',
    )
    parser.add_argument(
        '--hard_lq_camera_target_size',
        type=int,
        default=16,
        choices=[8, 12, 16, 24, 32, 64],
        help='Native LQ side length from camera pipeline (same as camera_lq_pipeline --target-size)',
    )
    parser.add_argument('--hard_lq_seed', type=int, default=20260409, help='hard_LQ RNG seed')
    parser.add_argument(
        '--hard_lq_strength',
        type=float,
        default=1.0,
        help='hard_LQ global degradation strength (passed to camera_lq_pipeline sample_params)',
    )
    parser.add_argument(
        '--hard_lq_isp_gamma',
        type=float,
        default=0.92,
        help='hard_LQ ISP gamma (camera_lq_pipeline)',
    )
    parser.add_argument(
        '--hard_lq_no_surveillance_grade',
        action='store_true',
        help='Disable camera_lq_pipeline surveillance-style post-processing',
    )

    args = parser.parse_args()
    
    # If --start or --end is set, override --start_index / --end_index
    if args.start is not None:
        args.start_index = args.start
    if args.end is not None:
        args.end_index = args.end
    
    # Configure paths
    dataset_root = args.dataset_root
    models_dir = args.models_dir
    resolution = args.resolution
    
    # Path configuration
    input_metadata = args.input_metadata or os.path.join(
        dataset_root, "InternData/partition_filter/data_info.json"
    )
    
    # If image_dir is set, use it; otherwise default to dataset_root (backward compatible with celebA layout)
    if args.image_dir:
        image_dir = os.path.abspath(args.image_dir)
    else:
        # Default to dataset_root because metadata paths may already include subdirs (e.g. FFHQ-1024/...)
        image_dir = os.path.abspath(dataset_root)
    
    vae_path = os.path.join(models_dir, "sd-vae-ft-ema")
    # T5 model path: check whether t5-v1_1-xxl exists
    t5_model_path = os.path.join(models_dir, "t5-v1_1-xxl")
    if not os.path.exists(t5_model_path):
        # Try t5_ckpts/t5-v1_1-xxl
        t5_model_path = os.path.join(models_dir, "t5_ckpts", "t5-v1_1-xxl")
    # Convert to absolute path
    t5_model_path = os.path.abspath(t5_model_path) if os.path.exists(t5_model_path) else None
    t5_cache_dir = os.path.abspath(models_dir)  # cache_dir should be the parent directory 
    # Output directories
    vae_output = os.path.join(dataset_root, f"InternData/img_vae_features_{resolution}/noflip")
    # Choose T5 output dir based on whether long_prompt is used
    if args.use_long_prompt:
        if args.long_prompt_output_dir:
            t5_output = args.long_prompt_output_dir
        else:
            t5_output = os.path.join(dataset_root, "InternData/prompt_feature_long")
        print(f"Using long_prompt for T5 feature extraction")
        print(f"Output directory: {t5_output}")
    else:
        t5_output = os.path.join(dataset_root, "InternData/prompt_feature")
        print(f"Using original prompt for T5 feature extraction")
        print(f"Output directory: {t5_output}")
    # Choose edges/sketch output dir based on sketch_blurred mode
    if args.use_sketch_blurred:
        hed_output = os.path.join(dataset_root, f"InternData/sketch_feature_{resolution}")
        print(f"Using sketch_blurred mode: output directory will be {hed_output}")
    else:
        hed_output = os.path.join(dataset_root, f"InternData/edges_feature_{resolution}")
        print(f"Step 3: encoding control_image with VAE → {hed_output}")

    lq_feature_output = os.path.join(dataset_root, f"InternData/lq_feature_{resolution}")
    s_hlq = int(args.hard_lq_camera_target_size)
    hard_lq_vae_output = os.path.join(
        dataset_root, f"InternData/hard_lq_vae_features_{resolution}"
    )
    hard_lq_native_output = os.path.join(
        dataset_root, f"InternData/hard_lq_pixel_{s_hlq}x{s_hlq}"
    )
    print(f"LQ VAE features (step 4) will be saved to: {lq_feature_output}")
    print(f"Step 5 hard LQ: native PNG → {hard_lq_native_output}")
    print(f"Step 5 hard LQ: VAE .npz → {hard_lq_vae_output}")
    
    # Check input files
    if not os.path.exists(input_metadata):
        print(f"Error: {input_metadata} not found")
        return
    
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} not found")
        return
    
    # If an index range is set, slice the input metadata first (batch mode)
    if args.start_index > 0 or args.end_index is not None:
        print(f"Batch processing mode: start_index={args.start_index}, end_index={args.end_index}")
        with open(input_metadata, 'r') as f:
            input_data = json.load(f)
        end_idx = args.end_index if args.end_index is not None else len(input_data)
        input_data = input_data[args.start_index:end_idx]
        temp_input_metadata = input_metadata.replace('.json', f'_batch_{args.start_index}_{end_idx}.json')
        with open(temp_input_metadata, 'w') as f:
            json.dump(input_data, f, indent=2)
        input_metadata = temp_input_metadata
        print(f"  Created batch input metadata: {input_metadata} ({len(input_data)} items)")

    # Metadata is already in final format by default; use it for subsequent steps
    metadata_path = input_metadata
    
    if '1' not in args.skip_steps:
        if not os.path.exists(vae_path):
            print(f"Warning: VAE path {vae_path} not found, skipping VAE feature extraction")
        else:
            step1_extract_vae_features(metadata_path, image_dir, vae_output, vae_path, resolution, 
                                      preserve_aspect_ratio=args.preserve_aspect_ratio)
    
    if '2' not in args.skip_steps:
        step2_extract_t5_features(
            metadata_path,
            t5_output,
            t5_model_path,
            t5_cache_dir,
            use_long_prompt=args.use_long_prompt,
        )
    
    if '3' not in args.skip_steps:
        if not os.path.exists(vae_path):
            print(f"Warning: VAE path {vae_path} not found, skipping condition feature extraction")
        else:
            step3_extract_edges_features(
                metadata_path,
                image_dir,
                hed_output,
                vae_path,
                resolution,
                preserve_aspect_ratio=args.preserve_aspect_ratio,
                gaussian_blur_radius=args.gaussian_blur_radius,
                use_sketch_blurred=args.use_sketch_blurred,
            )

    if '4' not in args.skip_steps:
        if not os.path.exists(vae_path):
            print(f"Warning: VAE path {vae_path} not found, skipping LQ feature extraction (step 4)")
        else:
            step4_extract_lq_vae_features(
                metadata_path,
                image_dir,
                lq_feature_output,
                vae_path,
                resolution,
                preserve_aspect_ratio=args.preserve_aspect_ratio,
                lq_low_res=args.lq_low_res,
                lq_downsample_method=args.lq_downsample_method,
                lq_upsample_method=args.lq_upsample_method,
            )

    if "5" not in args.skip_steps:
        if not os.path.exists(vae_path):
            print(f"Warning: VAE path {vae_path} not found, skipping hard LQ (step 5)")
        else:
            step5_hard_lq_generation(
                metadata_path,
                image_dir,
                hard_lq_vae_output,
                hard_lq_native_output,
                vae_path,
                resolution=resolution,
                preserve_aspect_ratio=args.preserve_aspect_ratio,
                camera_target_size=s_hlq,
                hard_lq_seed=args.hard_lq_seed,
                hard_lq_strength=args.hard_lq_strength,
                hard_lq_isp_gamma=args.hard_lq_isp_gamma,
                hard_lq_surveillance_grade=not args.hard_lq_no_surveillance_grade,
            )

    print("=" * 50)
    print("✓ All preprocessing steps completed!")
    print(f"\nDataset structure:")
    print(f"  metadata used: {metadata_path}")
    print(f"  {dataset_root}/InternData/prompt_feature/")
    print(f"  {dataset_root}/InternData/img_vae_features_{resolution}/noflip/")
    # Show output dirs based on the mode actually used
    if args.use_sketch_blurred:
        print(f"  {dataset_root}/InternData/sketch_feature_{resolution}/")
    else:
        print(f"  {dataset_root}/InternData/edges_feature_{resolution}/")
    if '4' not in args.skip_steps:
        print(f"  {dataset_root}/InternData/lq_feature_{resolution}/")
    if "5" not in args.skip_steps:
        print(f"  {hard_lq_native_output}/")
        print(f"  {hard_lq_vae_output}/")

if __name__ == "__main__":
    main()
