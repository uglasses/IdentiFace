#!/usr/bin/env python3
"""
ControlNet inference script
Generate images using trained ControlNet weights
"""

import argparse
import copy
import glob
import os
import sys
import json
import hashlib
import tempfile
import runpy
import re
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import numpy as np


def str_to_bool(value):
    """Convert a string to a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)

# Add project paths
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "PixArt-alpha"))

from diffusion import DPMS
from diffusion.model.nets import PixArtMS_XL_2, ControlPixArtMSHalf
from diffusion.model.t5 import T5Embedder
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.data.datasets import ASPECT_RATIO_1024_TEST
from diffusers.models import AutoencoderKL
from tools.download import find_model
from SegFace.get_mask import generate_mask
from long_prompt_segmentation import parse_long_prompt_bracket_segments, compose_two_part_prompt
import cv2  # Must import cv2 before AdaFace
from AdaFace import (
    AdaFace,
    _get_or_extract_candidate_embeddings,
    _gallery_relpath_key,
    _search_topk_with_faiss,
    load_identity_map as load_identity_map_adaface,
    get_filename_from_path,
    _parse_low_res,
    _get_pil_resample,
)
vae_scale = 0.18215


def resolve_control_modality(use_freq: bool) -> str:
    """FreqFusion checkpoint returns 'fusion' (control_image→c + GT degrade→c2); otherwise edges."""
    if use_freq:
        print("[INFO] Fusion model: c=control_image, c2=LQ from GT")
        return "fusion"
    print("[INFO] Control modality: edges")
    return "edges"


def resolve_reference_from_item(item, dataset_base_path: str):
    """data_info: GT relative path comes from path."""
    reference_path = None
    reference_image_path = None
    if item.get("path"):
        reference_path = item["path"]
        reference_image_path = os.path.join(dataset_base_path, reference_path)
    if reference_image_path and not os.path.exists(reference_image_path):
        print(f"[WARNING] Reference image not found: {reference_image_path}")
        reference_path = None
        reference_image_path = None
    return reference_path, reference_image_path


def resolve_paths_from_data_item(
    item,
    dataset_base_path: str,
    entry_label: str = "",
):
    """
    Resolve paths from data_info only.
    Fusion and edges: c always uses control_image; for fusion, c2 is obtained by degrading GT inside generate_image.
    Returns (condition_image_path, reference_path, reference_image_path, err_msg); skip when err_msg is non-empty.
    """
    reference_path, reference_image_path = resolve_reference_from_item(item, dataset_base_path)

    ci = item.get("control_image")
    if not ci:
        return (
            None,
            reference_path,
            reference_image_path,
            f"control_image missing in data_info{entry_label}",
        )
    cip = os.path.join(dataset_base_path, ci)
    if not os.path.exists(cip):
        return (
            None,
            reference_path,
            reference_image_path,
            f"control_image not found: {cip}",
        )
    return cip, reference_path, reference_image_path, ""


def decompose_long_prompt_with_tags(prompt: str):
    """
    Decompose a new-format long_prompt into per-iteration prompts and return tags for each round.
    New format: first sentence + multiple [tags]segment.
    """
    first_sentence, segments = parse_long_prompt_bracket_segments(prompt)
    if segments:
        prompts = []
        tag_lists = []
        for seg in segments:
            seg_text = str(seg.get("text", "")).strip()
            if not seg_text:
                continue
            prompts.append(compose_two_part_prompt(first_sentence, seg))
            tag_lists.append([str(t) for t in seg.get("tags", [])])
        if prompts:
            return prompts, tag_lists

    # Fallback: compatibility with old prompts — split by sentence only
    splitted_prompts = prompt.split(".")
    header = splitted_prompts[0].strip()
    with_header_prompts = []
    if len(splitted_prompts) > 1 and header:
        for sentence in splitted_prompts[1:len(splitted_prompts) - 1]:
            s = sentence.strip()
            if s:
                with_header_prompts.append(f"{header}. {s}.")
    if with_header_prompts:
        return with_header_prompts, [[] for _ in with_header_prompts]
    return [prompt], [[]]


def _normalize_face_tag(tag: str) -> str:
    """Normalize tag: lowercase and strip non-alphanumeric/underscore characters."""
    return re.sub(r"[^a-z0-9_]", "", str(tag).strip().lower())


def _sort_prompt_tag_pairs_by_part_priority(
    prompt_tag_pairs: List[Tuple[str, List[str]]]
) -> List[Tuple[str, List[str]]]:
    """
    Reorder iteration sequence by facial-part priority (stable sort):
    eyes -> brows -> mouth -> nose -> ears -> hair -> others
    """
    priority_groups = [
        {"overview"},
        {"l_eye", "r_eye"},
        {"l_brow", "r_brow"},
        {"u_lip","l_lip","mouth"},
        {"nose"},
        {"l_ear", "r_ear"},
        {"hair"},
    ]
    part_to_priority: Dict[str, int] = {}
    for idx, group in enumerate(priority_groups):
        for part in group:
            part_to_priority[part] = idx

    def pair_priority(pair: Tuple[str, List[str]]) -> int:
        _prompt, tags = pair
        if not tags:
            return len(priority_groups)
        norm_tags = [_normalize_face_tag(t) for t in tags if str(t).strip()]
        priorities = [part_to_priority[t] for t in norm_tags if t in part_to_priority]
        return min(priorities) if priorities else len(priority_groups)

    return sorted(prompt_tag_pairs, key=pair_priority)


def get_facial_mask_by_tags(tags, sample):
    """Generate a facial-part mask from tags; overview alone does not produce a mask."""
    tags_norm = [_normalize_face_tag(t) for t in (tags or []) if str(t).strip()]
    if not tags_norm:
        return None
    if "overview" in tags_norm:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        save_image(sample, tmp_path, nrow=1, normalize=True, value_range=(-1, 1))
        return generate_mask(tmp_path, target_parts=tags_norm, model=None)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _tensor_minus1_to_pil_rgb(t):
    """[1,3,H,W] in [-1,1] -> PIL RGB."""
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    arr = ((t + 1.0) * 0.5 * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode='RGB')


def _pil_rgb_to_tensor_minus1(pil, device, dtype):
    """PIL RGB -> [1,3,H,W] in [-1,1]."""
    arr = np.array(pil.convert('RGB')).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device=device, dtype=dtype)


def _cosine_similarity_np(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity for 1D numpy vectors."""
    v1 = np.asarray(vec1).reshape(-1)
    v2 = np.asarray(vec2).reshape(-1)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _load_sface_backbone(
    weights_path: str,
    device: torch.device,
    network_preference: str = "iresnet50",
    embedding_size: int = 512,
    use_se: bool = False,
) -> torch.nn.Module:
    """
    Load SFace backbone weights.

    The repo contains iresnet50/iresnet100; we try the preferred architecture first,
    then fall back to the other one if state dict loading fails.
    """
    # Ensure SFace's repo root + utils are in sys.path so that
    # `backbones.iresnet` and `align_trans` resolve.
    sface_root = str((current_dir / "SFace").resolve())
    utils_root = os.path.join(sface_root, "utils")
    if sface_root not in sys.path:
        sys.path.insert(0, sface_root)
    if utils_root not in sys.path and os.path.isdir(utils_root):
        sys.path.insert(0, utils_root)

    from backbones.iresnet import iresnet50, iresnet100

    def build(network: str) -> torch.nn.Module:
        if network == "iresnet50":
            return iresnet50(dropout=0.4, num_features=embedding_size, use_se=use_se)
        if network == "iresnet100":
            return iresnet100(num_features=embedding_size, use_se=use_se)
        raise ValueError(f"Unsupported SFace network: {network}")

    # Some checkpoints may have different key prefixes.
    ckpt: Any = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected SFace checkpoint format: {type(ckpt)}")

    state_dict: Dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        state_dict[nk] = v

    def try_load(network: str) -> Optional[torch.nn.Module]:
        model = build(network)
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[WARN] SFace missing keys ({network}): {missing[:5]}")
            if unexpected:
                print(f"[WARN] SFace unexpected keys ({network}): {unexpected[:5]}")
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"[WARN] Failed to load SFace backbone as {network}: {e}")
            return None

    model = try_load(network_preference)
    if model is not None:
        return model

    fallback = "iresnet100" if network_preference == "iresnet50" else "iresnet50"
    model = try_load(fallback)
    if model is None:
        raise RuntimeError("Failed to load SFace backbone weights with both iresnet50/iresnet100.")
    return model


@torch.no_grad()
def _extract_sface_feature_from_image_path(
    sface_model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    image_size: int = 112,
) -> np.ndarray:
    """Extract SFace embedding (L2-normalized) from an image path."""
    pil = Image.open(img_path).convert("RGB").resize(
        (image_size, image_size), resample=_get_pil_resample("bilinear")
    )
    x = _pil_rgb_to_tensor_minus1(pil, device=device, dtype=torch.float32)
    feat = sface_model(x)
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _sface_weights_fingerprint(weights_path: str) -> str:
    st = os.stat(weights_path)
    # Exclude mtime to avoid unnecessary cache misses when only file timestamp changes.
    return f"{os.path.realpath(weights_path)}|size={st.st_size}"


def _sface_load_embedding_cache(cache_file: str) -> Tuple[Dict[str, np.ndarray], bool]:
    if not cache_file or not os.path.exists(cache_file):
        return {}, False
    try:
        data = np.load(cache_file, allow_pickle=True)
        feats = data["feats"]
        paths = data["paths"].tolist()
        path_to_feat = {p: feats[i] for i, p in enumerate(paths)}
        return path_to_feat, True
    except Exception as e:
        print(f"[WARN] Failed to load SFace cache {cache_file}: {e}")
        return {}, False


def _sface_save_embedding_cache(cache_file: str, feats: np.ndarray, paths: List[str]) -> None:
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    np.savez_compressed(cache_file, feats=feats, paths=np.array(paths, dtype=object))


_GALLERY_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_GALLERY_SKIP_DIR_NAMES = frozenset({
    ".adaface_embedding_cache",
    ".adaface_cache",
    ".sface_embedding_cache",
})


def _list_gallery_image_paths(gallery_dir: str) -> List[str]:
    """Recursively list image files under gallery_dir (absolute real paths, sorted)."""
    gallery_dir = os.path.abspath(gallery_dir)
    if not os.path.isdir(gallery_dir):
        return []
    out: List[str] = []
    for root, dirs, files in os.walk(gallery_dir):
        dirs[:] = [d for d in dirs if d not in _GALLERY_SKIP_DIR_NAMES]
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _GALLERY_IMAGE_EXTS:
                continue
            out.append(os.path.realpath(os.path.join(root, fn)))
    out.sort()
    return out


def _build_sface_candidate_db(
    sface_model: torch.nn.Module,
    device: torch.device,
    gallery_dir: str,
    dataset_base_path: str,
    image_size: int,
    weights_fingerprint: str,
    sface_cache_file: str,
    skip_failed: bool = True,
) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, str]]]:
    """
    Build SFace candidate embedding DB from all images under `gallery_dir`.

    Returns:
      candidate_feats: (N, D) float32 embeddings
      candidate_paths_valid: list of absolute paths in same order as candidate_feats
      candidate_meta: absolute_path -> {relative_path: ...}
    """
    candidate_paths = _list_gallery_image_paths(gallery_dir)
    if not candidate_paths:
        raise ValueError(f"No SFace candidates found in: {gallery_dir} (expected image files)")

    path_to_feat_cached, loaded = _sface_load_embedding_cache(sface_cache_file)
    keys_need_rewrite = False
    if path_to_feat_cached:
        remapped = {}
        for p, feat in path_to_feat_cached.items():
            key = _gallery_relpath_key(str(p), gallery_dir)
            remapped[key] = feat
            if str(p).replace("\\", "/") != key:
                keys_need_rewrite = True
        path_to_feat_cached = remapped

    out_paths_valid: List[str] = []
    out_feats: List[np.ndarray] = []

    missing_paths: List[str] = []
    for p in candidate_paths:
        lookup = _gallery_relpath_key(p, gallery_dir)
        if lookup in path_to_feat_cached:
            out_paths_valid.append(p)
            out_feats.append(path_to_feat_cached[lookup])
        else:
            missing_paths.append(p)

    if missing_paths:
        print(f"[INFO] SFace extracting {len(missing_paths)} missing candidate embeddings...")
        for img_path in missing_paths:
            try:
                feat = _extract_sface_feature_from_image_path(
                    sface_model=sface_model,
                    img_path=img_path,
                    device=device,
                    image_size=image_size,
                )
                path_to_feat_cached[_gallery_relpath_key(img_path, gallery_dir)] = feat
                out_paths_valid.append(img_path)
                out_feats.append(feat)
            except Exception as e:
                if not skip_failed:
                    raise
                print(f"[WARN] Failed to extract SFace feature for {img_path}: {e}")

    if not out_feats:
        raise ValueError("No valid SFace candidate embeddings available after cache+extraction.")

    # Persist updated cache (merged); rewrite if keys were still absolute.
    merged_paths = list(path_to_feat_cached.keys())
    merged_feats = np.stack([path_to_feat_cached[p] for p in merged_paths], axis=0).astype(np.float32)
    if missing_paths or keys_need_rewrite or not loaded:
        _sface_save_embedding_cache(sface_cache_file, merged_feats, merged_paths)

    candidate_meta: Dict[str, Dict[str, str]] = {}
    for p in out_paths_valid:
        rel = os.path.relpath(p, dataset_base_path)
        candidate_meta[p] = {"relative_path": rel}

    return np.stack(out_feats, axis=0).astype(np.float32), out_paths_valid, candidate_meta


def degrade_image_resolution(
    gt_image_normalized,
    low_res,
    downsample_method,
    upsample_method,
):
    """
    Downsample then upsample back to the original size.
    gt_image_normalized: [1,3,H,W], value range [-1,1]
    low_res: e.g. "64" or "128x96"
    """
    _, _, H, W = gt_image_normalized.shape
    low_w, low_h = _parse_low_res(low_res)
    low_w = max(1, min(int(low_w), int(W)))
    low_h = max(1, min(int(low_h), int(H)))
    pil = _tensor_minus1_to_pil_rgb(gt_image_normalized)
    pil_low = pil.resize((low_w, low_h), resample=_get_pil_resample(downsample_method))
    pil_up = pil_low.resize((W, H), resample=_get_pil_resample(upsample_method))
    return _pil_rgb_to_tensor_minus1(pil_up, gt_image_normalized.device, gt_image_normalized.dtype)


_ALLOWED_HD_LQ_TARGET_SIZES = (8, 12, 16, 24, 32)
_camera_lq_pipeline_mod = None


def _get_camera_lq_pipeline_module():
    """Lazy import other_tools/camera_lq_pipeline.py (no __init__.py; do not use package import)."""
    global _camera_lq_pipeline_mod
    if _camera_lq_pipeline_mod is None:
        import importlib.util

        mod_path = Path(__file__).resolve().parent / "other_tools" / "camera_lq_pipeline.py"
        name = "camera_lq_pipeline_infer"
        spec = importlib.util.spec_from_file_location(name, mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load camera LQ pipeline from {mod_path}")
        _camera_lq_pipeline_mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = _camera_lq_pipeline_mod
        spec.loader.exec_module(_camera_lq_pipeline_mod)
    return _camera_lq_pipeline_mod


def _resolve_hd_lq_target_size(low_res: str) -> int:
    """Map lq_degrade_low_res (e.g. 8, 16, 128x96) to a square target_size allowed by the camera pipeline."""
    low_w, low_h = _parse_low_res(low_res)
    s = max(1, min(int(low_w), int(low_h)))
    if s in _ALLOWED_HD_LQ_TARGET_SIZES:
        return s
    return min(_ALLOWED_HD_LQ_TARGET_SIZES, key=lambda x: abs(x - s))


def degrade_image_via_hd_lq_camera_pipeline(
    gt_image_normalized,
    low_res: str,
    strength: float,
    isp_gamma: float,
    seed: int,
    *,
    surveillance_grade: bool = True,
):
    """
    other_tools/camera_lq_pipeline: sensor-domain degrade to a small LQ image, then bicubic upsample back to GT tensor (H, W).
    gt_image_normalized: [1, 3, H, W], value range [-1, 1].
    """
    _, _, height, width = gt_image_normalized.shape
    clp = _get_camera_lq_pipeline_module()
    target_size = _resolve_hd_lq_target_size(low_res)
    pil_gt = _tensor_minus1_to_pil_rgb(gt_image_normalized)
    rng = np.random.default_rng(int(seed))
    params = clp.sample_params(rng, target_size, float(strength))
    _oh, _sc, _sn, lq_small = clp.degrade_image(
        pil_gt,
        rng,
        params,
        float(isp_gamma),
        surveillance_grade=surveillance_grade,
        surveillance_strength=float(strength),
    )
    lq_up = lq_small.resize((int(width), int(height)), Image.Resampling.BICUBIC)
    return _pil_rgb_to_tensor_minus1(lq_up, gt_image_normalized.device, gt_image_normalized.dtype)


def lq_from_gt_with_optional_hd_pipeline(args, gt_image_normalized, log_prefix: str = ""):
    """
    Switch between LQ down-up and the HD camera pipeline based on args.use_hd_lq.
    log_prefix is used as a print prefix (e.g. single-lq / fusion-c2).
    """
    use_hd = bool(getattr(args, "use_hd_lq", False))
    if use_hd:
        ts = _resolve_hd_lq_target_size(args.lq_degrade_low_res)
        label = f"{log_prefix} " if log_prefix else ""
        sg = bool(getattr(args, "hd_lq_surveillance_grade", True))
        print(
            f"[INFO] {label}LQ via camera_lq_pipeline: target_size={ts}, "
            f"strength={getattr(args, 'hd_lq_strength', 1.0)}, isp_gamma={getattr(args, 'hd_lq_isp_gamma', 0.92)}, "
            f"surveillance_grade={sg}, "
            f"then bicubic to GT size {tuple(gt_image_normalized.shape[2:])}"
        )
        return degrade_image_via_hd_lq_camera_pipeline(
            gt_image_normalized,
            low_res=args.lq_degrade_low_res,
            strength=float(getattr(args, "hd_lq_strength", 1.0)),
            isp_gamma=float(getattr(args, "hd_lq_isp_gamma", 0.92)),
            seed=int(args.seed),
            surveillance_grade=sg,
        )
    return degrade_image_resolution(
        gt_image_normalized,
        low_res=args.lq_degrade_low_res,
        downsample_method=args.lq_downsample_method,
        upsample_method=args.lq_upsample_method,
    )


def load_identity_mapping(identity_file_path):
    """
    Load the identity mapping file.
    
    Args:
        identity_file_path: path to the identity file (e.g. dataset_train/CelebAMask-HQ/identity_CelebA-HQ.txt)
    
    Returns:
        dict: mapping of {filename number: identity id}, e.g. {3936: 3338, 15945: 3338}
    """
    try:
        if not os.path.exists(identity_file_path):
            print(
                f"[WARNING] Identity file not found: {identity_file_path} "
                f"(1:N identity alignment will fall back to path comparison; use --identity_file to specify the correct file)"
            )
            return {}
        identity_map = load_identity_map_adaface(identity_file_path)
        print(f"[INFO] Loaded {len(identity_map)} identity mappings from {identity_file_path}")
        return identity_map
    except Exception as e:
        print(f"[WARNING] Failed to load identity mapping: {e}")
        return {}


def extract_number_from_filename(filename):
    """
    Extract a number from a filename.
    
    Args:
        filename: filename, e.g. "15945.jpg" (CelebA-HQ-img) or "XM2VTS_242.jpg"
    
    Returns:
        int or None: extracted number, e.g. 3936 or 15945 or 242
    """
    import re
    basename = os.path.basename(filename)
    
    # Strategy 1: if underscore present, prefer digits after underscore (e.g. XM2VTS_242.jpg -> 242)
    if '_' in basename:
        # Extract digits after underscore (until a non-digit or file extension)
        match = re.search(r'_(\d+)', basename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
    
    # Strategy 2: extract the last digit sequence in the filename (before extension)
    # e.g. "file123.jpg" -> 123, "XM2VTS_242.jpg" -> 242
    # Strip extension first
    name_without_ext = os.path.splitext(basename)[0]
    # Extract the last digit sequence
    matches = list(re.finditer(r'(\d+)', name_without_ext))
    if matches:
        # Take the last matched digit sequence
        last_match = matches[-1]
        try:
            return int(last_match.group(1))
        except ValueError:
            pass
    
    # Strategy 3: if all above fail, extract the first digit sequence (backward compatible)
    match = re.search(r'(\d+)', basename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def reference_id_for_outputs(reference_image_path: Optional[str]) -> str:
    """Numeric id for output filenames: ref_img_{id}.jpg, one_shot_{id}.jpg, etc."""
    if not reference_image_path:
        return "0"
    n = extract_number_from_filename(reference_image_path)
    if n is not None:
        return str(n)
    stem = os.path.splitext(os.path.basename(reference_image_path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits if digits else "0"


def copy_reference_rgb_as_jpg(src_path: str, dst_jpg_path: str) -> None:
    d = os.path.dirname(os.path.abspath(dst_jpg_path))
    if d:
        os.makedirs(d, exist_ok=True)
    Image.open(src_path).convert("RGB").save(dst_jpg_path, "JPEG", quality=95)


def save_tensor_minus1_1_as_jpg(tensor: torch.Tensor, dst_jpg_path: str) -> None:
    d = os.path.dirname(os.path.abspath(dst_jpg_path))
    if d:
        os.makedirs(d, exist_ok=True)
    save_image(tensor, dst_jpg_path, nrow=1, normalize=True, value_range=(-1, 1))


def _json_ready(obj: Any) -> Any:
    """Convert nested dict/list values (e.g. numpy scalars) to JSON-serializable types."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def pick_best_match_iteration_index(
    similarity_scores: List[Any],
    best_matches: List[Any],
    num_samples: int,
) -> int:
    """Pick best_match image across iterations: among AdaFace Matches take highest similarity; if none, take global highest similarity."""
    n = num_samples
    match_indices = [
        i
        for i in range(n)
        if i < len(best_matches)
        and best_matches[i] is not None
        and best_matches[i].get("matches_reference_image") is True
    ]

    def sim_at(idx: int) -> float:
        if idx >= len(similarity_scores):
            return float("-inf")
        s = similarity_scores[idx]
        if s is None:
            return float("-inf")
        return float(s)

    if match_indices:
        return max(match_indices, key=sim_at)
    cand = [
        i
        for i in range(n)
        if i < len(similarity_scores) and similarity_scores[i] is not None
    ]
    if cand:
        return max(cand, key=lambda i: float(similarity_scores[i]))
    return 0


def is_celebA_dataset(reference_image_path, identity_file_path):
    """
    Check whether this is a CelebA dataset.
    
    Logic: primarily based on the reference image path; if the path contains CelebA or celebA, treat as CelebA.
    
    Args:
        reference_image_path: reference image path
        identity_file_path: identity file path (for logging only; does not affect the decision)
    
    Returns:
        bool: True if CelebA dataset, False otherwise
    """
    # Decide from reference image path (most reliable)
    if reference_image_path and ('CelebA' in reference_image_path or 'celebA' in reference_image_path):
        return True
    
    return False


def check_path_match(reference_image_path, best_match_image_path, dataset_base_path):
    """
    Check whether two images match by comparing paths.
    
    Args:
        reference_image_path: reference image path
        best_match_image_path: best-match image path (may be relative)
        dataset_base_path: dataset base path
    
    Returns:
        bool: True if match, False otherwise
    """
    ref_image_normalized = os.path.normpath(os.path.abspath(reference_image_path))
    if os.path.isabs(best_match_image_path):
        best_match_full_path = best_match_image_path
    else:
        best_match_full_path = os.path.join(dataset_base_path, best_match_image_path)
    best_match_normalized = os.path.normpath(os.path.abspath(best_match_full_path))
    
    matches_reference = (
        ref_image_normalized == best_match_normalized or
        os.path.basename(ref_image_normalized) == os.path.basename(best_match_normalized) or
        (os.path.exists(ref_image_normalized) and os.path.exists(best_match_full_path) and
         os.path.samefile(ref_image_normalized, best_match_full_path))
    )
    
    return matches_reference


def check_identity_match(reference_image_path, best_match_image_path, identity_map):
    """
    Check whether two images match by identity id.
    
    Args:
        reference_image_path: reference image path
        best_match_image_path: best-match image path (may be relative)
        identity_map: identity mapping dict
    
    Returns:
        bool or None: True if match, False if not, None if undetermined
    """
    if not identity_map:
        return None

    ref_key = get_filename_from_path(reference_image_path)
    match_key = get_filename_from_path(best_match_image_path)
    ref_identity = identity_map.get(ref_key)
    match_identity = identity_map.get(match_key)

    if ref_identity is None or match_identity is None:
        return None
    return ref_identity == match_identity


def get_adaface_gallery_cache_filepath(gallery_dir: str) -> str:
    """AdaFace gallery embedding cache: keep a single npz under gallery_dir/.adaface_embedding_cache/.

    If adaface_emb_*.npz already exists, return the first after sorting; otherwise return adaface_emb.npz as the new path.
    """
    cache_root = os.path.join(os.path.abspath(gallery_dir), ".adaface_embedding_cache")
    os.makedirs(cache_root, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(cache_root, "adaface_emb_*.npz")))
    if existing:
        return existing[0]
    return os.path.join(cache_root, "adaface_emb.npz")


def get_args():
    parser = argparse.ArgumentParser(description="ControlNet inference script")
    parser.add_argument('--model_path', type=str, 
                       default='output/controlnet_celebA_1024/checkpoints/epoch_10_step_36250.pth',
                       help='Path to trained ControlNet model')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Training output dir containing config.py/checkpoints; if omitted, defaults to parent of --model_path')
    parser.add_argument('--lq_degrade_low_res', type=str, default='8',
                       help='Target low resolution for automatic GT degrade (e.g. 8, 64, 128x96)')
    parser.add_argument('--lq_downsample_method', type=str, default='bicubic',
                       choices=['nearest', 'bilinear', 'bicubic', 'lanczos', 'box'],
                       help='Automatic GT degrade: downsample interpolation method')
    parser.add_argument('--lq_upsample_method', type=str, default='bicubic',
                       choices=['nearest', 'bilinear', 'bicubic', 'lanczos', 'box'],
                       help='Automatic GT degrade: upsample interpolation method')
    parser.add_argument(
        '--use_hd_lq',
        dest='use_hd_lq',
        type=str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help='If True, generate LQ via other_tools/camera_lq_pipeline (sensor simulation), then bicubic upsample to current GT tensor size; '
        'If False, use LQ downsample-then-upsample. Target side length from --lq_degrade_low_res, mapped to {8,12,16,24,32}.',
    )
    parser.add_argument(
        '--hd_lq_strength',
        dest='hd_lq_strength',
        type=float,
        default=1.0,
        help='Only when use_hd_lq: global degrade strength for camera_lq_pipeline (includes surveillance look strength scaling).',
    )
    parser.add_argument(
        '--hd_lq_isp_gamma',
        dest='hd_lq_isp_gamma',
        type=float,
        default=0.92,
        help='Only when use_hd_lq: weak ISP gamma after demosaicing.',
    )
    parser.add_argument(
        '--hd_lq_surveillance_grade',
        dest='hd_lq_surveillance_grade',
        type=str_to_bool,
        nargs='?',
        const=True,
        default=True,
        help='Only when use_hd_lq: after sensor+ISP, slightly reduce saturation and brightness on final LQ (PIL Color/Brightness) for a surveillance look.',
    )
    parser.add_argument('--output_path', type=str, default='output/inference_result.png',
                       help='Base output path; each metadata entry is written to its own subdirectory')
    parser.add_argument('--use_long_prompt', type=str_to_bool, nargs='?', const=True, default=True,
                       help='Use long_prompt instead of prompt (only for metadata_json batch inference); default True. Use --use_long_prompt False to disable')
    parser.add_argument(
        '--infer_range_start',
        type=int,
        default=0,
        help='Batch inference: metadata start index (inclusive); with infer_range_end forms half-open [start, end) for multi-process sharding. Default 0',
    )
    parser.add_argument(
        '--infer_range_end',
        type=int,
        default=None,
        help='Batch inference: metadata end index (exclusive); default None means through end of list.',
    )
    parser.add_argument('--num_sampling_steps', type=int, default=20,
                       help='Number of sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=4.5,
                       help='CFG guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Image resolution')
    parser.add_argument('--tokenizer_path', type=str,
                       default='models/sd-vae-ft-ema',
                       help='VAE model path')
    parser.add_argument('--t5_path', type=str,
                       default='models/t5-v1_1-xxl',
                       help='T5 model path')
    parser.add_argument('--sampling_algo', type=str, default='dpm-solver',
                       choices=['iddpm', 'dpm-solver', 'sa-solver'],
                       help='Sampling algorithm')
    parser.add_argument('--adaface_model_path', type=str,
                       default='models/adaface_ir101_webface12m.ckpt',
                       help='AdaFace model path')
    parser.add_argument('--metadata_json', type=str, required=True,
                       help='data_info JSON: condition and GT paths are all resolved from this file')
    parser.add_argument('--dataset_base_path', type=str, default='dataset_train',
                       help='Dataset base path (for resolving relative paths in metadata)')
    parser.add_argument('--identity_file', type=str,
                       default='dataset_new/CelebA/identity_celebA.txt',
                       help='Identity mapping file (each line: filename identity_id, keys match {id}.jpg from get_filename_from_path)')
    parser.add_argument('--adaface_top_k', type=int, default=1,
                       help='AdaFace 1:N retrieval top-k (default 1)')
    parser.add_argument('--adaface_faiss_index_type', type=str, default='flat',
                       choices=['ivf', 'flat'],
                       help='AdaFace retrieval index type: ivf or flat')
    parser.add_argument('--adaface_faiss_nlist', type=int, default=100,
                       help='AdaFace IVF nlist parameter')
    parser.add_argument('--adaface_faiss_nprobe', type=int, default=10,
                       help='AdaFace IVF nprobe parameter')
    parser.add_argument(
        '--adaface_gallery_dir',
        type=str,
        default=None,
        help='AdaFace 1:N candidate face directory (recursively scan images; embeddings cached under .adaface_embedding_cache/)',
    )

    # -------- SFace (CL backbone) similarity & 1:N retrieval --------
    parser.add_argument('--sface_model_path', type=str,
                       default='models/SFace_backbone.pth',
                       help='SFace CL backbone model path')
    parser.add_argument('--sface_network', type=str, default='iresnet50', choices=['iresnet50', 'iresnet100'],
                       help='SFace backbone architecture')
    parser.add_argument('--sface_image_size', type=int, default=112, help='SFace input resolution (usually 112)')
    parser.add_argument('--sface_top_k', type=int, default=1,
                       help='SFace 1:N retrieval top-k (default 1)')
    parser.add_argument('--sface_faiss_index_type', type=str, default='flat',
                       choices=['ivf', 'flat'],
                       help='SFace retrieval index type: ivf or flat')
    parser.add_argument('--sface_faiss_nlist', type=int, default=100,
                       help='SFace IVF nlist parameter')
    parser.add_argument('--sface_faiss_nprobe', type=int, default=10,
                       help='SFace IVF nprobe parameter')
    parser.add_argument(
        '--sface_gallery_dir',
        type=str,
        default=None,
        help='SFace 1:N candidate face directory (recursively scan images; embeddings cached under .sface_embedding_cache/)',
    )
    # Inference-time architecture toggles (can be auto-filled from model_dir/config.py)
    parser.add_argument('--use_frequency_control_fusion', type=str_to_bool, nargs='?', const=True, default=None,
                       help='Enable FrequencyControlFusion (frequency-domain fusion + small_residual); default auto-inferred from model_dir/config.py')
    parser.add_argument('--copy_blocks_num', type=int, default=None,
                       help='Number of ControlNet copied blocks (must match training); default auto-inferred from model_dir/config.py')
    return parser.parse_args()


@torch.inference_mode()
def generate_image(
    prompt,
    condition_image_path,
    model,
    vae,
    t5,
    args,
    device,
    base_ratios,
    adaface=None,
    reference_feature=None,
    sface=None,
    sface_reference_feature=None,
):
    """Generate images."""
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    
    # Process prompts
    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(
        prompt, base_ratios, device=device
    )
    prompt_clean = prompt_clean.strip()
    prompts, prompt_tags = decompose_long_prompt_with_tags(prompt_clean)

    # Filter empty prompts and keep prompts aligned with tags
    prompt_tag_pairs = [
        (str(p).strip(), tags if isinstance(tags, list) else [])
        for p, tags in zip(prompts, prompt_tags)
        if p and str(p).strip()
    ]

    # Sort by facial-part priority, then T5-encode
    if prompt_tag_pairs:
        prompt_tag_pairs = _sort_prompt_tag_pairs_by_part_priority(prompt_tag_pairs)

    if not prompt_tag_pairs:
        print(f"[WARNING] prompt decomposition returned empty list, using original prompt")
        prompts = [prompt_clean]
        prompt_tags = [[]]
    else:
        prompts = [p for p, _ in prompt_tag_pairs]
        prompt_tags = [t for _, t in prompt_tag_pairs]
    
    print(f"[INFO] Prompt: {prompt_clean[:100]}...")
    print(f"[INFO] Decomposed into {len(prompts)} prompt(s)")
    
    # Get text embeddings
    caption_embs, emb_masks = t5.get_text_embeddings(prompts)
    caption_embs = caption_embs[:, None]
    null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]
    
    # Condition image: control_image → VAE → c; for fusion, also degrade GT to get c2
    c = None
    c_vis = None
    c2 = None
    c2_vis = None
    lq_input_vis = None
    gt_image_normalized = None  # Standalone GT ([-1, 1]) for mask and c2 degrade

    if condition_image_path is None or not os.path.exists(condition_image_path):
        raise ValueError(f"condition image is required: {condition_image_path}")

    print(f"[INFO] Loading condition image: {condition_image_path}")
    condition_img = Image.open(condition_image_path).convert('RGB').resize((1024, 1024))

    ar = torch.tensor([condition_img.size[1] / condition_img.size[0]], device=device)[None]
    custom_hw = torch.tensor([condition_img.size[1], condition_img.size[0]], device=device)[None]
    closest_hw = base_ratios[min(base_ratios.keys(), key=lambda ratio: abs(float(ratio) - ar.item()))]
    hw = torch.tensor(closest_hw, device=device)[None]

    condition_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(int(min(closest_hw))),
        T.CenterCrop([int(closest_hw[0]), int(closest_hw[1])]),
        T.ToTensor(),
    ])

    given_image = condition_transform(condition_img).unsqueeze(0).to(device)

    print("[INFO] Condition image: RGB → VAE encode (no HED)")
    given_image_normalized = given_image * 2.0 - 1.0
    posterior = vae.encode(given_image_normalized).latent_dist
    condition = posterior.sample()
    c = condition * vae_scale

    c_vis = vae.decode(condition)['sample']
    c_vis = torch.clamp(127.5 * c_vis + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    print(f"[INFO] Condition shape: {c.shape}")

    gt_image_path = getattr(args, 'reference_image', None)
    if gt_image_path is not None and os.path.exists(gt_image_path):
        print(f"[INFO] Loading GT image for masked DDIM init: {gt_image_path}")
        gt_img = Image.open(gt_image_path).convert('RGB')
        target_h, target_w = int(hw[0, 0].item()), int(hw[0, 1].item())
        gt_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(int(min(target_h, target_w))),
            T.CenterCrop([target_h, target_w]),
            T.ToTensor(),
        ])
        gt_tensor = gt_transform(gt_img).unsqueeze(0).to(device)
        gt_image_normalized = gt_tensor * 2.0 - 1.0  # [1, 3, H, W], value range [-1, 1]
    elif gt_image_path:
        print(f"[WARNING] GT image not found, skip masked DDIM init: {gt_image_path}")

    # Fusion: degrade GT to get c2 (HD camera pipeline or LQ down/up; chosen by --use_hd_lq)
    if c2 is None and gt_image_normalized is not None:
        try:
            if bool(getattr(args, "use_hd_lq", False)):
                print("[INFO] Auto-generating low-quality condition from GT (HD camera pipeline).")
            else:
                print(
                    "[INFO] Auto-generating low-quality condition from GT: "
                    f"low_res={args.lq_degrade_low_res}, down={args.lq_downsample_method}, up={args.lq_upsample_method}"
                )
            lq_from_gt = lq_from_gt_with_optional_hd_pipeline(
                args, gt_image_normalized, log_prefix="fusion auto c2"
            )
            posterior2 = vae.encode(lq_from_gt).latent_dist
            condition2 = posterior2.sample()
            c2 = condition2 * vae_scale
            lq_input_vis = torch.clamp(127.5 * lq_from_gt + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            c2_vis = torch.clamp(127.5 * lq_from_gt + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            print(f"[INFO] Auto-generated low-quality condition (c2) shape: {c2.shape}")
        except Exception as e:
            print(f"[WARNING] Failed to auto-generate low-quality condition from GT: {e}")
    
    # Compute latent spatial size
    latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
    print(f"[INFO] Latent size: {latent_size_h} x {latent_size_w}")
    
    # Sample / generate images
    print(f"[INFO] Sampling with {args.sampling_algo} ({args.num_sampling_steps} steps)...")
    samples = []
    similarity_scores = []
    best_matches = []
    sface_similarity_scores = []
    sface_best_matches = []
    early_stopped_both_matches = 0  # Iteration index (1-based) when early-stopped; 0 means not early-stopped
    if args.sampling_algo == 'dpm-solver':
        n = len(prompts)
        input_latents = None
        prev_adaface_similarity = None
        facial_masks = []

        # Main sampling loop: for i>0 use SegFace part mask; early-stop when both AdaFace and SFace match reference identity.
        had_adaface_ref_match = False
        had_sface_ref_match = False

        for i in range(n):
        
            z = torch.randn(1, 4, latent_size_h, latent_size_w, device=device)
            facial_mask = None

            # mask must have a batch dim; the model handles CFG repeat automatically
            mask_i = emb_masks[i]
            if mask_i.dim() == 1:
                mask_i = mask_i.unsqueeze(0)  # [seq_len] -> [1, seq_len]

            model_kwargs = dict(
                data_info={'img_hw': hw, 'aspect_ratio': ar},
                mask=mask_i,
                c=c
            )
            # FreqFusion: pass c2 (GT-degraded LQ)
            if getattr(args, 'use_frequency_control_fusion', False) and c2 is not None:
                model_kwargs['c2'] = c2
            dpm_solver = DPMS(
                model.forward_with_dpmsolver,
                condition=caption_embs[i],
                uncondition=null_y[i],
                cfg_scale=args.cfg_scale,
                model_kwargs=model_kwargs
            )
            # Normal first round, then per-iteration get_facial_mask
            if i == 0:
                latent_samples, candidate_input_latents = dpm_solver.sample(
                    z,
                    steps=args.num_sampling_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                    return_intermediate=True,
                )
            else:
                tags_for_iter = prompt_tags[i] if i < len(prompt_tags) else []
                facial_mask = get_facial_mask_by_tags(tags_for_iter, samples[i - 1])
                gt_facial_mask = None
                if gt_image_normalized is not None:
                    gt_facial_mask = get_facial_mask_by_tags(tags_for_iter, gt_image_normalized)

                if facial_mask is not None and gt_facial_mask is not None:
                    facial_mask = np.logical_or(facial_mask > 0, gt_facial_mask > 0).astype(np.uint8) * 255
                elif facial_mask is None and gt_facial_mask is not None:
                    facial_mask = gt_facial_mask
                if facial_mask is not None:
                    facial_mask_tensor = torch.tensor(facial_mask).float().to(device) / 255.0
                    if facial_mask_tensor.dim() == 2:
                        facial_mask_tensor = facial_mask_tensor.unsqueeze(0).unsqueeze(0)
                    mask_latent_weight = torch.nn.functional.interpolate(
                        facial_mask_tensor,
                        size=(latent_size_h, latent_size_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    mask_latent_weight = torch.clamp(mask_latent_weight, 0.0, 1.0)
                else:
                    mask_latent_weight = None

                latent_samples, candidate_input_latents = dpm_solver.sample(
                    z,
                    steps=args.num_sampling_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                    return_intermediate=True,
                    facial_mask=mask_latent_weight,
                    input_latents=input_latents,
                )
            facial_masks.append(facial_mask)
            print("[INFO] Decoding VAE latent...")
            sample = vae.decode(latent_samples / vae_scale).sample
            sample = resize_and_crop_tensor(sample, custom_hw[0, 1], custom_hw[0, 0])
            samples.append(sample)

            tmp_gen_path: Optional[str] = None
            if adaface is not None or sface is not None:
                fd, tmp_gen_path = tempfile.mkstemp(suffix=".png", prefix="infer_iter_")
                os.close(fd)
                save_image(sample, tmp_gen_path, nrow=1, normalize=True, value_range=(-1, 1))

            # Per-iteration AdaFace similarity (evaluate right after generation)
            current_adaface_similarity = None
            if adaface is not None:
                try:
                    print(f"[INFO] Computing AdaFace similarity for iteration {i}...")
                    generated_feature = adaface.extract_feature(tmp_gen_path)

                    best_match = None
                    candidate_feats = getattr(args, 'adaface_candidate_feats', None)
                    candidate_paths = getattr(args, 'adaface_candidate_paths', None)
                    candidate_meta = getattr(args, 'adaface_candidate_meta', {})
                    if candidate_feats is not None and candidate_paths is not None and len(candidate_paths) > 0:
                        print(f"[INFO] Retrieving from {len(candidate_paths)} candidate faces...")
                        top_k = max(1, int(getattr(args, 'adaface_top_k', 1)))
                        hits = _search_topk_with_faiss(
                            query_feat=generated_feature,
                            candidate_feats=candidate_feats,
                            candidate_paths=candidate_paths,
                            top_k=top_k,
                            index_type=getattr(args, 'adaface_faiss_index_type', 'ivf'),
                            nlist=int(getattr(args, 'adaface_faiss_nlist', 100)),
                            nprobe=int(getattr(args, 'adaface_faiss_nprobe', 10)),
                            use_gpu=bool(getattr(args, 'adaface_faiss_use_gpu', False)),
                            gpu_id=int(getattr(args, 'adaface_faiss_gpu_id', 0)),
                        )
                        if hits:
                            hit_path, hit_score = hits[0]
                            hit_meta = candidate_meta.get(hit_path, {})
                            display_path = hit_meta.get('relative_path', hit_path)
                            best_match = {
                                'image_path': display_path,
                                'absolute_image_path': hit_path,
                                '1_N_similarity_score': float(hit_score),
                                'prompt': hit_meta.get('prompt', ''),
                            }
                            print(
                                f"[INFO] Iteration {i} best match: {display_path} "
                                f"(1:N similarity: {float(hit_score):.6f})"
                            )
                        else:
                            print(f"[WARNING] Iteration {i} no best match found")

                    if reference_feature is not None:
                        similarity = adaface.cosine_similarity(reference_feature, generated_feature)
                        current_adaface_similarity = float(similarity)
                        similarity_scores.append(similarity)
                        print(f"[INFO] Iteration {i} similarity with reference_image: {similarity:.6f}")
                    else:
                        similarity_scores.append(None)

                    if best_match and getattr(args, 'reference_image', None):
                        best_match_compare_path = best_match.get('absolute_image_path', best_match['image_path'])
                        reference_image_path = args.reference_image
                        identity_file_path = getattr(args, 'identity_file', None)
                        is_celebA = is_celebA_dataset(reference_image_path, identity_file_path)
                        if is_celebA:
                            identity_map = getattr(args, 'identity_map', {})
                            matches_reference = check_identity_match(
                                reference_image_path,
                                best_match_compare_path,
                                identity_map
                            )
                            if matches_reference is None:
                                matches_reference = check_path_match(
                                    reference_image_path,
                                    best_match_compare_path,
                                    args.dataset_base_path
                                )
                        else:
                            matches_reference = check_path_match(
                                reference_image_path,
                                best_match_compare_path,
                                args.dataset_base_path
                            )
                        best_match['matches_reference_image'] = matches_reference
                    elif best_match:
                        best_match['matches_reference_image'] = None

                    best_matches.append(best_match if best_match else None)
                except Exception as e:
                    print(f"[WARNING] Failed to compute AdaFace similarity for iteration {i}: {e}")
                    similarity_scores.append(None)
                    best_matches.append(None)
            else:
                similarity_scores.append(None)
                best_matches.append(None)

            # Update next-round input_latents only if AdaFace similarity did not decrease this round
            if i == 0:
                input_latents = candidate_input_latents
            else:
                if (
                    current_adaface_similarity is not None
                    and prev_adaface_similarity is not None
                    and current_adaface_similarity >= prev_adaface_similarity
                ):
                    input_latents = candidate_input_latents
                    print(
                        f"[INFO] Iteration {i}: AdaFace similarity improved/kept "
                        f"({current_adaface_similarity:.6f} >= {prev_adaface_similarity:.6f}), update input_latents."
                    )
                else:
                    print(
                        f"[INFO] Iteration {i}: AdaFace similarity decreased or unavailable, keep previous input_latents."
                    )

            if current_adaface_similarity is not None:
                prev_adaface_similarity = current_adaface_similarity

            # Per-iteration SFace similarity (evaluate right after generation)
            if sface is not None:
                try:
                    print(f"[INFO] Computing SFace similarity for iteration {i}...")
                    generated_sface_feature = _extract_sface_feature_from_image_path(
                        sface_model=sface,
                        img_path=tmp_gen_path,
                        device=device,
                        image_size=getattr(args, 'sface_image_size', 112),
                    )

                    sface_best_match = None
                    sface_candidate_feats = getattr(args, 'sface_candidate_feats', None)
                    sface_candidate_paths = getattr(args, 'sface_candidate_paths', None)
                    sface_candidate_meta = getattr(args, 'sface_candidate_meta', {})
                    if (
                        sface_candidate_feats is not None
                        and sface_candidate_paths is not None
                        and len(sface_candidate_paths) > 0
                    ):
                        print(f"[INFO] SFace retrieving from {len(sface_candidate_paths)} candidate faces...")
                        top_k_sface = max(1, int(getattr(args, 'sface_top_k', 1)))
                        sface_hits = _search_topk_with_faiss(
                            query_feat=generated_sface_feature,
                            candidate_feats=sface_candidate_feats,
                            candidate_paths=sface_candidate_paths,
                            top_k=top_k_sface,
                            index_type=getattr(args, 'sface_faiss_index_type', 'ivf'),
                            nlist=int(getattr(args, 'sface_faiss_nlist', 100)),
                            nprobe=int(getattr(args, 'sface_faiss_nprobe', 10)),
                            use_gpu=bool(getattr(args, 'sface_faiss_use_gpu', False)),
                            gpu_id=int(getattr(args, 'sface_faiss_gpu_id', 0)),
                        )
                        if sface_hits:
                            hit_path, hit_score = sface_hits[0]
                            hit_meta = sface_candidate_meta.get(hit_path, {})
                            display_path = hit_meta.get('relative_path', hit_path)
                            sface_best_match = {
                                'image_path': display_path,
                                'absolute_image_path': hit_path,
                                '1_N_similarity_score': float(hit_score),
                                'prompt': hit_meta.get('prompt', ''),
                            }
                            print(
                                f"[INFO] Iteration {i} SFace best match: {display_path} "
                                f"(1:N similarity: {float(hit_score):.6f})"
                            )
                        else:
                            print(f"[WARNING] Iteration {i} SFace no best match found")

                    if sface_reference_feature is not None:
                        sface_similarity = _cosine_similarity_np(sface_reference_feature, generated_sface_feature)
                        sface_similarity_scores.append(sface_similarity)
                        print(f"[INFO] Iteration {i} SFace similarity with reference_image: {sface_similarity:.6f}")
                    else:
                        sface_similarity_scores.append(None)

                    if sface_best_match and getattr(args, 'reference_image', None):
                        sface_best_compare_path = sface_best_match.get(
                            'absolute_image_path',
                            sface_best_match.get('image_path'),
                        )
                        reference_image_path = args.reference_image
                        identity_file_path = getattr(args, 'identity_file', None)
                        is_celebA = is_celebA_dataset(reference_image_path, identity_file_path)
                        if is_celebA:
                            identity_map = getattr(args, 'identity_map', {})
                            matches_reference = check_identity_match(
                                reference_image_path,
                                sface_best_compare_path,
                                identity_map,
                            )
                            if matches_reference is None:
                                matches_reference = check_path_match(
                                    reference_image_path,
                                    sface_best_compare_path,
                                    args.dataset_base_path,
                                )
                        else:
                            matches_reference = check_path_match(
                                reference_image_path,
                                sface_best_compare_path,
                                args.dataset_base_path,
                            )
                        sface_best_match['matches_reference_image'] = matches_reference
                    elif sface_best_match:
                        sface_best_match['matches_reference_image'] = None

                    sface_best_matches.append(sface_best_match if sface_best_match else None)
                except Exception as e:
                    print(f"[WARNING] Failed to compute SFace similarity for iteration {i}: {e}")
                    sface_similarity_scores.append(None)
                    sface_best_matches.append(None)
            else:
                sface_similarity_scores.append(None)
                sface_best_matches.append(None)

            if tmp_gen_path and os.path.exists(tmp_gen_path):
                try:
                    os.unlink(tmp_gen_path)
                except OSError:
                    pass

            if best_matches and best_matches[-1] is not None:
                if best_matches[-1].get("matches_reference_image") is True:
                    had_adaface_ref_match = True
            if sface_best_matches and sface_best_matches[-1] is not None:
                if sface_best_matches[-1].get("matches_reference_image") is True:
                    had_sface_ref_match = True
            if had_adaface_ref_match and had_sface_ref_match:
                early_stopped_both_matches = i + 1
                print(
                    f"[INFO] Early stop at iteration {early_stopped_both_matches} (1-based): "
                    "both AdaFace and SFace gallery top-1 have matched reference in some iteration."
                )
                break

    else:
        raise NotImplementedError(f"Sampling algorithm {args.sampling_algo} not implemented in this script")
    similarity_data = {
        'similarity_scores': similarity_scores,
        'best_matches': best_matches,
        'sface_similarity_scores': sface_similarity_scores,
        'sface_best_matches': sface_best_matches,
        'early_stopped_both_matches': early_stopped_both_matches,
    }
    return samples, (c_vis, c2_vis, lq_input_vis), facial_masks, prompts, similarity_data


def process_single_inference(prompt, condition_image_path, reference_image_path, output_path, 
                             model, vae, t5, adaface, reference_feature,
                             sface, sface_reference_feature,
                             args, device, base_ratios):
    """
    Run a single inference: generate images and save results.
    
    Args:
        prompt: text prompt
        condition_image_path: absolute path to condition image (from data_info control_image)
        reference_image_path: reference image path (for similarity)
        output_path: output image path
        model, vae, t5: model objects
        adaface: AdaFace model (may be None)
        reference_feature: reference image feature (may be None)
        args: command-line arguments
        device: device
        base_ratios: base aspect ratios
    
    Returns:
        prompts_data: dict containing inference results
    """
    # Generate images
    print("\n" + "="*50)
    print(f"Starting generation for: {os.path.basename(output_path)}")
    print(f"Prompt: {prompt[:100]}...")
    print("="*50)
    
    # Create a temporary args object for parameter passing
    class TempArgs:
        def __init__(self, base_args):
            for key, value in vars(base_args).items():
                setattr(self, key, value)
            self.prompt = prompt
            self.condition_image = condition_image_path
            self.output_path = output_path
            self.reference_image = reference_image_path
    
    temp_args = TempArgs(args)
    
    samples, _cond_vis, facial_masks, prompts, similarity_data = generate_image(
        prompt,
        condition_image_path,
        model,
        vae,
        t5,
        temp_args,
        device,
        base_ratios,
        adaface=adaface,
        reference_feature=reference_feature,
        sface=sface,
        sface_reference_feature=sface_reference_feature,
    )
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ref_id = reference_id_for_outputs(reference_image_path)
    ref_jpg = os.path.join(output_dir, f"ref_img_{ref_id}.jpg")
    one_shot_jpg = os.path.join(output_dir, f"one_shot_{ref_id}.jpg")
    best_match_jpg = os.path.join(output_dir, f"best_match_{ref_id}.jpg")
    prompts_file = os.path.join(output_dir, "result_prompts.json")
    results_best_file = os.path.join(output_dir, "results_best.json")

    allowed_in_output_dir = {
        os.path.basename(ref_jpg),
        os.path.basename(one_shot_jpg),
        os.path.basename(best_match_jpg),
        os.path.basename(prompts_file),
        os.path.basename(results_best_file),
    }
    if output_dir and os.path.isdir(output_dir):
        for fn in os.listdir(output_dir):
            if fn not in allowed_in_output_dir:
                p = os.path.join(output_dir, fn)
                try:
                    if os.path.isfile(p):
                        os.unlink(p)
                except OSError:
                    pass

    similarity_scores = similarity_data.get('similarity_scores', [])
    best_matches = similarity_data.get('best_matches', [])
    sface_similarity_scores = similarity_data.get('sface_similarity_scores', [])
    sface_best_matches = similarity_data.get('sface_best_matches', [])

    if reference_image_path and os.path.exists(reference_image_path):
        try:
            copy_reference_rgb_as_jpg(reference_image_path, ref_jpg)
            print(f"[INFO] Reference (GT) saved to: {ref_jpg}")
        except Exception as e:
            print(f"[WARNING] Failed to save reference image as JPG: {e}")

    best_iter = 0
    if not samples:
        print("[WARNING] No samples generated; skipping image outputs")
    else:
        best_iter = pick_best_match_iteration_index(
            similarity_scores, best_matches, len(samples)
        )
        try:
            save_tensor_minus1_1_as_jpg(samples[0], one_shot_jpg)
            print(f"[INFO] First-iteration image saved to: {one_shot_jpg}")
        except Exception as e:
            print(f"[WARNING] Failed to save one_shot: {e}")
        try:
            save_tensor_minus1_1_as_jpg(samples[best_iter], best_match_jpg)
            print(
                f"[INFO] Best AdaFace-scored image (iter {best_iter}) saved to: {best_match_jpg}"
            )
        except Exception as e:
            print(f"[WARNING] Failed to save best_match: {e}")

    # Build prompt info: original prompt and per-iteration prompts
    ada_n = len(getattr(args, 'adaface_candidate_paths', None) or [])
    sface_n = len(getattr(args, 'sface_candidate_paths', None) or [])
    prompts_data = {
        'original_prompt': prompt,
        'reference_image': reference_image_path if reference_image_path else None,
        'metadata_json': args.metadata_json if args.metadata_json else None,
        'adaface_gallery_candidate_count': ada_n,
        'sface_gallery_candidate_count': sface_n,
        'early_stopped_both_matches': similarity_data.get('early_stopped_both_matches', 0),
        'iterations': []
    }
    
    # Note: prompts length should match samples; keep a fallback to avoid OOB in edge cases
    # Assign a prompt to each iteration
    for i in range(len(samples)):
        iteration_data = {
            'iteration': i,
            'prompt': prompts[i] if i < len(prompts) else (prompt if i == 0 else prompts[-1] if len(prompts) > 0 else prompt)
        }
        
        # Add similarity score (always, even if None)
        if i < len(similarity_scores):
            iteration_data['similarity_score'] = similarity_scores[i]
        else:
            iteration_data['similarity_score'] = None

        # Add SFace similarity score (always, even if None)
        if i < len(sface_similarity_scores):
            iteration_data['sface_similarity_score'] = sface_similarity_scores[i]
        else:
            iteration_data['sface_similarity_score'] = None
        
        # Add best-match info (1:N comparison result)
        if i < len(best_matches) and best_matches[i] is not None:
            iteration_data['best_match'] = best_matches[i]
            match_status = ""
            if best_matches[i].get('matches_reference_image') is True:
                match_status = " ✓ MATCHES REFERENCE"
            elif best_matches[i].get('matches_reference_image') is False:
                match_status = " ✗ DOES NOT MATCH REFERENCE"
            print(f"[INFO] Iteration {i} best match: {best_matches[i]['image_path']} (1:N similarity: {best_matches[i]['1_N_similarity_score']:.6f}){match_status}")

        # Add SFace best-match info (1:N comparison result)
        if i < len(sface_best_matches) and sface_best_matches[i] is not None:
            iteration_data['sface_best_match'] = sface_best_matches[i]
        
        prompts_data['iterations'].append(iteration_data)
    
    # Add average similarity (if every iteration has a score)
    if similarity_scores and all(s is not None for s in similarity_scores):
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        prompts_data['average_similarity'] = avg_similarity
        print(f"[INFO] Average similarity across all iterations: {avg_similarity:.6f}")

    if sface_similarity_scores and all(s is not None for s in sface_similarity_scores):
        avg_sface_similarity = sum(sface_similarity_scores) / len(sface_similarity_scores)
        prompts_data['average_sface_similarity'] = avg_sface_similarity
        print(f"[INFO] Average SFace similarity across all iterations: {avg_sface_similarity:.6f}")
    
    # Add best-match statistics (1:N comparison)
    if best_matches and any(m is not None for m in best_matches):
        best_match_similarities = [m['1_N_similarity_score'] for m in best_matches if m is not None]
        if best_match_similarities:
            prompts_data['average_1_N_similarity_score'] = sum(best_match_similarities) / len(best_match_similarities)
            prompts_data['max_1_N_similarity_score'] = max(best_match_similarities)
            prompts_data['min_1_N_similarity_score'] = min(best_match_similarities)
            print(f"[INFO] 1:N similarity statistics - Avg: {prompts_data['average_1_N_similarity_score']:.6f}, "
                  f"Max: {prompts_data['max_1_N_similarity_score']:.6f}, "
                  f"Min: {prompts_data['min_1_N_similarity_score']:.6f}")
        
        # Count matches against the reference image (if provided)
        if reference_image_path:
            match_count = sum(1 for m in best_matches if m is not None and m.get('matches_reference_image') is True)
            total_count = sum(1 for m in best_matches if m is not None)
            if total_count > 0:
                match_rate = match_count / total_count
                prompts_data['reference_image_match_count'] = match_count
                prompts_data['reference_image_match_rate'] = match_rate
                print(f"[INFO] Reference image match: {match_count}/{total_count} iterations ({match_rate*100:.2f}%)")
    
    results_best: List[Dict[str, Any]] = []
    if samples and prompts_data.get("iterations"):
        iters = prompts_data["iterations"]
        for iter_idx, typ in ((0, "one_shot"), (best_iter, "best_match")):
            if 0 <= iter_idx < len(iters):
                entry = copy.deepcopy(iters[iter_idx])
                entry["type"] = typ
                results_best.append(_json_ready(entry))
            else:
                results_best.append(_json_ready({"type": typ, "iteration": iter_idx}))

    with open(prompts_file, 'w', encoding='utf-8') as f:
        json.dump(_json_ready(prompts_data), f, indent=2, ensure_ascii=False)
    print(f"[INFO] Prompts and similarity scores saved to: {prompts_file}")

    with open(results_best_file, 'w', encoding='utf-8') as f:
        json.dump(results_best, f, indent=2, ensure_ascii=False)
    print(f"[INFO] results_best saved to: {results_best_file}")
    
    print(f"\n[INFO] Generation completed for: {os.path.basename(output_path)}")
    
    return prompts_data


def main():
    args = get_args()
    if not os.path.isfile(args.metadata_json):
        raise FileNotFoundError(f"--metadata_json not found: {args.metadata_json}")

    print(f"[INFO] use_long_prompt: {args.use_long_prompt}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # Model setup
    lewei_scale = {512: 1, 1024: 2}
    latent_size = args.image_size // 8
    weight_dtype = torch.float16
    print(f"[INFO] Inference with {weight_dtype}")
    
    # Initialize base model
    print("[INFO] Initializing base model...")
    base_model = PixArtMS_XL_2(
        input_size=latent_size,
        lewei_scale=lewei_scale[args.image_size]
    )
    
    # Wrap as ControlNet
    print("[INFO] Wrapping with ControlNet...")
    # Auto-infer architecture params from model_dir/config.py
    infer_model_dir = args.model_dir
    if infer_model_dir is None and args.model_path is not None:
        infer_model_dir = os.path.dirname(os.path.abspath(args.model_path))
    inferred_freq = None
    inferred_copy_blocks = None
    cfg_vars = {}
    if infer_model_dir:
        cfg_py = os.path.join(infer_model_dir, 'config.py')
        if os.path.exists(cfg_py):
            try:
                cfg_vars = runpy.run_path(cfg_py)
                inferred_freq = bool(cfg_vars.get('use_frequency_control_fusion', False))
                inferred_copy_blocks = int(cfg_vars.get('copy_blocks_num', 13))
                print(
                    f"[INFO] Loaded model config: "
                    f"use_frequency_control_fusion={inferred_freq}, copy_blocks_num={inferred_copy_blocks}"
                )
            except Exception as e:
                print(f"[WARNING] Failed to read model config.py: {e}")
    use_freq = inferred_freq if args.use_frequency_control_fusion is None else bool(args.use_frequency_control_fusion)
    copy_blocks_num = inferred_copy_blocks if args.copy_blocks_num is None else int(args.copy_blocks_num)
    model = ControlPixArtMSHalf(
        base_model,
        copy_blocks_num=copy_blocks_num if copy_blocks_num is not None else 13,
        use_frequency_control_fusion=bool(use_freq),
    ).to(device)
    print(
        f"[INFO] ControlNet init: copy_blocks_num={copy_blocks_num}, "
        f"use_frequency_control_fusion={bool(use_freq)}"
    )
    args.use_frequency_control_fusion = bool(use_freq)

    args._resolved_control_modality = resolve_control_modality(bool(use_freq))
    
    # Load trained weights
    print(f"[INFO] Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    state_dict = find_model(args.model_path)['state_dict']
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    elif 'base_model.pos_embed' in state_dict:
        del state_dict['base_model.pos_embed']
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Missing keys (missing pos_embed is normal): {len(missing)} keys")
    print(f"[INFO] Unexpected keys: {len(unexpected)} keys")

    model.eval()
    model.to(weight_dtype)
    
    # Load VAE
    print(f"[INFO] Loading VAE from: {args.tokenizer_path}")
    vae = AutoencoderKL.from_pretrained(
        args.tokenizer_path,
        use_safetensors=False
    ).to(device)
    
    # Load T5 text encoder
    print(f"[INFO] Loading T5 from: {args.t5_path}")
    t5 = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=os.path.dirname(args.t5_path),
        torch_dtype=torch.float
    )
    
    # Get base aspect ratios
    if args.image_size == 1024:
        base_ratios = ASPECT_RATIO_1024_TEST
    else:
        raise ValueError(f"Unsupported image size: {args.image_size}. Only 1024 is supported.")
    
    # Load identity mapping (to decide whether images match)
    print(f"[INFO] Loading identity mapping from: {args.identity_file}")
    identity_map = load_identity_mapping(args.identity_file)
    args.identity_map = identity_map  # Attach identity_map to args for later use
    
    # Init AdaFace (ref similarity + optional 1:N gallery from --adaface_gallery_dir)
    adaface = None
    args.adaface_candidate_feats = None
    args.adaface_candidate_paths = None
    args.adaface_candidate_meta = {}

    adaface_gallery_dir = getattr(args, "adaface_gallery_dir", None)
    init_adaface = os.path.exists(args.adaface_model_path)
    if init_adaface:
        print(f"[INFO] Loading AdaFace model from: {args.adaface_model_path}")
        try:
            adaface = AdaFace(args.adaface_model_path, device=device)
            print("[INFO] AdaFace model loaded successfully")

            if adaface_gallery_dir and os.path.isdir(os.path.abspath(adaface_gallery_dir)):
                gdir = os.path.abspath(adaface_gallery_dir)
                candidate_paths = _list_gallery_image_paths(gdir)
                if not candidate_paths:
                    print(f"[WARNING] AdaFace gallery_dir has no images: {gdir}")
                else:
                    cache_file_path = get_adaface_gallery_cache_filepath(gdir)
                    print(
                        f"[INFO] AdaFace candidate gallery: {gdir} ({len(candidate_paths)} images); "
                        f"cache: {cache_file_path}"
                    )
                    candidate_feats, candidate_paths_valid = _get_or_extract_candidate_embeddings(
                        adaface=adaface,
                        candidate_paths=candidate_paths,
                        cache_file=cache_file_path,
                        skip_failed=True,
                        gallery_dir=gdir,
                    )
                    args.adaface_candidate_feats = candidate_feats
                    args.adaface_candidate_paths = candidate_paths_valid
                    d0 = os.path.abspath(args.dataset_base_path)
                    meta: Dict[str, Dict[str, str]] = {}
                    for p in candidate_paths_valid:
                        ap = os.path.abspath(p)
                        try:
                            rel = os.path.relpath(ap, d0)
                            if rel.startswith(".."):
                                rel = os.path.relpath(ap, gdir)
                        except ValueError:
                            rel = os.path.relpath(ap, gdir)
                        meta[ap] = {
                            "relative_path": rel,
                        }
                    args.adaface_candidate_meta = meta
                    print(
                        f"[INFO] AdaFace candidate DB ready: "
                        f"{len(candidate_paths_valid)}/{len(candidate_paths)} valid faces"
                    )
            elif adaface_gallery_dir:
                print(
                    f"[WARNING] AdaFace --adaface_gallery_dir is not a directory: {adaface_gallery_dir}"
                )

        except FileNotFoundError as e:
            print(f"[ERROR] File not found: {e}")
            print("[WARNING] Continuing without similarity evaluation...")
            adaface = None
        except Exception as e:
            import traceback
            print("[ERROR] Failed to initialize AdaFace:")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            print(f"[ERROR] Exception message: {str(e)}")
            traceback.print_exc()
            print("[WARNING] Continuing without similarity evaluation...")
            adaface = None

    # Init SFace (generated vs GT + optional 1:N gallery from --sface_gallery_dir)
    sface = None
    args.sface_candidate_feats = None
    args.sface_candidate_paths = None
    args.sface_candidate_meta = {}

    if args.sface_model_path and os.path.exists(args.sface_model_path):
        print(f"[INFO] Loading SFace model from: {args.sface_model_path}")
        try:
            sface = _load_sface_backbone(
                weights_path=args.sface_model_path,
                device=device,
                network_preference=getattr(args, "sface_network", "iresnet50"),
                embedding_size=512,
                use_se=bool(getattr(args, "sface_use_se", False)),
            )
            print("[INFO] SFace model loaded successfully")

            sface_gallery_dir = getattr(args, "sface_gallery_dir", None)
            if sface_gallery_dir and os.path.isdir(os.path.abspath(sface_gallery_dir)):
                sgdir = os.path.abspath(sface_gallery_dir)
                n_img = len(_list_gallery_image_paths(sgdir))
                if n_img == 0:
                    print(f"[WARNING] SFace gallery_dir has no images: {sgdir}")
                else:
                    cache_dir = os.path.join(sgdir, ".sface_embedding_cache")
                    os.makedirs(cache_dir, exist_ok=True)
                    fingerprint = _sface_weights_fingerprint(args.sface_model_path)
                    cache_key = (
                        f"{fingerprint}|net={args.sface_network}"
                        f"|use_se={bool(getattr(args, 'sface_use_se', False))}"
                        f"|size={args.sface_image_size}"
                        f"|align=none"
                    )
                    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
                    sface_cache_file = os.path.join(cache_dir, f"sface_emb_{cache_hash}.npz")

                    print(
                        f"[INFO] SFace candidate gallery: {sgdir} ({n_img} images); "
                        f"cache: {sface_cache_file}"
                    )
                    candidate_feats, candidate_paths_valid, candidate_meta = _build_sface_candidate_db(
                        sface_model=sface,
                        device=device,
                        gallery_dir=sgdir,
                        dataset_base_path=args.dataset_base_path,
                        image_size=getattr(args, "sface_image_size", 112),
                        weights_fingerprint=fingerprint,
                        sface_cache_file=sface_cache_file,
                        skip_failed=True,
                    )
                    args.sface_candidate_feats = candidate_feats
                    args.sface_candidate_paths = candidate_paths_valid
                    args.sface_candidate_meta = candidate_meta
                    print(
                        f"[INFO] SFace candidate DB ready: {len(candidate_paths_valid)} valid candidates"
                    )
            elif sface_gallery_dir:
                print(
                    f"[WARNING] SFace --sface_gallery_dir is not a directory: {sface_gallery_dir}"
                )
        except Exception as e:
            import traceback
            print(f"[WARNING] Failed to initialize SFace: {e}")
            traceback.print_exc()
            sface = None
    else:
        print(f"[WARNING] SFace model not found: {args.sface_model_path}")
    
    print("\n" + "="*70)
    print("BATCH INFERENCE MODE")
    print("="*70)
    
    # Reload metadata JSON (for batch inference)
    with open(args.metadata_json, 'r', encoding='utf-8') as f:
        batch_metadata_full = json.load(f)

    total_entries = len(batch_metadata_full)
    print(f"[INFO] Found {total_entries} entries in metadata JSON")

    range_start = int(args.infer_range_start)
    if range_start < 0:
        raise ValueError("--infer_range_start must be >= 0")
    range_end_excl = args.infer_range_end
    if range_end_excl is not None and range_end_excl < range_start:
        raise ValueError("--infer_range_end must be >= infer_range_start (left-closed right-open interval)")
    if range_end_excl is None:
        range_end_excl = total_entries
    else:
        range_end_excl = int(range_end_excl)
    range_end_excl = min(max(range_end_excl, 0), total_entries)

    batch_metadata = batch_metadata_full[range_start:range_end_excl]
    if not batch_metadata and range_start >= total_entries:
        print(
            f"[WARNING] infer_range_start={range_start} >= total_entries={total_entries}; nothing to process"
        )
    print(
        f"[INFO] infer_range [{range_start}, {range_end_excl}) -> {len(batch_metadata)} entries "
        f"(left-closed right-open on full metadata)"
    )
    
    # Prepare output directory
    output_dir = os.path.dirname(args.output_path)
    if not output_dir:
        output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    base_output_name = os.path.basename(args.output_path)
    base_output_prefix = os.path.splitext(base_output_name)[0]
    
    # Store all inference results
    all_results = []
    successful_count = 0
    failed_count = 0
    processed_count = 0

    cursor = range_start
    stop_excl = range_end_excl
    while cursor < stop_excl:
        metadata_global_idx = cursor
        item = batch_metadata_full[metadata_global_idx]
        args._current_batch_entry_index = metadata_global_idx
        print("\n" + "="*70)
        print(
            f"Processing metadata index {metadata_global_idx} / {total_entries} "
            f"(range [{range_start}, {range_end_excl}))"
        )
        print("="*70)
        
        # Get prompt (based on use_long_prompt)
        if args.use_long_prompt:
            # If use_long_prompt is True, prefer long_prompt
            if 'long_prompt' in item and item['long_prompt']:
                prompt = item['long_prompt']
                print(f"[INFO] Entry metadata[{metadata_global_idx}]: Using long_prompt (length: {len(prompt)} chars)")
            elif 'prompt' in item and item['prompt']:
                prompt = item['prompt']
                print(f"[WARNING] Entry metadata[{metadata_global_idx}]: long_prompt2 not found, falling back to prompt (length: {len(prompt)} chars)")
            elif getattr(args, "interaction_allow_empty_prompt", False):
                prompt = str(getattr(args, "interaction_empty_prompt_fallback", "") or "face portrait")
                print(
                    f"[WARNING] Entry metadata[{metadata_global_idx}] has no prompt or long_prompt2; "
                    f"using interaction fallback ({len(prompt)} chars)"
                )
            else:
                print(f"[WARNING] Entry metadata[{metadata_global_idx}] has no prompt or long_prompt2, skipping...")
                failed_count += 1
                cursor += 1
                continue
        else:
            # If use_long_prompt is False, use prompt
            if 'prompt' in item and item['prompt']:
                prompt = item['prompt']
                print(f"[INFO] Entry metadata[{metadata_global_idx}]: Using prompt (use_long_prompt=False, length: {len(prompt)} chars)")
            elif 'long_prompt' in item and item['long_prompt']:
                # If prompt is missing, fall back to long_prompt
                prompt = item['long_prompt']
                print(f"[WARNING] Entry metadata[{metadata_global_idx}]: prompt not found, falling back to long_prompt (length: {len(prompt)} chars)")
            elif getattr(args, "interaction_allow_empty_prompt", False):
                prompt = str(getattr(args, "interaction_empty_prompt_fallback", "") or "face portrait")
                print(
                    f"[WARNING] Entry metadata[{metadata_global_idx}] has no prompt; "
                    f"using interaction fallback ({len(prompt)} chars)"
                )
            else:
                print(f"[WARNING] Entry metadata[{metadata_global_idx}] has no prompt, skipping...")
                failed_count += 1
                cursor += 1
                continue
        
        condition_image_path, reference_path, reference_image_path, path_err = resolve_paths_from_data_item(
            item,
            args.dataset_base_path,
            entry_label=f" [metadata {metadata_global_idx}]",
        )
        if path_err:
            print(f"[WARNING] {path_err}, skipping...")
            failed_count += 1
            cursor += 1
            continue

        if reference_path:
            reference_basename = os.path.basename(reference_path)
            reference_name = os.path.splitext(reference_basename)[0]
            entry_output_dir = os.path.join(
                output_dir, f'{base_output_prefix}_{metadata_global_idx:05d}_{reference_name}'
            )
        else:
            entry_output_dir = os.path.join(
                output_dir, f'{base_output_prefix}_{metadata_global_idx:05d}'
            )
        
        os.makedirs(entry_output_dir, exist_ok=True)
        output_path = os.path.join(entry_output_dir, 'result.png')
        
        current_reference_feature = None
        if adaface is not None and reference_image_path and os.path.exists(reference_image_path):
            try:
                print(f"[INFO] Extracting reference feature from: {reference_image_path}")
                current_reference_feature = adaface.extract_feature(reference_image_path)
            except Exception as e:
                print(f"[WARNING] Failed to extract reference feature: {e}")
                current_reference_feature = None

        current_sface_reference_feature = None
        if sface is not None and reference_image_path and os.path.exists(reference_image_path):
            try:
                print(f"[INFO] Extracting SFace reference feature from: {reference_image_path}")
                current_sface_reference_feature = _extract_sface_feature_from_image_path(
                    sface_model=sface,
                    img_path=reference_image_path,
                    device=device,
                    image_size=getattr(args, 'sface_image_size', 112),
                )
            except Exception as e:
                print(f"[WARNING] Failed to extract SFace reference feature: {e}")
                current_sface_reference_feature = None
        
        try:
            prompts_data = process_single_inference(
                prompt=prompt,
                condition_image_path=condition_image_path,
                reference_image_path=reference_image_path,
                output_path=output_path,
                model=model,
                vae=vae,
                t5=t5,
                adaface=adaface,
                reference_feature=current_reference_feature,
                sface=sface,
                sface_reference_feature=current_sface_reference_feature,
                args=args,
                device=device,
                base_ratios=base_ratios
            )
            
            prompts_data['entry_index'] = metadata_global_idx
            # Support both formats: prefer 'path', fall back to 'gt_image'
            prompts_data['path'] = item.get('path', '')  # data_info.json format
            prompts_data['gt_image'] = item.get('gt_image', '')  # metadata.json format (backward compatible)
            prompts_data['control_image'] = item.get('control_image', '')
            prompts_data['output_dir'] = entry_output_dir
            all_results.append(prompts_data)
            successful_count += 1
            processed_count += 1

            jump = prompts_data.get('switch_entry_index')
            if jump is not None:
                try:
                    jump_idx = int(jump)
                except (TypeError, ValueError):
                    jump_idx = None
                if jump_idx is not None and 0 <= jump_idx < total_entries:
                    print(f"[INFO] switch_entry_index={jump_idx}; jumping metadata cursor")
                    cursor = jump_idx
                    stop_excl = total_entries
                    continue
                print(f"[WARNING] ignore invalid switch_entry_index={jump}")
            cursor += 1

        except Exception as e:
            print(
                f"[ERROR] Failed to process metadata[{metadata_global_idx}]: {e}"
            )
            import traceback
            traceback.print_exc()
            failed_count += 1
            cursor += 1
            continue
    
    # Save batch inference summary results
    summary_file = os.path.join(output_dir, f'{base_output_prefix}_batch_summary.json')
    summary_data = {
        'total_entries_in_metadata': total_entries,
        'infer_range_start': range_start,
        'infer_range_end_exclusive': range_end_excl,
            'processed_entries': processed_count,
        'successful': successful_count,
        'failed': failed_count,
        'results': all_results
    }
    
    # Compute overall statistics
    if all_results:
        all_avg_similarities = [r.get('average_similarity') for r in all_results if r.get('average_similarity') is not None]
        all_1n_avg_similarities = [r.get('average_1_N_similarity_score') for r in all_results if r.get('average_1_N_similarity_score') is not None]
        
        if all_avg_similarities:
            summary_data['overall_average_similarity'] = sum(all_avg_similarities) / len(all_avg_similarities)
        if all_1n_avg_similarities:
            summary_data['overall_average_1_N_similarity'] = sum(all_1n_avg_similarities) / len(all_1n_avg_similarities)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("BATCH INFERENCE COMPLETED")
    print("="*70)
    print(f"[INFO] Total entries in metadata: {total_entries}")
    print(f"[INFO] infer_range [{range_start}, {range_end_excl}) on full metadata")
    print(f"[INFO] Processed entries (this run): {processed_count}")
    print(f"[INFO] Successful: {successful_count}")
    print(f"[INFO] Failed: {failed_count}")
    print(f"[INFO] Summary saved to: {summary_file}")
    
    print("\n[INFO] All inference completed!")


if __name__ == '__main__':
    main()
