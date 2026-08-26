import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    # Fallback: if tqdm isn't installed, keep code runnable.
    def tqdm(x, *args, **kwargs):
        return x

# Add AdaFace directory to path
ADAFACE_DIR = os.path.join(os.path.dirname(__file__), "AdaFace")
if os.path.isdir(ADAFACE_DIR) and ADAFACE_DIR not in sys.path:
    sys.path.insert(0, ADAFACE_DIR)  # Use insert(0) so this path is searched first

try:
    import net as adaface_net
    from face_alignment import align as adaface_align
except ImportError as e:
    raise ImportError(f"Failed to import AdaFace modules: {e}. Please ensure AdaFace directory is present.")


def _to_input(pil_rgb_image):
    """Convert PIL RGB image to input tensor for AdaFace model.
    This function is copied from AdaFace/inference.py to avoid import conflicts.
    """
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor


class AdaFace(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super(AdaFace, self).__init__()
        self.device = device
        self.model_path = model_path
        self.model = self._load_model(model_path, device)

    def _infer_architecture_from_path(self, model_path):
        """Infer architecture from model path (e.g., 'ir_101' from 'adaface_ir101_*.ckpt')."""
        basename = os.path.basename(model_path).lower()
        if 'ir_101' in basename or 'ir101' in basename:
            return 'ir_101'
        elif 'ir_50' in basename or 'ir50' in basename:
            return 'ir_50'
        elif 'ir_se_50' in basename or 'irse50' in basename:
            return 'ir_se_50'
        elif 'ir_34' in basename or 'ir34' in basename:
            return 'ir_34'
        elif 'ir_18' in basename or 'ir18' in basename:
            return 'ir_18'
        else:
            # Default to ir_50 if cannot infer
            return 'ir_50'

    def _load_model(self, model_path, device):
        """Load AdaFace model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AdaFace model not found at: {model_path}")
        
        # Infer architecture from model path
        architecture = self._infer_architecture_from_path(model_path)
        print(f"Inferred AdaFace architecture: {architecture}")
        
        # Build model
        model = adaface_net.build_model(architecture)
        
        # Load checkpoint
        print(f"Loading AdaFace model from: {model_path}")
        statedict = torch.load(model_path, map_location=device)
        
        # Extract model state dict (remove 'model.' prefix if present)
        if 'state_dict' in statedict:
            statedict = statedict['state_dict']
        model_statedict = {
            key[6:]: val for key, val in statedict.items() 
            if key.startswith('model.')
        }
        if not model_statedict:
            # If no 'model.' prefix, use the state_dict as is
            model_statedict = statedict
        
        model.load_state_dict(model_statedict)
        model.to(device)
        model.eval()
        print("AdaFace model loaded successfully")
        return model

    def forward(self, x):
        feature, norm = self.model(x)
        return feature

    def extract_feature(self, img_path):
        """Extract face feature from image path."""
        # Load and align face
        aligned_face = adaface_align.get_aligned_face(img_path)
        if aligned_face is None:
            raise ValueError(f"Failed to detect or align face in image: {img_path}")
        
        # Convert to input tensor
        input_tensor = _to_input(aligned_face)
        input_tensor = input_tensor.to(self.device)
        
        # Extract feature
        with torch.no_grad():
            feature, _ = self.model(input_tensor)
        
        # Convert to numpy and squeeze batch dimension
        feature = feature.detach().cpu().numpy()
        feature = feature.squeeze(0)  # Remove batch dimension
        
        return feature

    def extract_feature_from_pil(self, pil_rgb_image):
        """
        Extract face feature from a PIL RGB image.
        Useful when you don't want to save intermediate preprocessing results to disk.
        """
        aligned_face = adaface_align.get_aligned_face("", rgb_pil_image=pil_rgb_image)
        if aligned_face is None:
            raise ValueError("Failed to detect or align face from provided PIL image.")

        input_tensor = _to_input(aligned_face)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            feature, _ = self.model(input_tensor)

        feature = feature.detach().cpu().numpy()
        feature = feature.squeeze(0)
        return feature

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two feature vectors."""
        # Convert to numpy if torch tensors
        if isinstance(vec1, torch.Tensor):
            vec1 = vec1.detach().cpu().numpy()
        if isinstance(vec2, torch.Tensor):
            vec2 = vec2.detach().cpu().numpy()
        
        # Flatten if needed
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)


def extract_feature(model, device, img_path):
    """Standalone function to extract feature from image."""
    # Load and align face
    aligned_face = adaface_align.get_aligned_face(img_path)
    if aligned_face is None:
        raise ValueError(f"Failed to detect or align face in image: {img_path}")
    
    # Convert to input tensor
    input_tensor = _to_input(aligned_face)
    input_tensor = input_tensor.to(device)
    
    # Extract feature
    with torch.no_grad():
        feature, _ = model(input_tensor)
    
    # Convert to numpy and squeeze batch dimension
    feature = feature.detach().cpu().numpy()
    feature = feature.squeeze(0)  # Remove batch dimension
    
    return feature


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two feature vectors."""
    # Convert to numpy if torch tensors
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.detach().cpu().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.detach().cpu().numpy()
    
    # Flatten if needed
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def _parse_low_res(low_res: str) -> Tuple[int, int]:
    """
    Parse low-res argument.
    - "256" -> (256, 256)
    - "256x192" -> (256, 192)
    """
    low_res = str(low_res).strip().lower()
    if "x" in low_res:
        w_str, h_str = low_res.split("x", 1)
        return int(w_str), int(h_str)
    v = int(low_res)
    return v, v


def _fix_image_to_size(
    img_path: str,
    size: str,
    resample_method: str,
    out_path: str = "",
) -> str:
    """
    Resize an image to a fixed resolution (e.g. AdaFace expects 112x112).
    Saves the resized image and returns the output path.
    """
    target_w, target_h = _parse_low_res(size)
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"--adaface_input_size must be positive, got {target_w}x{target_h}")

    if not out_path:
        img_dir = os.path.dirname(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(img_dir, f"{stem}_toAda{target_w}x{target_h}.png")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img_fixed = img.resize((target_w, target_h), resample=_get_pil_resample(resample_method))
    img_fixed.save(out_path)
    return out_path


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize features to make inner product equivalent to cosine similarity."""
    norms = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.clip(norms, eps, None)


def _iter_candidate_images_from_dir(
    candidates_dir: str,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    only_original_suffix: bool = False,
) -> List[str]:
    paths: List[str] = []
    for name in sorted(os.listdir(candidates_dir)):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in exts):
            if only_original_suffix:
                stem = os.path.splitext(name)[0].lower()
                if not stem.endswith("_original"):
                    continue
            paths.append(os.path.join(candidates_dir, name))
    return paths


def _read_candidate_list(list_file: str) -> List[str]:
    paths: List[str] = []
    with open(list_file, "r") as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            paths.append(p)
    return paths


def _extract_features_for_paths(
    adaface: "AdaFace",
    img_paths: List[str],
    skip_failed: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract AdaFace embeddings for image paths (face-aligned internally)."""
    features: List[np.ndarray] = []
    valid_paths: List[str] = []
    for img_path in tqdm(img_paths, desc="Extracting candidates", total=len(img_paths)):
        try:
            feat = adaface.extract_feature(img_path)
        except Exception as e:
            if not skip_failed:
                raise
            print(f"Warning: failed to extract feature for {img_path}: {e}")
            continue
        features.append(feat)
        valid_paths.append(img_path)

    if not features:
        raise ValueError("No valid candidate embeddings extracted (all failed).")
    return np.stack(features, axis=0), valid_paths


def _load_embedding_cache(cache_file: str) -> Tuple[dict, bool]:
    """
    Load embedding cache.
    Returns (path_to_feat, loaded_successfully).
    """
    if not cache_file or not os.path.exists(cache_file):
        return {}, False
    try:
        data = np.load(cache_file, allow_pickle=True)
        feats = data["feats"]
        paths = data["paths"].tolist()
        path_to_feat = {p: feats[i] for i, p in enumerate(paths)}
        return path_to_feat, True
    except Exception as e:
        print(f"Warning: failed to load embedding cache {cache_file}: {e}")
        return {}, False


def _save_embedding_cache(cache_file: str, feats: np.ndarray, paths: List[str]) -> None:
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    np.savez_compressed(cache_file, feats=feats, paths=np.array(paths, dtype=object))


def _get_or_extract_candidate_embeddings(
    adaface: "AdaFace",
    candidate_paths: List[str],
    cache_file: str,
    skip_failed: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Get embeddings from cache if possible, otherwise extract and update cache.
    Embeddings are keyed by absolute image path.
    """
    path_to_feat_cached, loaded = _load_embedding_cache(cache_file)

    out_paths_valid: List[str] = []
    out_feats: List[np.ndarray] = []
    missing_paths: List[str] = []

    for p in candidate_paths:
        if p in path_to_feat_cached:
            out_paths_valid.append(p)
            out_feats.append(path_to_feat_cached[p])
        else:
            missing_paths.append(p)

    if missing_paths:
        print(f"Extracting {len(missing_paths)} missing candidate embeddings...")
        missing_feats, missing_valid_paths = _extract_features_for_paths(
            adaface, missing_paths, skip_failed=skip_failed
        )
        for p, feat in zip(missing_valid_paths, missing_feats):
            path_to_feat_cached[p] = feat
            out_paths_valid.append(p)
            out_feats.append(feat)

    if not out_feats:
        raise ValueError("No valid candidate embeddings available after cache+extraction.")

    # Persist updated cache (merged).
    merged_paths = list(path_to_feat_cached.keys())
    merged_feats = np.stack([path_to_feat_cached[p] for p in merged_paths], axis=0)
    if loaded:
        # Only rewrite if we actually extracted something.
        if missing_paths:
            _save_embedding_cache(cache_file, merged_feats, merged_paths)
    else:
        _save_embedding_cache(cache_file, merged_feats, merged_paths)

    return np.stack(out_feats, axis=0), out_paths_valid


def _extract_query_features_for_paths(
    adaface: "AdaFace",
    img_paths: List[str],
    low_res: str,
    downsample_method: str,
    adaface_input_size: str,
    tmp_dir: str,
    save_low_res_dir: Optional[str] = None,
    save_fixed_dir: Optional[str] = None,
    skip_failed: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract query embeddings for images with the same degradation+resize
    pipeline as the single-query mode.
    """
    low_w, low_h = _parse_low_res(low_res)
    ada_w, ada_h = _parse_low_res(adaface_input_size)

    low_tag = f"low{low_w}x{low_h}"
    ada_tag = f"toAda{ada_w}x{ada_h}"

    low_dir = save_low_res_dir
    fixed_dir = save_fixed_dir
    if low_dir:
        os.makedirs(low_dir, exist_ok=True)
    if fixed_dir:
        os.makedirs(fixed_dir, exist_ok=True)

    features: List[np.ndarray] = []
    valid_paths: List[str] = []

    for img_path in tqdm(img_paths, desc="Extracting inputs", total=len(img_paths)):
        try:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            # In-memory degrade+resize to AdaFace-fixed input size.
            img = Image.open(img_path).convert("RGB")
            low_img = img.resize((low_w, low_h), resample=_get_pil_resample(downsample_method))
            fixed_img = low_img.resize((ada_w, ada_h), resample=_get_pil_resample(downsample_method))

            if low_dir:
                low_res_out_path = os.path.join(low_dir, f"{stem}_{low_tag}_{downsample_method}.png")
                low_img.save(low_res_out_path)
            if fixed_dir:
                fixed_out_path = os.path.join(
                    fixed_dir, f"{stem}_{low_tag}_{ada_tag}_{downsample_method}.png"
                )
                fixed_img.save(fixed_out_path)

            feat = adaface.extract_feature_from_pil(fixed_img)
        except Exception as e:
            if not skip_failed:
                raise
            print(f"Warning: failed to extract query feature for {img_path}: {e}")
            continue

        features.append(feat)
        valid_paths.append(img_path)

    if not features:
        raise ValueError("No valid query embeddings extracted (all failed).")

    return np.stack(features, axis=0), valid_paths


def _get_or_extract_query_embeddings_cached(
    adaface: "AdaFace",
    query_paths: List[str],
    cache_file: str,
    low_res: str,
    downsample_method: str,
    adaface_input_size: str,
    tmp_dir: str,
    save_low_res_dir: Optional[str] = None,
    save_fixed_dir: Optional[str] = None,
    skip_failed: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Like _get_or_extract_candidate_embeddings, but uses the
    degradation+resize pipeline for query images.
    """
    path_to_feat_cached, loaded = _load_embedding_cache(cache_file)

    missing_paths: List[str] = []
    for p in query_paths:
        if p not in path_to_feat_cached:
            missing_paths.append(p)

    if missing_paths:
        print(f"Extracting {len(missing_paths)} missing query embeddings...")
        missing_feats, missing_valid_paths = _extract_query_features_for_paths(
            adaface=adaface,
            img_paths=missing_paths,
            low_res=low_res,
            downsample_method=downsample_method,
            adaface_input_size=adaface_input_size,
            tmp_dir=tmp_dir,
            save_low_res_dir=save_low_res_dir,
            save_fixed_dir=save_fixed_dir,
            skip_failed=skip_failed,
        )
        for p, feat in zip(missing_valid_paths, missing_feats):
            path_to_feat_cached[p] = feat

    out_paths_valid = [p for p in query_paths if p in path_to_feat_cached]
    out_feats = [path_to_feat_cached[p] for p in out_paths_valid]
    if not out_feats:
        raise ValueError("No valid query embeddings available after cache+extraction.")

    merged_paths = list(path_to_feat_cached.keys())
    merged_feats = np.stack([path_to_feat_cached[p] for p in merged_paths], axis=0)
    _save_embedding_cache(cache_file, merged_feats, merged_paths)

    return np.stack(out_feats, axis=0), out_paths_valid


def _build_faiss_index(
    candidate_feats: np.ndarray,
    index_type: str,
    nlist: int,
    nprobe: int,
    use_gpu: bool,
    gpu_id: int,
):
    """
    Build a FAISS index once for candidate embeddings.
    candidate_feats must already be L2-normalized.
    """
    try:
        import faiss  # type: ignore
    except Exception:
        if use_gpu:
            raise RuntimeError("faiss-gpu requested but faiss is not installed.")
        return None

    candidate_feats = np.ascontiguousarray(candidate_feats, dtype=np.float32)
    d = int(candidate_feats.shape[1])
    index_type = str(index_type).strip().lower()

    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(candidate_feats)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        nlist_eff = max(1, min(int(nlist), int(candidate_feats.shape[0]) - 1))
        index = faiss.IndexIVFFlat(
            quantizer, d, nlist_eff, faiss.METRIC_INNER_PRODUCT
        )
        index.train(candidate_feats)
        index.add(candidate_feats)
        index.nprobe = max(1, int(nprobe))
    else:
        raise ValueError("--faiss_index_type must be one of: flat|ivf")

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
            raise RuntimeError("faiss-gpu not available but --faiss_use_gpu was set.")
        res = faiss.StandardGpuResources()
        try:
            co = faiss.GpuClonerOptions()
            # Float16 can trigger cuBLAS issues in some builds; keep it stable in fp32.
            co.useFloat16 = False
            index = faiss.index_cpu_to_gpu(res, int(gpu_id), index, co)
        except Exception:
            index = faiss.index_cpu_to_gpu(res, int(gpu_id), index)

        if hasattr(index, "nprobe"):
            index.nprobe = max(1, int(nprobe))

    return index


def _faiss_search_batch(index, query_feats: np.ndarray, top_k: int):
    """Return (scores, idxs) from a FAISS index for a batch of queries."""
    query_feats = np.ascontiguousarray(query_feats, dtype=np.float32)
    scores, idxs = index.search(query_feats, int(top_k))
    return scores, idxs


def _search_topk_with_faiss(
    query_feat: np.ndarray,
    candidate_feats: np.ndarray,
    candidate_paths: List[str],
    top_k: int,
    index_type: str = "ivf",
    nlist: int = 100,
    nprobe: int = 10,
    use_gpu: bool = False,
    gpu_id: int = 0,
) -> List[Tuple[str, float]]:
    """Search top-k most similar candidates. Returns list of (path, cosine_similarity)."""
    query = _l2_normalize(query_feat.astype(np.float32).reshape(1, -1))
    cands = _l2_normalize(candidate_feats.astype(np.float32))
    query = np.ascontiguousarray(query, dtype=np.float32)
    cands = np.ascontiguousarray(cands, dtype=np.float32)

    try:
        import faiss  # type: ignore
    except Exception as e:
        print(f"faiss not available ({e}); falling back to brute-force search.")
        if use_gpu:
            raise RuntimeError(
                "GPU FAISS requested (--faiss_use_gpu) but faiss-gpu is not installed. "
                "Please install faiss-gpu (or remove --faiss_use_gpu)."
            )
        sims = (query @ cands.T).reshape(-1)
        top_idx = np.argsort(-sims)[:top_k]
        return [(candidate_paths[int(i)], float(sims[int(i)])) for i in top_idx]

    d = int(cands.shape[1])
    index_type = str(index_type).strip().lower()
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(cands)
    elif index_type == "ivf":
        # IVF uses k-means clustering to partition the vector space.
        quantizer = faiss.IndexFlatIP(d)
        nlist_eff = max(1, min(int(nlist), int(cands.shape[0]) - 1))
        index = faiss.IndexIVFFlat(
            quantizer, d, nlist_eff, faiss.METRIC_INNER_PRODUCT
        )
        index.train(cands)
        index.add(cands)
        index.nprobe = max(1, int(nprobe))
    else:
        raise ValueError("--faiss_index_type must be one of: flat|ivf")

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
            raise RuntimeError(
                "GPU FAISS requested (--faiss_use_gpu) but faiss-gpu is not available. "
                "Please install faiss-gpu."
            )
        # Move index to GPU for faster search.
        res = faiss.StandardGpuResources()
        try:
            co = faiss.GpuClonerOptions()
            # Float16 can trigger cuBLAS issues in some builds; keep it stable in fp32.
            co.useFloat16 = False
        except Exception:
            co = None

        if co is None:
            index = faiss.index_cpu_to_gpu(res, int(gpu_id), index)
        else:
            index = faiss.index_cpu_to_gpu(res, int(gpu_id), index, co)

        # Ensure IVF search params remain effective.
        if hasattr(index, "nprobe"):
            index.nprobe = max(1, int(nprobe))

    scores, idxs = index.search(query, int(top_k))
    out: List[Tuple[str, float]] = []
    for score, idx in zip(scores.reshape(-1), idxs.reshape(-1)):
        out.append((candidate_paths[int(idx)], float(score)))
    return out


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
        raise ValueError(
            f"Unknown --downsample_method '{method}'. "
            f"Supported: {', '.join(mapping.keys())}"
        )
    return mapping[method]


def degrade_img_to_low_res(
    img_path: str,
    low_res: str,
    downsample_method: str,
    low_res_out_path: str = "",
) -> str:
    """
    Degrade an image to low resolution using traditional downsampling.
    Saves the degraded image and returns the output path.
    """
    low_w, low_h = _parse_low_res(low_res)
    if low_w <= 0 or low_h <= 0:
        raise ValueError(f"--low_res must be positive, got {low_w}x{low_h}")

    if not low_res_out_path:
        img_dir = os.path.dirname(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        low_res_out_path = os.path.join(img_dir, f"{stem}_low{low_w}x{low_h}.png")

    os.makedirs(os.path.dirname(low_res_out_path) or ".", exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img_low = img.resize((low_w, low_h), resample=_get_pil_resample(downsample_method))
    img_low.save(low_res_out_path)
    return low_res_out_path


def load_identity_map(identity_file):
    """Load identity mapping from file.
    
    Args:
        identity_file: Path to identity file (format: filename identity_id)
    
    Returns:
        dict: Mapping from filename to identity_id
    """
    identity_map = {}
    if not os.path.exists(identity_file):
        print(f"Warning: Identity file not found at {identity_file}")
        return identity_map
    
    with open(identity_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                filename = parts[0]
                identity_id = parts[1]
                identity_map[filename] = identity_id
    
    return identity_map


def get_filename_from_path(img_path):
    """Extract identity lookup key from an image path.
    
    Maps common layouts (e.g. CelebA-HQ-img ``15945.jpg``, or legacy names containing digit ids)
    to identity-file keys ``{id}.jpg`` (see e.g. identity_celebA.txt).
    
    Args:
        img_path: Full path to image file
    
    Returns:
        str: Filename/key for identity lookup
    """
    basename = os.path.basename(img_path)
    dirname = os.path.dirname(img_path)

    # Try to find numeric prefix in various patterns
    # Pattern 1: result_00007_xxx -> extract 00007
    match = re.search(r'result_(\d+)', dirname)
    if not match:
        match = re.search(r'result_(\d+)', basename)
    if not match:
        # Pattern 2: Look for leading digits in filename
        match = re.search(r'^(\d+)', basename)
    if not match:
        # Pattern 3: Look for any sequence of digits
        match = re.search(r'(\d+)', basename)
    
    if match:
        numeric_str = match.group(1)
        # Convert to integer to remove leading zeros, then back to string
        numeric_value = int(numeric_str)
        return f"{numeric_value}.jpg"
    else:
        # Fallback: use basename as is
        return basename


def main():
    parser = argparse.ArgumentParser(description='AdaFace verification / fast candidate search')
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/adaface_ir101_webface12m.ckpt',
        help='Path to AdaFace model checkpoint',
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    parser.add_argument(
        '--img1_path',
        type=str,
        default='',
        help='Query face image path (required in pairwise or single-query batch mode).'
    )
    parser.add_argument(
        '--img2_path',
        type=str,
        default='',
        help='Gallery face image path (pairwise mode).'
    )

    parser.add_argument(
        '--low_res',
        type=str,
        default='1024',
        help="Low resolution for degrading img1 before feature extraction. Accept int or WxH."
    )
    parser.add_argument(
        '--downsample_method',
        type=str,
        default='bicubic',
        help="Downsample method for PIL resize: nearest|bilinear|bicubic|lanczos|box"
    )
    parser.add_argument(
        '--low_res_out_path',
        type=str,
        default=None,
        help="Optional output path to save degraded img1. If empty, auto-generate near img1."
    )
    parser.add_argument(
        '--adaface_fixed_out_path',
        type=str,
        default=None,
        help="Optional output path to save AdaFace-fixed image (112x112) for single-query/multi-query. If not set, do not save."
    )
    parser.add_argument(
        '--adaface_input_size',
        type=str,
        default='112',
        help="Fixed resize resolution fed into AdaFace alignment/model. Default: 112 (AdaFace expects 112x112)."
    )

    parser.add_argument(
        '--candidates_dir',
        type=str,
        default='',
        help='If set, run batch mode: compare img1 against all images in this directory.'
    )
    parser.add_argument(
        '--candidates_list',
        type=str,
        default='',
        help='If set, run batch mode: compare img1 against image paths listed in this text file.'
    )
    parser.add_argument(
        '--inputs_dir',
        type=str,
        default='',
        help='If set, run multi-query mode: read input images from this folder (only *_original.*), '
             'match each against the candidates set.'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='',
        help='Output json path for multi-query mode. If empty, auto-generate under inputs_dir.'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help='Return top-k matches in batch mode.'
    )
    parser.add_argument(
        '--faiss_index_type',
        type=str,
        default='ivf',
        help="Vector index type for fast search: ivf (clustering) or flat (exact)."
    )
    parser.add_argument('--faiss_nlist', type=int, default=100, help='IVF parameter nlist')
    parser.add_argument('--faiss_nprobe', type=int, default=10, help='IVF parameter nprobe')
    parser.add_argument(
        '--faiss_use_gpu',
        action='store_true',
        help='Move FAISS index to GPU before search (requires faiss-gpu).',
    )
    parser.add_argument(
        '--faiss_gpu_id',
        type=int,
        default=0,
        help='GPU id for faiss-gpu (default: 0).',
    )
    parser.add_argument(
        '--faiss_query_batch_size',
        type=int,
        default=512,
        help='Multi-query mode: number of query vectors to search per chunk.',
    )

    parser.add_argument(
        '--identity_file',
        type=str,
        default='dataset_train/identity.txt',
        help='Path to identity mapping file'
    )
    args = parser.parse_args()

    running_inputs_batch = bool(args.inputs_dir)
    running_batch = bool(args.candidates_dir or args.candidates_list)

    if running_inputs_batch:
        if not running_batch:
            raise ValueError("--inputs_dir requires --candidates_dir or --candidates_list.")
        if not os.path.isdir(args.inputs_dir):
            raise ValueError(f"--inputs_dir not found or not a directory: {args.inputs_dir}")
        if args.img2_path:
            print("Note: --img2_path will be ignored in multi-query mode.")
        if args.img1_path:
            print("Note: --img1_path is ignored in multi-query mode; inputs are read from --inputs_dir.")
    else:
        if running_batch and args.img2_path:
            print("Note: --img2_path will be ignored in batch mode.")

        if running_batch and not args.img1_path:
            raise ValueError("Batch mode requires --img1_path (single query image).")

        if (not running_batch) and (not args.img2_path):
            raise ValueError("Pairwise mode requires --img2_path, or set --candidates_dir/--candidates_list for batch mode.")
        if (not running_batch) and (not args.img1_path):
            raise ValueError("Pairwise mode requires --img1_path.")

    # Load identity mapping
    print(f"Loading identity mapping from: {args.identity_file}")
    identity_map = load_identity_map(args.identity_file)
    print(f"Loaded {len(identity_map)} identity mappings")

    if running_inputs_batch:
        top_k = max(1, int(args.top_k))

        # Initialize AdaFace once.
        adaface = AdaFace(args.model_path, device=args.device)

        # -------- Build candidate embeddings (cached) --------
        if not (args.candidates_dir or args.candidates_list):
            raise ValueError("multi-query mode requires candidates_dir or candidates_list.")

        if args.candidates_dir:
            candidate_paths = _iter_candidate_images_from_dir(
                args.candidates_dir,
                only_original_suffix=True,
            )
            candidate_paths = list(dict.fromkeys(candidate_paths))

            model_key = os.path.basename(args.model_path).replace(".", "_")
            cache_file = os.path.join(
                args.candidates_dir,
                ".adaface_embedding_cache",
                f"emb_{model_key}_crop112.npz",
            )
            print(f"Using candidate embedding cache: {cache_file}")
            candidate_feats, candidate_paths_valid = _get_or_extract_candidate_embeddings(
                adaface,
                candidate_paths,
                cache_file=cache_file,
                skip_failed=True,
            )
        else:
            candidate_paths = _read_candidate_list(args.candidates_list)
            candidate_paths = list(dict.fromkeys(candidate_paths))
            if not candidate_paths:
                raise ValueError("No candidate images found.")
            candidate_feats, candidate_paths_valid = _extract_features_for_paths(
                adaface,
                candidate_paths,
                skip_failed=True,
            )

        candidate_feats_norm = _l2_normalize(candidate_feats.astype(np.float32))
        top_k = min(top_k, candidate_feats_norm.shape[0])

        # -------- Build FAISS index once --------
        index = _build_faiss_index(
            candidate_feats_norm,
            index_type=args.faiss_index_type,
            nlist=args.faiss_nlist,
            nprobe=args.faiss_nprobe,
            use_gpu=bool(args.faiss_use_gpu),
            gpu_id=int(args.faiss_gpu_id),
        )

        # -------- Load input images --------
        input_paths = _iter_candidate_images_from_dir(
            args.inputs_dir,
            only_original_suffix=True,
        )
        input_paths = list(dict.fromkeys(input_paths))
        if not input_paths:
            raise ValueError(f"No input images found in {args.inputs_dir} (expected *_original.*).")

        # -------- Load/extract query embeddings with cache --------
        used_candidates_as_queries = False
        same_dir_as_candidates = (
            bool(args.candidates_dir)
            and os.path.realpath(args.inputs_dir) == os.path.realpath(args.candidates_dir)
        )

        if same_dir_as_candidates and args.low_res == '1024':
            # Special case: avoid duplicating feature extraction.
            # Only safe to reuse when no degradation/resizing difference exists:
            # - low_res == 1024 (no degradation)
            # - adaface_input_size == 1024 (no resize difference before alignment)
            # Otherwise we must extract query embeddings with low_res pipeline.
            if args.adaface_input_size == '1024':
                used_candidates_as_queries = True
                path_to_candidate_idx = {p: i for i, p in enumerate(candidate_paths_valid)}
                query_paths_valid = [p for p in input_paths if p in path_to_candidate_idx]
                query_idxs = [path_to_candidate_idx[p] for p in query_paths_valid]
                query_feats_norm = candidate_feats_norm[query_idxs]
                print(
                    f"inputs_dir == candidates_dir; reuse candidates embeddings for {len(query_paths_valid)} inputs."
                )
            else:
                # Skip reuse: will fall through to low_res extraction/caching.
                used_candidates_as_queries = False
        else:
            model_key = os.path.basename(args.model_path).replace(".", "_")
            low_w, low_h = _parse_low_res(args.low_res)
            ada_w, ada_h = _parse_low_res(args.adaface_input_size)
            low_tag = f"low{low_w}x{low_h}"
            ada_tag = f"toAda{ada_w}x{ada_h}"
            query_cache_dir = os.path.join(args.inputs_dir, ".adaface_embedding_cache_inputs")
            query_cache_file = os.path.join(
                query_cache_dir,
                f"query_{model_key}_{low_tag}_{ada_tag}_{args.downsample_method}.npz",
            )
            tmp_dir = os.path.join(query_cache_dir, "tmp")

            query_feats, query_paths_valid = _get_or_extract_query_embeddings_cached(
                adaface=adaface,
                query_paths=input_paths,
                cache_file=query_cache_file,
                low_res=args.low_res,
                downsample_method=args.downsample_method,
                adaface_input_size=args.adaface_input_size,
                tmp_dir=tmp_dir,
                save_low_res_dir=args.low_res_out_path,
                save_fixed_dir=args.adaface_fixed_out_path,
                skip_failed=True,
            )

            query_feats_norm = _l2_normalize(query_feats.astype(np.float32))

        # -------- Search all inputs --------
        query_feats_norm = np.ascontiguousarray(query_feats_norm, dtype=np.float32)
        if index is not None:
            bs = max(1, int(args.faiss_query_batch_size))
            n_queries = int(query_feats_norm.shape[0])
            scores = np.empty((n_queries, top_k), dtype=np.float32)
            idxs = np.empty((n_queries, top_k), dtype=np.int64)
            for start in tqdm(
                range(0, n_queries, bs),
                desc="FAISS search chunks",
                total=(n_queries + bs - 1) // bs,
            ):
                end = min(n_queries, start + bs)
                s, i = _faiss_search_batch(index, query_feats_norm[start:end], top_k)
                scores[start:end, :] = s
                idxs[start:end, :] = i
        else:
            # Brute-force fallback if faiss isn't installed.
            bs = max(1, int(args.faiss_query_batch_size))
            n_queries = int(query_feats_norm.shape[0])
            scores = np.empty((n_queries, top_k), dtype=np.float32)
            idxs = np.empty((n_queries, top_k), dtype=np.int64)
            for start in tqdm(
                range(0, n_queries, bs),
                desc="Brute-force search chunks",
                total=(n_queries + bs - 1) // bs,
            ):
                end = min(n_queries, start + bs)
                sims = query_feats_norm[start:end] @ candidate_feats_norm.T
                idxs_chunk = np.argsort(-sims, axis=1)[:, :top_k]
                scores_chunk = np.take_along_axis(sims, idxs_chunk, axis=1)
                scores[start:end, :] = scores_chunk
                idxs[start:end, :] = idxs_chunk

        # -------- Prepare JSON results --------
        results = []
        for qi, in_path in enumerate(
            tqdm(query_paths_valid, desc="Assembling JSON", total=len(query_paths_valid))
        ):
            in_name = os.path.basename(in_path)
            in_key = get_filename_from_path(in_path)
            in_identity = identity_map.get(in_key)

            top_matches = []
            for rank in range(top_k):
                ci = int(idxs[qi, rank])
                match_path = candidate_paths_valid[ci]
                match_name = os.path.basename(match_path)
                match_key = get_filename_from_path(match_path)
                match_identity = identity_map.get(match_key)
                identity_match = (
                    in_identity is not None
                    and match_identity is not None
                    and in_identity == match_identity
                )
                top_matches.append(
                    {
                        "rank": rank + 1,
                        "match_image": match_name,
                        "match_identity": match_identity,
                        "identity_match": identity_match,
                        "score": float(scores[qi, rank]),
                    }
                )

            results.append(
                {
                    "input_image": in_name,
                    "input_identity": in_identity,
                    "top_k": top_matches,
                }
            )

        if args.output_json:
            out_json_path = args.output_json
        else:
            out_json_path = os.path.join(
                args.inputs_dir, f"adaface_multiquery_top{top_k}.json"
            )

        payload = {
            "inputs_dir": args.inputs_dir,
            "candidates_dir": args.candidates_dir if args.candidates_dir else None,
            "candidates_list": args.candidates_list if args.candidates_list else None,
            "model_path": args.model_path,
            "identity_file": args.identity_file,
            "low_res": args.low_res,
            "downsample_method": args.downsample_method,
            "adaface_input_size": args.adaface_input_size,
            "top_k": top_k,
            "faiss_index_type": args.faiss_index_type,
            "faiss_use_gpu": bool(args.faiss_use_gpu),
            "faiss_gpu_id": int(args.faiss_gpu_id),
            "used_candidates_as_queries": used_candidates_as_queries,
            "results": results,
        }

        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Saved multi-query JSON to: {out_json_path}")
        return

    # Query identity (single-query mode)
    img1_filename = get_filename_from_path(args.img1_path)
    identity1 = identity_map.get(img1_filename)

    if running_batch:
        print(f"\n{'='*60}")
        print("Batch Search Mode")
        print(f"{'='*60}")
        print(f"Query image: {img1_filename}")
        print(f"  Identity ID: {identity1 if identity1 else 'NOT FOUND'}")
        print(f"{'='*60}\n")
    else:
        img2_filename = get_filename_from_path(args.img2_path)
        identity2 = identity_map.get(img2_filename)

        print(f"\n{'='*60}")
        print("Identity Comparison:")
        print(f"{'='*60}")
        print(f"Image 1: {img1_filename}")
        print(f"  Identity ID: {identity1 if identity1 else 'NOT FOUND'}")
        print(f"Image 2: {img2_filename}")
        print(f"  Identity ID: {identity2 if identity2 else 'NOT FOUND'}")

        if identity1 is None or identity2 is None:
            print(f"\nWarning: Could not find identity for one or both images.")
            if identity1 is None:
                print(f"  - {img1_filename} not found in identity file")
            if identity2 is None:
                print(f"  - {img2_filename} not found in identity file")
            identity_match = False
        else:
            identity_match = (identity1 == identity2)
            print(f"\nIdentity Match: {'YES' if identity_match else 'NO'}")
            if identity_match:
                print(f"  Both images belong to the same person (Identity ID: {identity1})")
            else:
                print(f"  Images belong to different persons")
        print(f"{'='*60}\n")

    # Initialize AdaFace
    adaface = AdaFace(args.model_path, device=args.device)

    # Extract query feature (always apply low-res degradation + resize-to-AdaFace input).
    # Only save intermediate images if output paths are explicitly provided.
    print(
        f"Degrading img1 to low resolution: {args.low_res} "
        f"(method={args.downsample_method}, save_low_res={args.low_res_out_path is not None})"
    )
    low_w, low_h = _parse_low_res(args.low_res)
    ada_w, ada_h = _parse_low_res(args.adaface_input_size)
    low_tag = f"low{low_w}x{low_h}"
    ada_tag = f"toAda{ada_w}x{ada_h}"

    img_pil = Image.open(args.img1_path).convert("RGB")
    low_img = img_pil.resize((low_w, low_h), resample=_get_pil_resample(args.downsample_method))
    fixed_img = low_img.resize((ada_w, ada_h), resample=_get_pil_resample(args.downsample_method))

    if args.low_res_out_path is not None:
        # If it's a directory, write a per-image file; otherwise treat as exact file path.
        if os.path.isdir(args.low_res_out_path):
            stem = os.path.splitext(os.path.basename(args.img1_path))[0]
            low_res_img1_path = os.path.join(
                args.low_res_out_path,
                f"{stem}_{low_tag}_{args.downsample_method}.png",
            )
        else:
            low_res_img1_path = args.low_res_out_path
        os.makedirs(os.path.dirname(low_res_img1_path) or ".", exist_ok=True)
        low_img.save(low_res_img1_path)
        print(f"Saved degraded img1 to: {low_res_img1_path}")

    if args.adaface_fixed_out_path is not None:
        if os.path.isdir(args.adaface_fixed_out_path):
            stem = os.path.splitext(os.path.basename(args.img1_path))[0]
            fixed_out_path = os.path.join(
                args.adaface_fixed_out_path,
                f"{stem}_{low_tag}_{ada_tag}_{args.downsample_method}.png",
            )
        else:
            fixed_out_path = args.adaface_fixed_out_path
        os.makedirs(os.path.dirname(fixed_out_path) or ".", exist_ok=True)
        fixed_img.save(fixed_out_path)
        print(f"Saved AdaFace-fixed degraded img1 to: {fixed_out_path}")

    feature1 = adaface.extract_feature_from_pil(fixed_img)
    print(f"Feature 1 shape: {feature1.shape}")

    if running_batch:
        # Build candidate list
        if args.candidates_dir:
            candidate_paths = _iter_candidate_images_from_dir(
                args.candidates_dir,
                only_original_suffix=True,
            )
            # De-duplicate while preserving order
            candidate_paths = list(dict.fromkeys(candidate_paths))

            model_key = os.path.basename(args.model_path).replace(".", "_")
            cache_file = os.path.join(
                args.candidates_dir,
                ".adaface_embedding_cache",
                f"emb_{model_key}_crop112.npz",
            )

            print(f"Using candidate embedding cache: {cache_file}")
            print(
                f"Extracting candidate embeddings for {len(candidate_paths)} images "
                f"(only '*_original.*' from candidates_dir)..."
            )
            candidate_feats, candidate_paths_valid = _get_or_extract_candidate_embeddings(
                adaface,
                candidate_paths,
                cache_file=cache_file,
                skip_failed=True,
            )
        else:
            candidate_paths = _read_candidate_list(args.candidates_list)
            candidate_paths = list(dict.fromkeys(candidate_paths))

        if not candidate_paths:
            raise ValueError("No candidate images found.")

        # In candidates_list mode (dir mode already used cache-aware extraction).
        if not args.candidates_dir:
            print(f"Extracting candidate embeddings for {len(candidate_paths)} images...")
            candidate_feats, candidate_paths_valid = _extract_features_for_paths(
                adaface,
                candidate_paths,
                skip_failed=True,
            )
        print(f"Extracted {candidate_feats.shape[0]} valid candidate embeddings.")

        top_hits = _search_topk_with_faiss(
            query_feat=feature1,
            candidate_feats=candidate_feats,
            candidate_paths=candidate_paths_valid,
            top_k=max(1, int(args.top_k)),
            index_type=args.faiss_index_type,
            nlist=args.faiss_nlist,
            nprobe=args.faiss_nprobe,
            use_gpu=bool(args.faiss_use_gpu),
            gpu_id=int(args.faiss_gpu_id),
        )

        print(f"\n{'='*60}")
        print(f"Top-{len(top_hits)} matches:")
        print(f"{'='*60}")
        for rank, (p, score) in enumerate(top_hits, start=1):
            cand_filename = get_filename_from_path(p)
            cand_identity = identity_map.get(cand_filename)
            match = (identity1 is not None and cand_identity is not None and identity1 == cand_identity)
            identity_str = (
                "MATCH" if match else
                ("NOT FOUND" if cand_identity is None else "NO MATCH")
            )
            print(f"[{rank}] {os.path.basename(p)} | score={score:.6f} | identity={identity_str}")
        print(f"{'='*60}\n")
        return top_hits[0][1] if top_hits else 0.0

    # Pairwise mode
    print(f"Extracting feature from: {args.img2_path}")
    feature2 = adaface.extract_feature(args.img2_path)
    print(f"Feature 2 shape: {feature2.shape}")

    similarity = adaface.cosine_similarity(feature1, feature2)
    print(f'Cosine similarity between the two faces: {similarity:.6f}')
    return similarity


if __name__ == '__main__':
    main()

