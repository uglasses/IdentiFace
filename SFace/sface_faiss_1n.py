#!/usr/bin/env python3
"""
SFace (CL backbone) FAISS 1:N retrieval with persistent embedding cache.

Example:
  python3 sface_faiss_1n.py \
    --query /path/to/query.jpg \
    --gallery_dir /path/to/gallery \
    --top_k 10 \
    --device cuda

Notes:
- SFace was trained with aligned 112x112 faces. If your inputs are not aligned,
  consider `--align mtcnn` (requires `facenet_pytorch` + its dependencies).
- Cache is stored under `gallery_dir/.sface_embedding_cache/` and is keyed by:
  weights file + preprocessing options.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _add_sface_to_path() -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sface_root = repo_root
    if not os.path.isdir(sface_root):
        raise FileNotFoundError(f"Cannot find SFace folder: {sface_root}")

    if sface_root not in sys.path:
        sys.path.insert(0, sface_root)
    utils_root = os.path.join(sface_root, "utils")
    if utils_root not in sys.path:
        sys.path.insert(0, utils_root)  # for `from align_trans import norm_crop`
    return sface_root


def _load_backbone(
    weights_path: str,
    device: torch.device,
    network: str,
    embedding_size: int = 512,
    use_se: bool = False,
) -> torch.nn.Module:
    _add_sface_to_path()

    from backbones.iresnet import iresnet100, iresnet50

    if network == "iresnet50":
        model = iresnet50(dropout=0.4, num_features=embedding_size, use_se=use_se)
    elif network == "iresnet100":
        model = iresnet100(num_features=embedding_size, use_se=use_se)
    else:
        raise ValueError(f"Unsupported network: {network}")

    ckpt: Any = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")

    # Strip common prefixes (DDP/DataParallel/backbone.*)
    state_dict: Dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        state_dict[nk] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading backbone ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading backbone ({len(unexpected)}): {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


def _pil_to_sface_tensor(pil_rgb: Image.Image) -> torch.Tensor:
    # SFace uses Normalize(mean=[0.5]*3, std=[0.5]*3) and expects pixels in [0,1].
    arr = np.asarray(pil_rgb).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0)  # (1,3,H,W)


def _preprocess_image(
    img_path: str,
    image_size: int,
    align: str,
    mtcnn: Optional[Any] = None,
) -> torch.Tensor:
    """
    Return tensor (1,3,image_size,image_size) normalized to [-1,1].
    """
    if align == "none":
        pil = Image.open(img_path).convert("RGB")
        pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
        return _pil_to_sface_tensor(pil)

    if align != "mtcnn":
        raise ValueError("align must be one of: none, mtcnn")

    if mtcnn is None:
        raise ValueError("mtcnn is required when align='mtcnn'")

    import cv2
    from align_trans import norm_crop

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    _, _, landmarks = mtcnn.detect(bgr, landmarks=True)
    if landmarks is None:
        raise ValueError(f"MT-CNN failed to detect face landmarks in: {img_path}")

    # landmarks: (1,5,2) for the largest face
    warped_bgr = norm_crop(bgr, landmark=landmarks[0], image_size=image_size)
    rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    return _pil_to_sface_tensor(Image.fromarray(rgb))


@torch.no_grad()
def _extract_embedding(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    feat = model(img_tensor.to(device))
    feat = F.normalize(feat, p=2, dim=1)  # unit-norm for cosine similarity
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _iter_images_from_dir(
    gallery_dir: str,
    recursive: bool = False,
    only_original_suffix: bool = True,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
) -> List[str]:
    gallery_dir = os.path.abspath(gallery_dir)
    if recursive:
        patterns = [os.path.join(gallery_dir, "**", f"*{ext}") for ext in extensions]
        paths: List[str] = []
        for ptn in patterns:
            paths.extend(glob.glob(ptn, recursive=True))
    else:
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(gallery_dir, f"*{ext}")))

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        if only_original_suffix:
            stem = os.path.splitext(os.path.basename(rp))[0]
            if not stem.endswith("_original"):
                continue
        seen.add(rp)
        out.append(rp)
    return out


def _load_embedding_cache(cache_file: str) -> Tuple[Dict[str, np.ndarray], bool]:
    if not cache_file or not os.path.exists(cache_file):
        return {}, False
    try:
        data = np.load(cache_file, allow_pickle=True)
        feats = data["feats"]
        paths = data["paths"].tolist()
        path_to_feat = {p: feats[i] for i, p in enumerate(paths)}
        return path_to_feat, True
    except Exception as e:
        print(f"[WARN] Failed to load embedding cache {cache_file}: {e}")
        return {}, False


def _save_embedding_cache(cache_file: str, feats: np.ndarray, paths: List[str]) -> None:
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    np.savez_compressed(cache_file, feats=feats, paths=np.array(paths, dtype=object))


def _get_weights_fingerprint(weights_path: str) -> str:
    st = os.stat(weights_path)
    # Include mtime+size to avoid stale cache when weights change.
    return f"{os.path.realpath(weights_path)}|size={st.st_size}|mtime={int(st.st_mtime)}"


def _get_cache_file_path(
    gallery_dir: str,
    cache_name_prefix: str,
    key_payload: Dict[str, Any],
) -> str:
    cache_dir = os.path.join(gallery_dir, ".sface_embedding_cache")
    key_json = json.dumps(key_payload, sort_keys=True, ensure_ascii=True)
    cache_hash = hashlib.md5(key_json.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{cache_name_prefix}_{cache_hash}.npz")


def _get_or_extract_candidate_embeddings(
    model: torch.nn.Module,
    candidate_paths: List[str],
    cache_file: str,
    device: torch.device,
    image_size: int,
    align: str,
    mtcnn: Optional[Any],
    skip_failed: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Incremental cache:
    - load existing (path->feat) from cache
    - extract only missing images
    - merge & rewrite cache after extraction
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
        print(f"[INFO] Extracting {len(missing_paths)} missing candidate embeddings...")
        for img_path in missing_paths:
            try:
                img_tensor = _preprocess_image(
                    img_path=img_path,
                    image_size=image_size,
                    align=align,
                    mtcnn=mtcnn,
                )
                feat = _extract_embedding(model, img_tensor, device=device)
            except Exception as e:
                if not skip_failed:
                    raise
                print(f"[WARN] Failed to extract feature for {img_path}: {e}")
                continue

            path_to_feat_cached[img_path] = feat
            out_paths_valid.append(img_path)
            out_feats.append(feat)

    if not out_feats:
        raise ValueError("No valid candidate embeddings available after cache+extraction.")

    merged_paths = list(path_to_feat_cached.keys())
    merged_feats = np.stack([path_to_feat_cached[p] for p in merged_paths], axis=0).astype(np.float32)

    if loaded:
        if missing_paths:
            _save_embedding_cache(cache_file, merged_feats, merged_paths)
    else:
        _save_embedding_cache(cache_file, merged_feats, merged_paths)

    # Important: out_feats correspond to (out_paths_valid), not merged_paths.
    return np.stack(out_feats, axis=0).astype(np.float32), out_paths_valid


def _build_faiss_index(
    candidate_feats_norm: np.ndarray,
    index_type: str,
    nlist: int,
    nprobe: int,
    use_gpu: bool,
    gpu_id: int,
):
    """
    candidate_feats_norm must be unit L2-norm. We use inner product (cosine similarity).
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        print(f"[WARN] faiss not available ({e}); will fallback to brute-force.")
        if use_gpu:
            print("[WARN] --faiss_use_gpu ignored because faiss isn't available.")
        return None

    cands = np.ascontiguousarray(candidate_feats_norm, dtype=np.float32)
    d = int(cands.shape[1])
    index_type = str(index_type).strip().lower()

    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(cands)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        nlist_eff = max(1, min(int(nlist), int(cands.shape[0]) - 1))
        index = faiss.IndexIVFFlat(quantizer, d, nlist_eff, faiss.METRIC_INNER_PRODUCT)
        index.train(cands)
        index.add(cands)
        index.nprobe = max(1, int(nprobe))
    else:
        raise ValueError("--faiss_index_type must be one of: flat|ivf")

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
            raise RuntimeError("faiss-gpu requested but GPU resources are not available in this faiss build.")
        res = faiss.StandardGpuResources()
        try:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False
            index = faiss.index_cpu_to_gpu(res, int(gpu_id), index, co)
        except Exception:
            index = faiss.index_cpu_to_gpu(res, int(gpu_id), index)
        if hasattr(index, "nprobe"):
            index.nprobe = max(1, int(nprobe))

    return index


def _search_topk(
    index,
    query_feat_norm: np.ndarray,
    candidate_feats_norm: np.ndarray,
    candidate_paths_valid: List[str],
    top_k: int,
):
    """
    Returns list of (path, score) sorted by score desc.
    """
    if index is None:
        # brute-force fallback
        q = np.ascontiguousarray(query_feat_norm.reshape(1, -1), dtype=np.float32)
        sims = (q @ np.ascontiguousarray(candidate_feats_norm.T, dtype=np.float32)).reshape(-1)
        top_idx = np.argsort(-sims)[:top_k]
        return [(candidate_paths_valid[int(i)], float(sims[int(i)])) for i in top_idx]

    # FAISS search
    q = np.ascontiguousarray(query_feat_norm.reshape(1, -1).astype(np.float32))
    scores, idxs = index.search(q, int(top_k))
    out: List[Tuple[str, float]] = []
    for score, idx in zip(scores.reshape(-1), idxs.reshape(-1)):
        out.append((candidate_paths_valid[int(idx)], float(score)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SFace CL FAISS 1:N retrieval")
    parser.add_argument("--query", required=True, type=str, help="Path to query image")
    parser.add_argument("--gallery_dir", required=True, type=str, help="Directory with gallery images")

    parser.add_argument("--weights", type=str, default="models/SFace_backbone.pth")
    parser.add_argument("--network", type=str, default="iresnet50", choices=["iresnet50", "iresnet100"])
    parser.add_argument("--use-se", action="store_true", help="Enable SE module (if required by your weights)")
    parser.add_argument("--embedding-size", type=int, default=512)

    parser.add_argument("--image-size", type=int, default=112, help="SFace input resolution")
    parser.add_argument("--align", type=str, default="none", choices=["none", "mtcnn"], help="Face alignment")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan gallery_dir for images")
    parser.add_argument(
        "--include_all_suffix",
        action="store_true",
        help="Include all suffixes in gallery (default: only *_original.*).",
    )

    parser.add_argument("--top_k", type=int, default=10, help="Top-k retrieval results")

    parser.add_argument("--faiss_index_type", type=str, default="flat", choices=["flat", "ivf"])
    parser.add_argument("--faiss_nlist", type=int, default=100)
    parser.add_argument("--faiss_nprobe", type=int, default=10)
    parser.add_argument("--faiss_use_gpu", action="store_true", help="Use GPU for FAISS (if available)")
    parser.add_argument("--faiss_gpu_id", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--cache_name_prefix", type=str, default="sface_emb")
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON path (empty means no file).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.query):
        raise FileNotFoundError(f"--query not found: {args.query}")
    if not os.path.isdir(args.gallery_dir):
        raise NotADirectoryError(f"--gallery_dir not found: {args.gallery_dir}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"--weights not found: {args.weights}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    mtcnn = None
    if args.align == "mtcnn":
        try:
            from facenet_pytorch import MTCNN
        except Exception as e:
            raise ImportError(
                "align='mtcnn' requires `facenet_pytorch`. Install it first, e.g. `pip install facenet-pytorch`."
            ) from e
        mtcnn = MTCNN(select_largest=True, post_process=False, device=0 if device.type == "cuda" else "cpu")

    model = _load_backbone(
        weights_path=args.weights,
        device=device,
        network=args.network,
        embedding_size=args.embedding_size,
        use_se=bool(args.use_se),
    )

    query_real = os.path.realpath(args.query)
    gallery_dir_real = os.path.realpath(args.gallery_dir)
    only_original_suffix = not bool(args.include_all_suffix)
    candidate_paths = _iter_images_from_dir(
        gallery_dir_real,
        recursive=bool(args.recursive),
        only_original_suffix=only_original_suffix,
    )
    if not candidate_paths:
        suffix_desc = "*_original.*" if only_original_suffix else "all supported extensions"
        raise ValueError(f"No gallery images found in {args.gallery_dir} ({suffix_desc}).")

    top_k = max(1, int(args.top_k))
    # Build cache key
    cache_file = _get_cache_file_path(
        gallery_dir=gallery_dir_real,
        cache_name_prefix=args.cache_name_prefix,
        key_payload={
            "weights": _get_weights_fingerprint(args.weights),
            "network": args.network,
            "use_se": bool(args.use_se),
            "embedding_size": args.embedding_size,
            "image_size": args.image_size,
            "align": args.align,
            "only_original_suffix": bool(only_original_suffix),
        },
    )

    print(f"[INFO] Gallery size: {len(candidate_paths)} images")
    print(f"[INFO] Candidate embedding cache: {cache_file}")

    candidate_feats, candidate_paths_valid = _get_or_extract_candidate_embeddings(
        model=model,
        candidate_paths=candidate_paths,
        cache_file=cache_file,
        device=device,
        image_size=args.image_size,
        align=args.align,
        mtcnn=mtcnn,
        skip_failed=True,
    )

    # Ensure unit-norm (safety)
    candidate_feats_norm = F.normalize(torch.from_numpy(candidate_feats), p=2, dim=1).numpy().astype(np.float32)

    n = candidate_feats_norm.shape[0]
    top_k = min(top_k, n)

    index = _build_faiss_index(
        candidate_feats_norm=candidate_feats_norm,
        index_type=args.faiss_index_type,
        nlist=int(args.faiss_nlist),
        nprobe=int(args.faiss_nprobe),
        use_gpu=bool(args.faiss_use_gpu),
        gpu_id=int(args.faiss_gpu_id),
    )

    print(f"[INFO] Building query embedding for: {query_real}")
    query_tensor = _preprocess_image(
        img_path=query_real,
        image_size=args.image_size,
        align=args.align,
        mtcnn=mtcnn,
    )
    query_feat = _extract_embedding(model, query_tensor, device=device)
    query_feat_norm = F.normalize(torch.from_numpy(query_feat), p=2, dim=0).numpy().astype(np.float32)

    results = _search_topk(
        index=index,
        query_feat_norm=query_feat_norm,
        candidate_feats_norm=candidate_feats_norm,
        candidate_paths_valid=candidate_paths_valid,
        top_k=top_k,
    )

    print("\nTop-k results:")
    for rank, (path, score) in enumerate(results, start=1):
        print(f"  {rank:02d}. {os.path.basename(path)}\tscore={score:.6f}\tpath={path}")

    if args.output_json:
        payload = {
            "query": query_real,
            "gallery_dir": gallery_dir_real,
            "top_k": top_k,
            "faiss_index_type": args.faiss_index_type,
            "faiss_use_gpu": bool(args.faiss_use_gpu),
            "results": [
                {"rank": rank, "match_path": p, "match_name": os.path.basename(p), "score": score}
                for rank, (p, score) in enumerate(results, start=1)
            ],
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()

