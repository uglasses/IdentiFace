#!/usr/bin/env python3
"""
Two-image cosine similarity inference using SFace(CL) backbone weights.

The script:
  1) Loads `models/SFace_backbone.pth`
  2) Encodes image1 and image2 into embeddings
  3) Computes cosine similarity between embeddings

Note: SFace is typically trained with aligned 112x112 faces.
If input images are not aligned, enable --align mtcnn to align them.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _add_sface_to_path() -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sface_root = repo_root
    if not os.path.isdir(sface_root):
        raise FileNotFoundError(f"Cannot find SFace folder: {sface_root}")

    # `backbones` has __init__.py, while `utils` does not.
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

    # Import after sys.path manipulation.
    from backbones.iresnet import iresnet100, iresnet50

    if network == "iresnet50":
        model = iresnet50(dropout=0.4, num_features=embedding_size, use_se=use_se)
    elif network == "iresnet100":
        model = iresnet100(num_features=embedding_size, use_se=use_se)
    else:
        raise ValueError(f"Unsupported network: {network}")

    ckpt: Any = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        # Common checkpoint formats.
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Strip common prefixes from DDP/DataParallel.
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
    # SFace training uses Normalize(mean=[0.5]*3, std=[0.5]*3) with inputs in [0,1].
    arr = np.asarray(pil_rgb).astype(np.float32) / 255.0  # (H,W,3), RGB
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
    Returns input tensor with shape (1,3,image_size,image_size), normalized to [-1,1].
    """
    if align == "none":
        pil = Image.open(img_path).convert("RGB")
        pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
        return _pil_to_sface_tensor(pil)

    if align != "mtcnn":
        raise ValueError("align must be one of: none, mtcnn")

    if mtcnn is None:
        raise ValueError("mtcnn is required when align='mtcnn'")

    # Alignment path (optional dependency).
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
    pil_rgb = Image.fromarray(rgb)
    return _pil_to_sface_tensor(pil_rgb)


@torch.no_grad()
def _extract_embedding(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    feats = model(img_tensor.to(device))
    feats = F.normalize(feats, p=2, dim=1)  # cosine-ready
    return feats.squeeze(0).detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="SFace pair cosine similarity inference")
    parser.add_argument("--img1", required=True, type=str, help="Path to first image")
    parser.add_argument("--img2", required=True, type=str, help="Path to second image")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/SFace_backbone.pth",
        help="Path to SFace backbone weights",
    )
    parser.add_argument("--network", type=str, default="iresnet50", choices=["iresnet50", "iresnet100"])
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--use-se", action="store_true", help="Whether to enable SE module (if your weights use it)")
    parser.add_argument("--image-size", type=int, default=112, help="SFace input resolution (usually 112)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--align", type=str, default="none", choices=["none", "mtcnn"], help="Face alignment")
    parser.add_argument("--print-embeddings", action="store_true", help="Also print embedding vectors (debug)")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    mtcnn = None
    if args.align == "mtcnn":
        # Optional dependency; imported lazily to keep default path lightweight.
        from facenet_pytorch import MTCNN

        # `device` for facenet_pytorch accepts torch.device or int; keep consistent.
        mtcnn = MTCNN(select_largest=True, post_process=False, device=0 if device.type == "cuda" else "cpu")

    model = _load_backbone(
        weights_path=args.weights,
        device=device,
        network=args.network,
        embedding_size=args.embedding_size,
        use_se=args.use_se,
    )

    img1_tensor = _preprocess_image(args.img1, args.image_size, args.align, mtcnn=mtcnn)
    img2_tensor = _preprocess_image(args.img2, args.image_size, args.align, mtcnn=mtcnn)

    emb1 = _extract_embedding(model, img1_tensor, device=device)
    emb2 = _extract_embedding(model, img2_tensor, device=device)

    cosine_sim = float(torch.sum(emb1 * emb2).item())
    print(f"cosine_similarity: {cosine_sim:.6f}")
    if args.print_embeddings:
        print(f"emb1: {emb1.numpy()}")
        print(f"emb2: {emb2.numpy()}")


if __name__ == "__main__":
    main()

