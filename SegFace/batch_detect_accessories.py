import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# SegFace/ may not be a Python package in this repo (no __init__.py),
# so import from the same directory to avoid import failures.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from get_mask import LABELS, load_segface_model


DEFAULT_PARTS = ["hat", "eye_g"]


def _resolve_device(device: str) -> str:
    if device in ("auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _find_images(input_dir: Path, recursive: bool, exts: set[str]) -> list[Path]:
    if recursive:
        candidates = [
            p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in exts
        ]
    else:
        candidates = [
            p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower().lstrip(".") in exts
        ]
    candidates.sort()
    return candidates


@torch.no_grad()
def _infer_preds(image_path: Path, model: torch.nn.Module) -> np.ndarray | None:
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img_resized = cv2.resize(img, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    device = next(model.parameters()).device
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # SegFace forward: (img_tensor, labels=None, dataset=None)
    seg_output = model(img_tensor, None, None)
    mask_probs = F.interpolate(seg_output, size=(512, 512), mode="bilinear", align_corners=False)
    preds = torch.argmax(mask_probs, dim=1).detach().cpu().numpy()[0]
    return preds


def _check_parts_from_preds(preds: np.ndarray, parts: list[str]) -> list[str]:
    # Keep output order consistent with SegFace LABELS index order.
    parts_sorted = sorted(parts, key=lambda p: LABELS.index(p))

    found: list[str] = []
    for part in parts_sorted:
        idx = LABELS.index(part)
        if np.any(preds == idx):
            found.append(part)
    return found


def _load_done_set(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    done: set[str] = set()
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                img_name = obj.get("image")
                if isinstance(img_name, str):
                    done.add(img_name)
            except json.JSONDecodeError:
                # If the file is partially written/corrupted, keep going.
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch detect SegFace accessories masks (hat/eye_g).")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing face images.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="SegFace checkpoint path. If omitted, uses get_mask.py default.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu/auto.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively.")
    parser.add_argument(
        "--exts",
        type=str,
        default="jpg,jpeg,png,webp",
        help="Comma-separated extensions. Default: jpg,jpeg,png,webp",
    )
    parser.add_argument(
        "--skip_done",
        action="store_true",
        help="If output_jsonl exists, skip images already present in it.",
    )
    parser.add_argument("--parts", type=str, default=",".join(DEFAULT_PARTS), help="Comma-separated: hat,ear_r,eye_g")
    parser.add_argument("--print_every", type=int, default=100, help="Print progress every N images.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    exts = {e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()}

    done = set()
    if args.skip_done and output_jsonl.exists():
        done = _load_done_set(output_jsonl)
        print(f"[INFO] Loaded {len(done)} done entries from {output_jsonl}")

    device = _resolve_device(args.device)
    model = load_segface_model(checkpoint_path=args.checkpoint, device=device)
    if model is None:
        raise RuntimeError("Failed to load SegFace model.")

    images = _find_images(input_dir, args.recursive, exts)
    if not images:
        raise RuntimeError(f"No images found in {input_dir} with extensions {sorted(exts)}")

    total = 0
    ok = 0
    with output_jsonl.open("a", encoding="utf-8") as out_f:
        for i, img_path in enumerate(images):
            img_name = img_path.name
            if img_name in done:
                continue

            total += 1
            preds = _infer_preds(img_path, model=model)
            if preds is None:
                accessories_str = ""
            else:
                found_parts = _check_parts_from_preds(preds, parts=parts)
                accessories_str = "" if len(found_parts) == 0 else str(found_parts)

            record = {"image": img_name, "accesories": accessories_str}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            ok += 1

            if args.print_every and (ok % args.print_every == 0):
                print(f"[INFO] Processed {ok}/{total} new images (output appended). Last: {img_name}")

    print(f"[DONE] Finished. Newly processed={total}, written_lines={ok}, output={output_jsonl}")


if __name__ == "__main__":
    main()

