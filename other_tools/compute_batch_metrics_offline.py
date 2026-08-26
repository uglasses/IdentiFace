#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
for _p in (_TOOLS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FGCLIP_TOOLS = _TOOLS_DIR / "FG-CLIP"
if _FGCLIP_TOOLS.is_dir() and str(_FGCLIP_TOOLS) not in sys.path:
    sys.path.insert(0, str(_FGCLIP_TOOLS))

from other_impls import (
    InferenceMetricSuite,
    _compute_ms_ssim_pair,
    _compute_pairwise_metric_lists,
    _ensure_rgb_uint8_bhwc,
    _resize_rgb_pair,
    _to_gray_float01,
    summarize_metric_lists,
)


def _mean_valid(xs: list[float | None]) -> float | None:
    """Mean over non-None entries; float NaN is skipped (does not poison the mean)."""
    vals: list[float] = []
    for x in xs:
        if x is None:
            continue
        v = float(x)
        if math.isnan(v):
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else None


def _similarity_nullish_count(xs: list[Any]) -> int:
    """Count entries with no usable similarity: None, or float NaN (JSON rarely yields the latter)."""
    n = 0
    for v in xs:
        if v is None:
            n += 1
        elif isinstance(v, float) and math.isnan(v):
            n += 1
    return n


def _ratio_true(xs: list[bool | None]) -> float | None:
    vals = [bool(x) for x in xs if x is not None]
    if not vals:
        return None
    return float(sum(1 for x in vals if x) / len(vals))


def _find_first_prefix(subdir: Path, prefix: str) -> Path | None:
    matches = sorted(p for p in subdir.iterdir() if p.is_file() and p.name.startswith(prefix))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] {subdir.name}: multiple {prefix}* found; using {matches[0].name}")
    return matches[0]


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] {path}: cannot load json: {e}")
        return None


def _merge_summary_with_existing(
    existing: Any,
    fresh: dict[str, Any],
    args: argparse.Namespace,
    *,
    mode: str,
) -> dict[str, Any]:
    """
    If `existing` is a prior summary dict, keep metric fields where the corresponding CLI flag is 0,
    and overwrite from `fresh` only where the flag is 1. Always refresh mode/root/num_samples from `fresh`.
    """
    out: dict[str, Any] = copy.deepcopy(existing) if isinstance(existing, dict) else {}
    out["mode"] = fresh["mode"]
    out["root"] = fresh["root"]
    out["num_samples"] = fresh["num_samples"]

    pm_old = out.get("pairwise_mean")
    pm_old = dict(pm_old) if isinstance(pm_old, dict) else {}
    fm = fresh.get("pairwise_mean") if isinstance(fresh.get("pairwise_mean"), dict) else {}
    pm = dict(pm_old)
    if args.sr_sim:
        pm["sr_sim"] = fm.get("sr_sim")
    if args.fsim:
        pm["fsim"] = fm.get("fsim")
    if args.vif:
        pm["vif"] = fm.get("vif")
    if args.lpips:
        pm["lpips"] = fm.get("lpips")
    if args.ms_ssim:
        pm["ms_ssim"] = fm.get("ms_ssim")
    if any([args.sr_sim, args.fsim, args.vif, args.lpips, args.ms_ssim]):
        out["pairwise_mean"] = pm
    elif pm_old:
        out["pairwise_mean"] = pm_old

    if args.fid:
        out["fid"] = fresh.get("fid")
    if args.inception_score:
        out["inception_score_mean"] = fresh.get("inception_score_mean")
        out["inception_score_std"] = fresh.get("inception_score_std")
    if args.fg_clip_score:
        for k in ("fgclip_similarity_mean", "fgclip_probability_percent_mean", "fgclip_walk_type"):
            if k in fresh:
                out[k] = fresh[k]
    if args.adaface:
        for k in ("adaface_match_ratio", "adaface_similarity_null_count", "adaface_avg_similarity"):
            if k in fresh:
                out[k] = fresh[k]
    if args.sface:
        for k in ("sface_match_ratio", "sface_similarity_null_count", "sface_avg_similarity"):
            if k in fresh:
                out[k] = fresh[k]
    if mode == "best_match" and args.early_stopped_both_matches_mean:
        for k in ("early_stopped_both_matches_mean", "early_stopped_both_matches_zero_mapped_to"):
            if k in fresh:
                out[k] = fresh[k]
    return out


def _merge_similarity_null_existing(
    existing: Any,
    one_summary: dict[str, Any],
    best_summary: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out: dict[str, Any] = copy.deepcopy(existing) if isinstance(existing, dict) else {}
    out["root"] = one_summary["root"]
    osub = dict(out.get("one_shot") or {})
    osub["num_samples"] = one_summary["num_samples"]
    if args.adaface:
        if "adaface_similarity_null_count" in one_summary:
            osub["adaface_similarity_null_count"] = one_summary["adaface_similarity_null_count"]
        if best_summary and "adaface_similarity_null_count" in best_summary:
            bsub = dict(out.get("best_match") or {})
            bsub["num_samples"] = best_summary["num_samples"]
            bsub["adaface_similarity_null_count"] = best_summary["adaface_similarity_null_count"]
            out["best_match"] = bsub
    if args.sface:
        if "sface_similarity_null_count" in one_summary:
            osub["sface_similarity_null_count"] = one_summary["sface_similarity_null_count"]
        if best_summary and "sface_similarity_null_count" in best_summary:
            bsub = dict(out.get("best_match") or {})
            bsub["num_samples"] = best_summary["num_samples"]
            bsub["sface_similarity_null_count"] = best_summary["sface_similarity_null_count"]
            out["best_match"] = bsub
    out["one_shot"] = osub
    return out


def _load_csv_rows(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            return [dict(row) for row in r]
    except Exception as e:
        print(f"[WARN] {path}: cannot read csv: {e}")
        return None


def _merge_per_sample_rows(
    existing: list[dict[str, Any]] | None,
    fresh: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    """Merge prior CSV rows with freshly computed rows: overwrite metric columns only where args flag is 1."""
    if not existing:
        return fresh
    by_id = {str(r.get("sample_id")): r for r in existing}
    merged: list[dict[str, Any]] = []
    for fr in fresh:
        sid = str(fr.get("sample_id"))
        old = by_id.get(sid)
        if not old:
            merged.append(dict(fr))
            continue
        row: dict[str, Any] = dict(old)
        for k in ("sample_id", "gt_path", "gen_path", "prompt_one_shot"):
            if k in fr:
                row[k] = fr[k]
        if mode == "best_match" and "prompt_best_match" in fr:
            row["prompt_best_match"] = fr["prompt_best_match"]
        if args.sr_sim:
            row["sr_sim"] = fr.get("sr_sim")
        if args.fsim:
            row["fsim"] = fr.get("fsim")
        if args.vif:
            row["vif"] = fr.get("vif")
        if args.lpips:
            row["lpips"] = fr.get("lpips")
        if args.ms_ssim:
            row["ms_ssim"] = fr.get("ms_ssim")
        if args.adaface:
            row["adaface_match"] = fr.get("adaface_match")
            row["adaface_similarity"] = fr.get("adaface_similarity")
        if args.sface:
            row["sface_match"] = fr.get("sface_match")
            row["sface_similarity"] = fr.get("sface_similarity")
        if args.fg_clip_score:
            row["fgclip_similarity"] = fr.get("fgclip_similarity")
            row["fgclip_probability_percent"] = fr.get("fgclip_probability_percent")
        if mode == "best_match" and args.early_stopped_both_matches_mean:
            row["early_stopped_both_matches_raw"] = fr.get("early_stopped_both_matches_raw")
            row["early_stopped_both_matches_for_stats"] = fr.get("early_stopped_both_matches_for_stats")
        merged.append(row)
    return merged


def _entry_by_type(results_best: Any, typ: str) -> dict[str, Any] | None:
    if not isinstance(results_best, list):
        return None
    for e in results_best:
        if isinstance(e, dict) and e.get("type") == typ:
            return e
    return None


def _read_bool(d: dict[str, Any] | None, *keys: str) -> bool | None:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if cur is True:
        return True
    if cur is False:
        return False
    return None


def _read_float(d: dict[str, Any] | None, *keys: str) -> float | None:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if cur is None:
        return None
    try:
        return float(cur)
    except Exception:
        return None


def _read_prompt(d: dict[str, Any] | None) -> str | None:
    if not isinstance(d, dict):
        return None
    p = d.get("prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()
    return None


def _build_records(
    root: Path,
    subdir_prefix: str,
    *,
    require_best_match: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    one_shot_records: list[dict[str, Any]] = []
    best_match_records: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith(subdir_prefix):
            continue

        gt = _find_first_prefix(child, "ref_img_")
        one = _find_first_prefix(child, "one_shot_")
        best = None
        if require_best_match:
            best = _find_first_prefix(child, "best_match_")
            if best is None:
                best = _find_first_prefix(child, "best_match")

        if gt is None or one is None or (require_best_match and best is None):
            print(
                f"[WARN] skip {child.name}: missing "
                f"{'ref_img_*/one_shot_' if not require_best_match else 'one of ref_img_*/one_shot_*/best_match_*'}"
            )
            continue

        results_best = _load_json(child / "results_best.json")
        result_prompts = _load_json(child / "result_prompts.json")
        one_entry = _entry_by_type(results_best, "one_shot")
        best_entry = _entry_by_type(results_best, "best_match") if require_best_match else None

        # Fallback prompt route from result_prompts first/second iteration.
        rp_iterations = result_prompts.get("iterations") if isinstance(result_prompts, dict) else None
        fallback_one_prompt = None
        fallback_best_prompt = None
        if isinstance(rp_iterations, list) and rp_iterations:
            if isinstance(rp_iterations[0], dict):
                fallback_one_prompt = _read_prompt(rp_iterations[0])
            if len(rp_iterations) > 1 and isinstance(rp_iterations[1], dict):
                fallback_best_prompt = _read_prompt(rp_iterations[1])

        one_prompt = _read_prompt(one_entry) or fallback_one_prompt
        best_prompt = _read_prompt(best_entry) or fallback_best_prompt if require_best_match else None
        early_stop = 0
        if isinstance(result_prompts, dict):
            try:
                early_stop = int(result_prompts.get("early_stopped_both_matches", 0) or 0)
            except Exception:
                early_stop = 0
        early_stop_for_stats = 7 if early_stop == 0 else early_stop

        one_shot_records.append(
            {
                "sample_id": child.name,
                "subdir": child,
                "gt_path": gt,
                "gen_path": one,
                "prompt_one_shot": one_prompt,
                "adaface_match": _read_bool(one_entry, "best_match", "matches_reference_image"),
                "sface_match": _read_bool(one_entry, "sface_best_match", "matches_reference_image"),
                "adaface_similarity": _read_float(one_entry, "similarity_score"),
                "sface_similarity": _read_float(one_entry, "sface_similarity_score"),
            }
        )
        if require_best_match:
            best_match_records.append(
                {
                    "sample_id": child.name,
                    "subdir": child,
                    "gt_path": gt,
                    "gen_path": best,
                    "prompt_one_shot": one_prompt,
                    "prompt_best_match": best_prompt,
                    "adaface_match": _read_bool(best_entry, "best_match", "matches_reference_image"),
                    "sface_match": _read_bool(best_entry, "sface_best_match", "matches_reference_image"),
                    "adaface_similarity": _read_float(best_entry, "similarity_score"),
                    "sface_similarity": _read_float(best_entry, "sface_similarity_score"),
                    "early_stopped_both_matches_raw": early_stop,
                    "early_stopped_both_matches_for_stats": early_stop_for_stats,
                }
            )
    return one_shot_records, best_match_records


def _compute_fgclip(
    records: list[dict[str, Any]],
    *,
    mode: str,
    evaluator: Any,
    walk_type: str,
) -> tuple[list[float | None], list[float | None]]:
    sims: list[float | None] = []
    probs: list[float | None] = []
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, total=len(records), desc=f"FG-CLIP ({mode})", leave=False)
    for r in iterator:
        img_path = str(r["gen_path"])
        if mode == "one_shot":
            p = r.get("prompt_one_shot")
            if not isinstance(p, str) or not p.strip():
                sims.append(None)
                probs.append(None)
                continue
            try:
                out = evaluator.compute_similarity(img_path, p, walk_type=walk_type)
                sims.append(float(out["similarity"]))
                probs.append(float(out["probability"]) * 100.0)
            except Exception as ex:
                print(f"[WARN] {r['sample_id']}: FG-CLIP(one_shot) failed: {ex}")
                sims.append(None)
                probs.append(None)
        else:
            p1 = r.get("prompt_one_shot")
            p2 = r.get("prompt_best_match")
            if not isinstance(p1, str) or not p1.strip() or not isinstance(p2, str) or not p2.strip():
                sims.append(None)
                probs.append(None)
                continue
            try:
                out1 = evaluator.compute_similarity(img_path, p1, walk_type=walk_type)
                out2 = evaluator.compute_similarity(img_path, p2, walk_type=walk_type)
                s1 = float(out1["similarity"])
                s2 = float(out2["similarity"])
                mean_sim = 0.5 * (s1 + s2)
                # probability is computed from averaged similarity(logit).
                mean_prob = float(torch.sigmoid(torch.tensor(mean_sim)).item() * 100.0)
                sims.append(mean_sim)
                probs.append(mean_prob)
            except Exception as ex:
                print(f"[WARN] {r['sample_id']}: FG-CLIP(best_match) failed: {ex}")
                sims.append(None)
                probs.append(None)
    return sims, probs


def _evaluate_mode(
    records: list[dict[str, Any]],
    *,
    mode: str,
    suite: InferenceMetricSuite,
    args: argparse.Namespace,
    fgclip_evaluator: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not records:
        raise SystemExit(f"No valid records for mode={mode}")

    real_paths = [r["gt_path"] for r in records]
    gen_paths = [r["gen_path"] for r in records]

    want_lpips = bool(args.lpips)
    want_struct = bool(args.sr_sim or args.fsim or args.vif)

    if not want_struct and not want_lpips:
        n = len(records)
        pairwise = {
            "sr_sim": [None] * n,
            "fsim": [None] * n,
            "vif": [None] * n,
            "lpips": [None] * n,
        }
    elif not want_lpips:
        pairwise = _compute_pairwise_metric_lists(
            real_paths,
            gen_paths,
            compute_lpips=False,
            lpips_model=None,
            device=torch.device("cpu"),
        )
    else:
        pairwise_kwargs: dict[str, Any] = dict(
            compute_lpips=True,
            lpips_model=suite._get_lpips_model(),
            device=suite.device,
        )
        try:
            sig = inspect.signature(_compute_pairwise_metric_lists)
            if "lpips_batch_size" in sig.parameters:
                pairwise_kwargs["lpips_batch_size"] = args.lpips_batch_size
        except Exception:
            pass
        pairwise = _compute_pairwise_metric_lists(
            real_paths,
            gen_paths,
            **pairwise_kwargs,
        )

    if not want_struct:
        n = len(records)
        pairwise["sr_sim"] = [None] * n
        pairwise["fsim"] = [None] * n
        pairwise["vif"] = [None] * n
    else:
        if not args.sr_sim:
            pairwise["sr_sim"] = [None] * len(records)
        if not args.fsim:
            pairwise["fsim"] = [None] * len(records)
        if not args.vif:
            pairwise["vif"] = [None] * len(records)

    summary_pairwise = summarize_metric_lists(
        {k: v for k, v in pairwise.items() if k in ("sr_sim", "fsim", "vif", "lpips")}
    )

    ms_ssim_scores: list[float | None] = []
    if args.ms_ssim:
        real_bhwc = _ensure_rgb_uint8_bhwc(real_paths)
        gen_bhwc = _ensure_rgb_uint8_bhwc(gen_paths)
        pairs_iter = zip(real_bhwc, gen_bhwc)
        if tqdm is not None:
            pairs_iter = tqdm(
                pairs_iter,
                total=len(real_bhwc),
                desc=f"MS-SSIM ({mode})",
                leave=False,
            )
        for real_rgb, generated_rgb in pairs_iter:
            real_rgb, generated_rgb = _resize_rgb_pair(real_rgb, generated_rgb)
            g1 = _to_gray_float01(real_rgb)
            g2 = _to_gray_float01(generated_rgb)
            try:
                ms_ssim_scores.append(float(_compute_ms_ssim_pair(g1, g2)))
            except Exception as ex:
                print(f"[WARN] {mode}: MS-SSIM failed for one sample: {ex}")
                ms_ssim_scores.append(None)

    if args.fid or args.inception_score:
        dataset_metrics = suite.compute_dataset_metrics_from_paths(real_paths, gen_paths)
        if not args.fid:
            dataset_metrics["fid"] = None
        if not args.inception_score:
            dataset_metrics["inception_score_mean"] = None
            dataset_metrics["inception_score_std"] = None
    else:
        dataset_metrics = {"fid": None, "inception_score_mean": None, "inception_score_std": None}

    fgclip_sims: list[float | None] = []
    fgclip_probs: list[float | None] = []
    if fgclip_evaluator is not None and args.fg_clip_score:
        fgclip_sims, fgclip_probs = _compute_fgclip(
            records,
            mode=mode,
            evaluator=fgclip_evaluator,
            walk_type=args.fgclip_walk_type,
        )

    adaface_matches: list[bool | None] = []
    adaface_sims: list[float | None] = []
    adaface_null_n = 0
    if args.adaface:
        adaface_matches = [r.get("adaface_match") for r in records]
        adaface_sims = [r.get("adaface_similarity") for r in records]
        adaface_null_n = _similarity_nullish_count(adaface_sims)
    sface_matches: list[bool | None] = []
    sface_sims: list[float | None] = []
    sface_null_n = 0
    if args.sface:
        sface_matches = [r.get("sface_match") for r in records]
        sface_sims = [r.get("sface_similarity") for r in records]
        sface_null_n = _similarity_nullish_count(sface_sims)

    summary_out: dict[str, Any] = {
        "mode": mode,
        "root": str(args.root.expanduser().resolve()),
        "num_samples": len(records),
        "pairwise_mean": {k: summary_pairwise.get(k) for k in ("sr_sim", "fsim", "vif", "lpips")}
        | ({"ms_ssim": _mean_valid(ms_ssim_scores)} if args.ms_ssim else {}),
        "fid": dataset_metrics.get("fid"),
        "inception_score_mean": dataset_metrics.get("inception_score_mean"),
        "inception_score_std": dataset_metrics.get("inception_score_std"),
    }
    if args.adaface:
        summary_out["adaface_match_ratio"] = _ratio_true(adaface_matches)
        summary_out["adaface_similarity_null_count"] = adaface_null_n
        summary_out["adaface_avg_similarity"] = _mean_valid(adaface_sims)
    if args.sface:
        summary_out["sface_match_ratio"] = _ratio_true(sface_matches)
        summary_out["sface_similarity_null_count"] = sface_null_n
        summary_out["sface_avg_similarity"] = _mean_valid(sface_sims)
    if fgclip_evaluator is not None and args.fg_clip_score:
        summary_out["fgclip_similarity_mean"] = _mean_valid(fgclip_sims)
        summary_out["fgclip_probability_percent_mean"] = _mean_valid(fgclip_probs)
        summary_out["fgclip_walk_type"] = args.fgclip_walk_type
    if mode == "best_match" and args.early_stopped_both_matches_mean:
        early_vals = [float(r["early_stopped_both_matches_for_stats"]) for r in records]
        summary_out["early_stopped_both_matches_mean"] = (
            float(sum(early_vals) / len(early_vals)) if early_vals else None
        )
        summary_out["early_stopped_both_matches_zero_mapped_to"] = 7

    per_sample: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        row: dict[str, Any] = {
            "sample_id": r["sample_id"],
            "gt_path": str(r["gt_path"]),
            "gen_path": str(r["gen_path"]),
            "prompt_one_shot": r.get("prompt_one_shot"),
            "ms_ssim": ms_ssim_scores[i] if args.ms_ssim and i < len(ms_ssim_scores) else None,
            "sr_sim": pairwise["sr_sim"][i],
            "fsim": pairwise["fsim"][i],
            "vif": pairwise["vif"][i],
            "lpips": pairwise["lpips"][i] if want_lpips else None,
        }
        if args.adaface:
            row["adaface_match"] = r.get("adaface_match")
            row["adaface_similarity"] = r.get("adaface_similarity")
        if args.sface:
            row["sface_match"] = r.get("sface_match")
            row["sface_similarity"] = r.get("sface_similarity")
        if mode == "best_match":
            row["prompt_best_match"] = r.get("prompt_best_match")
            if args.early_stopped_both_matches_mean:
                row["early_stopped_both_matches_raw"] = r.get("early_stopped_both_matches_raw")
                row["early_stopped_both_matches_for_stats"] = r.get("early_stopped_both_matches_for_stats")
        if fgclip_evaluator is not None and args.fg_clip_score:
            row["fgclip_similarity"] = fgclip_sims[i] if i < len(fgclip_sims) else None
            row["fgclip_probability_percent"] = fgclip_probs[i] if i < len(fgclip_probs) else None
        per_sample.append(row)
    return summary_out, per_sample


def _write_outputs(
    out_base: Path,
    mode: str,
    summary: dict[str, Any],
    per_sample: list[dict[str, Any]],
    *,
    write_csv: bool,
) -> None:
    json_path = Path(str(out_base) + f"_{mode}_summary.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {json_path}")

    if write_csv:
        csv_path = Path(str(out_base) + f"_{mode}_per_sample.csv")
        fieldnames = sorted({k for row in per_sample for k in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in per_sample:
                w.writerow(row)
        print(f"[OK] Wrote {csv_path}")


def _parse_01(s: str) -> int:
    v = int(s, 10)
    if v not in (0, 1):
        raise argparse.ArgumentTypeError("expected 0 or 1")
    return v


def _write_similarity_null_split_summary(out_base: Path, payload: dict[str, Any]) -> None:
    """Small JSON with one_shot vs best_match null counts side by side."""
    path = Path(str(out_base) + "_similarity_null_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline metrics for batch output folders with files: "
            "ref_img_*.jpg, one_shot_*.jpg, best_match_*.jpg, results_best.json, result_prompts.json. "
            "If batch_metrics_*_summary.json already exist next to the output prefix, they are loaded first; "
            "only metrics whose CLI switch is 1 overwrite or add values (0 keeps prior file content)."
        )
    )
    parser.add_argument("root", type=Path, help="Batch folder containing result_* subdirs")
    parser.add_argument("--subdir-prefix", default="result_", help="Only subdirs whose name starts with this")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--inception-batch-size", type=int, default=32)
    parser.add_argument("--inception-splits", type=int, default=10)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument(
        "--lpips-batch-size",
        type=int,
        default=32,
        help="Mini-batch size for LPIPS pairwise computation (default: 8).",
    )
    parser.add_argument(
        "--FG-CLIP_Score",
        type=_parse_01,
        default=1,
        dest="fg_clip_score",
        help="1=compute FG-CLIP similarity/probability, 0=skip",
    )
    parser.add_argument(
        "--LPIPS",
        type=_parse_01,
        default=1,
        dest="lpips",
        help="1=compute LPIPS, 0=skip",
    )
    parser.add_argument(
        "--SR_SIM",
        type=_parse_01,
        default=1,
        dest="sr_sim",
        help="1=include SR-SIM, 0=omit (pairwise pass may still run for other metrics)",
    )
    parser.add_argument(
        "--FSIM",
        type=_parse_01,
        default=1,
        dest="fsim",
        help="1=include FSIM, 0=omit",
    )
    parser.add_argument(
        "--VIF",
        type=_parse_01,
        default=1,
        dest="vif",
        help="1=include VIF, 0=omit",
    )
    parser.add_argument(
        "--MS-SSIM",
        type=_parse_01,
        default=1,
        dest="ms_ssim",
        help="1=compute MS-SSIM, 0=skip",
    )
    parser.add_argument(
        "--FID",
        type=_parse_01,
        default=1,
        dest="fid",
        help="1=compute FID, 0=skip",
    )
    parser.add_argument(
        "--IS",
        type=_parse_01,
        default=1,
        dest="inception_score",
        help="1=compute Inception Score, 0=skip",
    )
    parser.add_argument(
        "--early_stopped_both_matches_mean",
        type=_parse_01,
        default=1,
        dest="early_stopped_both_matches_mean",
        help="1=include early-stopped stats (best_match mode only), 0=omit",
    )
    parser.add_argument(
        "--AdaFace",
        type=_parse_01,
        default=1,
        dest="adaface",
        help="1=include AdaFace match ratio / similarity / null counts, 0=omit",
    )
    parser.add_argument(
        "--SFace",
        type=_parse_01,
        default=1,
        dest="sface",
        help="1=include SFace match ratio / similarity / null counts, 0=omit",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help=(
            "Write <out-prefix>_one_shot_summary.json, <out-prefix>_best_match_summary.json, "
            "and <out-prefix>_similarity_null_summary.json (null counts only if --AdaFace / --SFace are 1)"
        ),
    )
    parser.add_argument(
        "--only_one_shot",
        action="store_true",
        help="Only compute/write one_shot metrics; skip best_match computations and outputs.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write per-sample CSV files (default: off, only JSON summaries).",
    )
    parser.add_argument("--fgclip-model", type=Path, default=None)
    parser.add_argument("--fgclip-walk-type", choices=("long", "short"), default="long")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    args.root = root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    one_shot_records, best_match_records = _build_records(
        root,
        args.subdir_prefix,
        require_best_match=not args.only_one_shot,
    )
    if not one_shot_records:
        raise SystemExit("No valid one_shot records found. Check folder contents and file naming.")
    if not args.only_one_shot and not best_match_records:
        raise SystemExit("No valid best_match records found. Check folder contents and file naming.")
    print(
        f"[INFO] valid records: one_shot={len(one_shot_records)}"
        + (f", best_match={len(best_match_records)}" if not args.only_one_shot else "")
    )

    out_base = Path(args.out_prefix).expanduser().resolve() if args.out_prefix else root / "batch_metrics"
    p_one = Path(str(out_base) + "_one_shot_summary.json")
    p_best = Path(str(out_base) + "_best_match_summary.json")
    p_null = Path(str(out_base) + "_similarity_null_summary.json")
    existing_one_summary = _load_json(p_one)
    existing_best_summary = _load_json(p_best)
    existing_similarity_null = _load_json(p_null)
    if isinstance(existing_one_summary, dict) or isinstance(existing_best_summary, dict) or isinstance(
        existing_similarity_null, dict
    ):
        print(
            f"[INFO] Merging with existing outputs under {out_base} "
            "(only metrics with flag 1 are overwritten; 0 preserves prior JSON/CSV values)."
        )

    fgclip_evaluator = None
    if args.fg_clip_score:
        try:
            from fgclip_similarity import FGCLIPSimilarityEvaluator
        except ImportError as e:
            raise SystemExit(
                f"FG-CLIP: cannot import fgclip_similarity ({e}). expected {_FGCLIP_TOOLS / 'fgclip_similarity.py'}"
            ) from e
        mp = str(args.fgclip_model) if args.fgclip_model else None
        fgclip_evaluator = FGCLIPSimilarityEvaluator(model_path=mp, device=args.device)

    suite = InferenceMetricSuite(
        device=args.device,
        lpips_net=args.lpips_net,
        fid_batch_size=args.fid_batch_size,
        inception_batch_size=args.inception_batch_size,
        inception_splits=args.inception_splits,
    )

    one_summary, one_rows = _evaluate_mode(
        one_shot_records,
        mode="one_shot",
        suite=suite,
        args=args,
        fgclip_evaluator=fgclip_evaluator,
    )
    best_summary: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    if not args.only_one_shot:
        best_summary, best_rows = _evaluate_mode(
            best_match_records,
            mode="best_match",
            suite=suite,
            args=args,
            fgclip_evaluator=fgclip_evaluator,
        )

    one_summary = _merge_summary_with_existing(existing_one_summary, one_summary, args, mode="one_shot")
    if not args.only_one_shot and best_summary is not None:
        best_summary = _merge_summary_with_existing(existing_best_summary, best_summary, args, mode="best_match")
    else:
        best_summary = existing_best_summary if isinstance(existing_best_summary, dict) else None

    if args.write_csv:
        one_rows = _merge_per_sample_rows(
            _load_csv_rows(Path(str(out_base) + "_one_shot_per_sample.csv")),
            one_rows,
            args,
            mode="one_shot",
        )
        if not args.only_one_shot:
            best_rows = _merge_per_sample_rows(
                _load_csv_rows(Path(str(out_base) + "_best_match_per_sample.csv")),
                best_rows,
                args,
                mode="best_match",
            )

    similarity_null_by_mode = _merge_similarity_null_existing(
        existing_similarity_null, one_summary, best_summary, args
    )
    _write_outputs(out_base, "one_shot", one_summary, one_rows, write_csv=args.write_csv)
    if not args.only_one_shot and best_summary is not None:
        _write_outputs(out_base, "best_match", best_summary, best_rows, write_csv=args.write_csv)
    _write_similarity_null_split_summary(out_base, similarity_null_by_mode)

    print(
        json.dumps(
            {
                "one_shot": one_summary,
                "best_match": best_summary,
                "similarity_null_by_mode": similarity_null_by_mode,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
