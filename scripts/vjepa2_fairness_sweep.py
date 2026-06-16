#!/usr/bin/env python3
"""Fairness checks for the V-JEPA 2 Detector A harness.

This script separates two questions:

1. Did the cached V-JEPA 2 run look weak because of a scoring/mapping choice?
2. Does the live V-JEPA 2 predictor path react to temporal order and stride?

The cached analysis is cheap and covers all clips. The live inference sweep is
small by default because the target machine has 8 GB unified memory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from rich.console import Console
from rich.progress import track

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_detector_a import metrics
from physicscourt.detectors.lecun_vjepa import VJEPALatentPredictor
from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest
from physicscourt.pipeline.score_utils import (
    first_sustained_crossing,
    fit_normal_stats,
    load_score_cache,
    median_filter_1d,
    summarize_scores,
    zscore,
)
from physicscourt.pipeline.video_io import iter_video_frames_rgb

console = Console()


def cache_path(features_dir: Path, clip: ClipRecord) -> Path:
    return features_dir / "vjepa2" / f"{clip.clip_id}.npz"


def load_model_id(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return str(cfg["models"]["vjepa2"]["model_id"])


def resolve_local_model_path(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
        ref_path = repo_dir / "refs" / "main"
        if ref_path.exists():
            revision = ref_path.read_text(encoding="utf-8").strip()
            snapshot = repo_dir / "snapshots" / revision
            if snapshot.exists():
                return str(snapshot)
        raise


def mapped_indices(endpoint_indices: np.ndarray, mapping: str, window_frames: int) -> np.ndarray:
    endpoints = endpoint_indices.astype(np.float32)
    if mapping == "endpoint":
        mapped = endpoints
    elif mapping == "center":
        mapped = endpoints - float(window_frames - 1) * 0.5
    elif mapping == "target_start_approx":
        mapped = endpoints - 1.0
    else:
        raise ValueError(f"Unknown mapping {mapping}")
    return np.maximum(mapped, 0.0)


def interpolate_window_scores(
    *,
    endpoint_indices: np.ndarray,
    window_scores: np.ndarray,
    num_frames: int,
    mapping: str,
    window_frames: int,
) -> np.ndarray:
    if endpoint_indices.size == 0:
        return np.zeros((num_frames,), dtype=np.float32)
    x = mapped_indices(endpoint_indices, mapping, window_frames)
    y = window_scores.astype(np.float32)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return np.interp(np.arange(num_frames, dtype=np.float32), x, y, left=y[0], right=y[-1]).astype(np.float32)


def cached_rows_for_variant(
    *,
    records: list[ClipRecord],
    features_dir: Path,
    metric_name: str,
    mapping: str,
    window_frames: int,
    smooth_kernel: int,
    onset_threshold: float,
    onset_consecutive: int,
    onset_prior_below: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration_arrays = []
    raw_by_clip: dict[str, np.ndarray] = {}
    for record in records:
        cached = load_score_cache(cache_path(features_dir, record))
        score_key = "window_scores" if metric_name == "l2" else "window_cosine_scores"
        raw = interpolate_window_scores(
            endpoint_indices=np.asarray(cached["score_frame_indices"], dtype=np.int32),
            window_scores=np.asarray(cached[score_key], dtype=np.float32),
            num_frames=record.num_frames,
            mapping=mapping,
            window_frames=window_frames,
        )
        raw_by_clip[record.clip_id] = raw
        if record.split == "calibration" and record.possible:
            calibration_arrays.append(raw)
    stats = fit_normal_stats(calibration_arrays)
    rows = []
    for record in records:
        raw = raw_by_clip[record.clip_id]
        calibrated = zscore(raw, stats["mean"], stats["std"])
        smoothed = median_filter_1d(calibrated, smooth_kernel)
        summary = summarize_scores(smoothed)
        onset_frame = first_sustained_crossing(
            smoothed,
            onset_threshold,
            onset_consecutive,
            min_prior_below=onset_prior_below,
        )
        rows.append(
            {
                "clip_id": record.clip_id,
                "pair_id": record.pair_id,
                "detector": f"vjepa2_cached_{metric_name}_{mapping}",
                "split": record.split,
                "category": record.category,
                "possible": record.possible,
                "violation_frame": record.violation_frame,
                "clip_level_score": summary.clip_level_score,
                "predicted_frame": summary.predicted_frame,
                "argmax_frame": summary.predicted_frame,
                "onset_frame": onset_frame,
                "onset_detected": onset_frame is not None,
                "onset_threshold_z": onset_threshold,
                "onset_consecutive_frames": onset_consecutive,
                "onset_prior_below_frames": onset_prior_below,
            }
        )
    return rows, stats


def cached_variant_analysis(
    *,
    records: list[ClipRecord],
    features_dir: Path,
    window_frames: int,
    smooth_kernel: int,
    localization_tolerance: int,
    onset_threshold: float,
    onset_consecutive: int,
    onset_prior_below: int,
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for metric_name in ("l2", "cosine"):
        for mapping in ("endpoint", "center", "target_start_approx"):
            rows, stats = cached_rows_for_variant(
                records=records,
                features_dir=features_dir,
                metric_name=metric_name,
                mapping=mapping,
                window_frames=window_frames,
                smooth_kernel=smooth_kernel,
                onset_threshold=onset_threshold,
                onset_consecutive=onset_consecutive,
                onset_prior_below=onset_prior_below,
            )
            key = f"{metric_name}_{mapping}"
            variants[key] = {
                "metric": metric_name,
                "frame_mapping": mapping,
                "calibration": stats,
                "metrics": metrics(rows, localization_tolerance),
            }
    return variants


def frames_for_clip(clip: ClipRecord) -> list[Image.Image]:
    return [Image.fromarray(frame) for frame in iter_video_frames_rgb(Path(clip.video_path))]


def endpoint_indices(num_frames: int, window_frames: int, stride: int) -> list[int]:
    out = []
    for frame_index in range(window_frames - 1, num_frames):
        if (frame_index - (window_frames - 1)) % stride == 0 or frame_index == num_frames - 1:
            out.append(frame_index)
    return sorted(set(out))


def limit_indices_around_violation(indices: list[int], clip: ClipRecord, max_windows: int | None) -> list[int]:
    if max_windows is None or max_windows <= 0 or len(indices) <= max_windows:
        return indices
    target = int(clip.violation_frame if clip.violation_frame is not None else clip.num_frames // 2)
    selected = sorted(indices, key=lambda item: (abs(item - target), item))[:max_windows]
    return sorted(selected)


def score_clip_live(
    detector: VJEPALatentPredictor,
    clip: ClipRecord,
    *,
    stride: int,
    max_windows: int | None,
) -> dict[str, Any]:
    frames = frames_for_clip(clip)
    all_indices = endpoint_indices(len(frames), detector.window_frames, stride)
    indices = limit_indices_around_violation(all_indices, clip, max_windows)
    l2_scores = []
    cosine_scores = []
    started = time.perf_counter()
    for end in indices:
        window = frames[end - detector.window_frames + 1 : end + 1]
        l2, cosine = detector._score_window(window)
        l2_scores.append(l2)
        cosine_scores.append(cosine)
    return {
        "clip_id": clip.clip_id,
        "pair_id": clip.pair_id,
        "category": clip.category,
        "possible": clip.possible,
        "violation_frame": clip.violation_frame,
        "stride": stride,
        "num_windows": len(indices),
        "num_available_windows": len(all_indices),
        "live_window_subset": bool(len(indices) != len(all_indices)),
        "seconds": float(time.perf_counter() - started),
        "score_frame_indices": indices,
        "l2_window_scores": l2_scores,
        "cosine_window_scores": cosine_scores,
        "l2_clip_max": float(max(l2_scores)) if l2_scores else 0.0,
        "cosine_clip_max": float(max(cosine_scores)) if cosine_scores else 0.0,
    }


def selected_pairs(records: list[ClipRecord], categories: list[str], pairs_per_category: int) -> list[ClipRecord]:
    by_category_pair: dict[str, dict[str, list[ClipRecord]]] = {}
    for record in records:
        if record.split != "test" or record.category not in categories:
            continue
        by_category_pair.setdefault(record.category, {}).setdefault(record.pair_id, []).append(record)
    selected: list[ClipRecord] = []
    for category in categories:
        pairs = []
        for pair_records in by_category_pair.get(category, {}).values():
            if {record.possible for record in pair_records} == {True, False}:
                pairs.append(sorted(pair_records, key=lambda item: item.possible, reverse=True))
        for pair in sorted(pairs, key=lambda items: items[0].pair_id)[:pairs_per_category]:
            selected.extend(pair)
    return selected


def live_stride_sweep(
    *,
    detector: VJEPALatentPredictor,
    clips: list[ClipRecord],
    strides: list[int],
    max_windows_per_clip: int | None,
) -> dict[str, Any]:
    records = []
    for stride in strides:
        for clip in track(clips, description=f"vjepa2 live stride={stride}"):
            records.append(score_clip_live(detector, clip, stride=stride, max_windows=max_windows_per_clip))
    margins = []
    by_stride_pair: dict[tuple[int, str], dict[bool, dict[str, Any]]] = {}
    for row in records:
        by_stride_pair.setdefault((int(row["stride"]), str(row["pair_id"])), {})[bool(row["possible"])] = row
    for (stride, pair_id), pair in by_stride_pair.items():
        if True not in pair or False not in pair:
            continue
        possible = pair[True]
        impossible = pair[False]
        margins.append(
            {
                "stride": stride,
                "pair_id": pair_id,
                "category": impossible["category"],
                "l2_margin_impossible_minus_possible": float(impossible["l2_clip_max"] - possible["l2_clip_max"]),
                "cosine_margin_impossible_minus_possible": float(
                    impossible["cosine_clip_max"] - possible["cosine_clip_max"]
                ),
                "possible_clip_id": possible["clip_id"],
                "impossible_clip_id": impossible["clip_id"],
            }
        )
    return {"records": records, "pair_margins": margins}


def temporal_liveness_check(detector: VJEPALatentPredictor, clip: ClipRecord, seed: int) -> dict[str, Any]:
    frames = frames_for_clip(clip)
    t_star = int(clip.violation_frame or max(detector.window_frames - 1, len(frames) // 2))
    end = min(max(t_star + 4, detector.window_frames - 1), len(frames) - 1)
    window = frames[end - detector.window_frames + 1 : end + 1]
    rng = random.Random(seed)
    shuffled = list(window)
    rng.shuffle(shuffled)
    variants = {
        "original": window,
        "reversed": list(reversed(window)),
        "shuffled": shuffled,
        "static_first_frame": [window[0]] * len(window),
        "static_last_frame": [window[-1]] * len(window),
    }
    scores = {}
    for name, variant in variants.items():
        l2, cosine = detector._score_window(variant)
        scores[name] = {"l2": l2, "cosine": cosine}
    l2_values = [item["l2"] for item in scores.values()]
    cosine_values = [item["cosine"] for item in scores.values()]
    return {
        "clip_id": clip.clip_id,
        "category": clip.category,
        "possible": clip.possible,
        "window_end_frame": end,
        "predictor_path": "VJEPA2Model(..., skip_predictor=False).predictor_output",
        "scores": scores,
        "l2_range": float(max(l2_values) - min(l2_values)),
        "cosine_range": float(max(cosine_values) - min(cosine_values)),
        "temporal_order_changes_score": bool((max(l2_values) - min(l2_values)) > 1e-4),
    }


def compact_metric_table(variant_metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    table = {}
    for variant, payload in variant_metrics.items():
        table[variant] = {
            category: {
                "roc_auc": float(metric["roc_auc"]),
                "paired_accuracy": float(metric["paired_accuracy"]),
            }
            for category, metric in payload["metrics"].items()
        }
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--models-config", type=Path, default=ROOT / "config" / "models.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "vjepa2_fairness_report.json")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--max-live-windows-per-clip", type=int, default=2)
    parser.add_argument("--pairs-per-category", type=int, default=1)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "continuity_teleportation",
            "gravity_support",
            "momentum_causality",
            "object_permanence",
            "solidity",
            "spontaneous_vanishing",
        ],
    )
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--smooth-kernel", type=int, default=5)
    parser.add_argument("--localization-tolerance", type=int, default=12)
    parser.add_argument("--onset-threshold-z", type=float, default=0.5)
    parser.add_argument("--onset-consecutive-frames", type=int, default=5)
    parser.add_argument("--onset-prior-below-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    cached_variants = cached_variant_analysis(
        records=records,
        features_dir=args.features_dir,
        window_frames=args.window_frames,
        smooth_kernel=args.smooth_kernel,
        localization_tolerance=args.localization_tolerance,
        onset_threshold=args.onset_threshold_z,
        onset_consecutive=args.onset_consecutive_frames,
        onset_prior_below=args.onset_prior_below_frames,
    )

    live_report: dict[str, Any] | None = None
    if not args.skip_live:
        model_id = load_model_id(args.models_config)
        model_path = resolve_local_model_path(model_id)
        detector = VJEPALatentPredictor(
            model_id=model_path,
            device=args.device,
            fp16=args.fp16,
            window_frames=args.window_frames,
            stride_frames=4,
        )
        try:
            clips = selected_pairs(records, args.categories, args.pairs_per_category)
            liveness_clip = next((clip for clip in clips if not clip.possible), clips[0])
            live_report = {
                "model_id": model_id,
                "local_model_path": model_path,
                "device": detector.device.type,
                "dtype": str(detector.dtype).replace("torch.", ""),
                "window_frames": detector.window_frames,
                "temporal_liveness": temporal_liveness_check(detector, liveness_clip, args.seed),
                "stride_sweep": live_stride_sweep(
                    detector=detector,
                    clips=clips,
                    strides=args.strides,
                    max_windows_per_clip=args.max_live_windows_per_clip,
                ),
            }
        finally:
            detector.close()

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Check whether the weak V-JEPA 2 Phase 2 result is robust to cached score choice, "
            "frame mapping, temporal liveness, and small live stride sweeps before treating "
            "SSRD vs V-JEPA 2 as a settled head-to-head."
        ),
        "cached_variants": cached_variants,
        "cached_metric_table": compact_metric_table(cached_variants),
        "live_sweep": live_report,
        "notes": [
            "Cached variants cover all clips and use calibration normals only.",
            "Live stride sweep is intentionally small on the 8 GB M2 target and is not a final benchmark.",
            "A strong claim over V-JEPA 2 requires this report plus any follow-up full rerun requested by the results.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    console.print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
