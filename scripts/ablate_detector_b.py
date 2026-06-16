#!/usr/bin/env python3
"""Detector B scoring ablations from cached rule channels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
MPL_CONFIG_DIR = ROOT / "results" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_detector_a import metrics
from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest
from physicscourt.pipeline.score_utils import (
    first_sustained_crossing,
    fit_normal_stats,
    load_score_cache,
    median_filter_1d,
    summarize_scores,
    zscore,
)

console = Console()

DETECTOR = "li_state_rules"
RULE_PREFIX_BY_CATEGORY = {
    "continuity_teleportation": "continuity_",
    "gravity_support": "gravity_",
    "momentum_causality": "momentum_",
    "object_permanence": "permanence_",
    "solidity": "solidity_",
    "spontaneous_vanishing": "vanishing_",
}


def cache_path(features_dir: Path, clip: ClipRecord) -> Path:
    return features_dir / DETECTOR / f"{clip.clip_id}.npz"


def load_rule_cache(features_dir: Path, clip: ClipRecord) -> tuple[np.ndarray, list[str]]:
    cached = load_score_cache(cache_path(features_dir, clip))
    rule_scores = np.asarray(cached["rule_scores"], dtype=np.float32)
    rule_names = json.loads(str(cached["rule_names_json"][0]))
    return rule_scores, rule_names


def category_rule_indices(category: str, rule_names: list[str]) -> list[int]:
    prefix = RULE_PREFIX_BY_CATEGORY[category]
    indices = [idx for idx, name in enumerate(rule_names) if name.startswith(prefix)]
    if not indices:
        raise ValueError(f"No rules found for category {category}")
    return indices


def raw_score_for_mode(mode: str, clip: ClipRecord, rule_scores: np.ndarray, rule_names: list[str]) -> np.ndarray:
    if mode == "category_conditioned_oracle":
        indices = category_rule_indices(clip.category, rule_names)
        return np.nanmax(rule_scores[:, indices], axis=1).astype(np.float32)
    if mode == "category_blind_raw_max":
        return np.nanmax(rule_scores, axis=1).astype(np.float32)
    raise ValueError(f"{mode} does not use a raw-score global calibration path")


def fit_global_stats(mode: str, records: list[ClipRecord], features_dir: Path) -> dict[str, Any]:
    calibration = [record for record in records if record.split == "calibration" and record.possible]
    arrays = []
    for record in calibration:
        rule_scores, rule_names = load_rule_cache(features_dir, record)
        arrays.append(raw_score_for_mode(mode, record, rule_scores, rule_names))
    stats = fit_normal_stats(arrays)
    stats["calibration_mode"] = mode
    stats["normal_clip_count"] = len(calibration)
    stats["split"] = "calibration"
    return stats


def fit_rule_stats(records: list[ClipRecord], features_dir: Path) -> dict[str, Any]:
    calibration = [record for record in records if record.split == "calibration" and record.possible]
    arrays = []
    rule_names: list[str] | None = None
    for record in calibration:
        rule_scores, names = load_rule_cache(features_dir, record)
        if rule_names is None:
            rule_names = names
        elif names != rule_names:
            raise ValueError(f"Rule names differ for {record.clip_id}")
        arrays.append(rule_scores)
    joined = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays], axis=0)
    mean = joined.mean(axis=0).astype(np.float32)
    std = joined.std(axis=0, ddof=0).astype(np.float32)
    return {
        "calibration_mode": "category_blind_per_rule_calibrated",
        "normal_clip_count": len(calibration),
        "split": "calibration",
        "rule_names": rule_names or [],
        "rule_mean": mean.astype(float).tolist(),
        "rule_std": std.astype(float).tolist(),
        "rule_safe_std": np.where(std > 1e-8, std, 1.0).astype(float).tolist(),
        "num_values_per_rule": int(joined.shape[0]),
    }


def fit_rule_robust_mad_stats(records: list[ClipRecord], features_dir: Path) -> dict[str, Any]:
    """Fit per-rule robust calibration from calibration normals only."""
    calibration = [record for record in records if record.split == "calibration" and record.possible]
    arrays = []
    rule_names: list[str] | None = None
    for record in calibration:
        rule_scores, names = load_rule_cache(features_dir, record)
        if rule_names is None:
            rule_names = names
        elif names != rule_names:
            raise ValueError(f"Rule names differ for {record.clip_id}")
        arrays.append(rule_scores)
    joined = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays], axis=0)
    median = np.nanmedian(joined, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(joined - median[None, :]), axis=0).astype(np.float32)
    robust_scale = (1.4826 * mad).astype(np.float32)
    safe_scale = np.where(robust_scale > 1e-8, robust_scale, 1.0).astype(np.float32)
    return {
        "calibration_mode": "category_blind_per_rule_robust_mad_calibrated",
        "normal_clip_count": len(calibration),
        "split": "calibration",
        "rule_names": rule_names or [],
        "rule_median": median.astype(float).tolist(),
        "rule_mad": mad.astype(float).tolist(),
        "rule_robust_scale": robust_scale.astype(float).tolist(),
        "rule_safe_scale": safe_scale.astype(float).tolist(),
        "num_values_per_rule": int(joined.shape[0]),
        "scale_note": "scale = 1.4826 * MAD, fit on calibration-normal frames only; safe scale falls back to 1.0 for zero-MAD channels",
    }


def score_clip_for_mode(
    *,
    mode: str,
    clip: ClipRecord,
    rule_scores: np.ndarray,
    rule_names: list[str],
    stats: dict[str, Any],
) -> np.ndarray:
    if mode in {"category_conditioned_oracle", "category_blind_raw_max"}:
        raw = raw_score_for_mode(mode, clip, rule_scores, rule_names)
        return zscore(raw, float(stats["mean"]), float(stats["std"]))
    if mode == "category_blind_per_rule_calibrated":
        mean = np.asarray(stats["rule_mean"], dtype=np.float32)
        safe_std = np.asarray(stats["rule_safe_std"], dtype=np.float32)
        calibrated_rules = ((rule_scores - mean[None, :]) / safe_std[None, :]).astype(np.float32)
        return np.nanmax(calibrated_rules, axis=1).astype(np.float32)
    if mode == "category_blind_per_rule_robust_mad_calibrated":
        median = np.asarray(stats["rule_median"], dtype=np.float32)
        safe_scale = np.asarray(stats["rule_safe_scale"], dtype=np.float32)
        calibrated_rules = ((rule_scores - median[None, :]) / safe_scale[None, :]).astype(np.float32)
        return np.nanmax(calibrated_rules, axis=1).astype(np.float32)
    raise ValueError(f"Unknown ablation mode {mode}")


def rows_for_mode(
    *,
    mode: str,
    records: list[ClipRecord],
    features_dir: Path,
    stats: dict[str, Any],
    smooth_kernel: int,
    onset_threshold: float,
    onset_consecutive: int,
    onset_prior_below: int,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rule_scores, rule_names = load_rule_cache(features_dir, record)
        calibrated = score_clip_for_mode(
            mode=mode,
            clip=record,
            rule_scores=rule_scores,
            rule_names=rule_names,
            stats=stats,
        )
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
                "detector": f"{DETECTOR}:{mode}",
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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "detector_b_ablation_report.json")
    parser.add_argument("--smooth-kernel", type=int, default=5)
    parser.add_argument("--localization-tolerance", type=int, default=12)
    parser.add_argument("--onset-threshold-z", type=float, default=0.5)
    parser.add_argument("--onset-consecutive-frames", type=int, default=5)
    parser.add_argument("--onset-prior-below-frames", type=int, default=5)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    modes = [
        "category_conditioned_oracle",
        "category_blind_raw_max",
        "category_blind_per_rule_calibrated",
        "category_blind_per_rule_robust_mad_calibrated",
    ]
    calibrations: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for mode in modes:
        if mode == "category_blind_per_rule_calibrated":
            stats = fit_rule_stats(records, args.features_dir)
        elif mode == "category_blind_per_rule_robust_mad_calibrated":
            stats = fit_rule_robust_mad_stats(records, args.features_dir)
        else:
            stats = fit_global_stats(mode, records, args.features_dir)
        rows = rows_for_mode(
            mode=mode,
            records=records,
            features_dir=args.features_dir,
            stats=stats,
            smooth_kernel=args.smooth_kernel,
            onset_threshold=args.onset_threshold_z,
            onset_consecutive=args.onset_consecutive_frames,
            onset_prior_below=args.onset_prior_below_frames,
        )
        calibrations[mode] = stats
        reports[mode] = {
            "metrics": metrics(rows, args.localization_tolerance),
            "rows": rows,
        }

    summary_rows: list[dict[str, Any]] = []
    categories = sorted(reports[modes[0]]["metrics"])
    for category in categories:
        row: dict[str, Any] = {"category": category}
        for mode in modes:
            metric = reports[mode]["metrics"][category]
            preferred = metric.get("preferred_localization", "argmax")
            row[mode] = {
                "roc_auc": float(metric["roc_auc"]),
                "paired_accuracy": float(metric["paired_accuracy"]),
                "preferred_localization_within_12": float(metric[f"{preferred}_localization_within_tolerance"]),
                "preferred_localization": preferred,
            }
        summary_rows.append(row)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": DETECTOR,
        "protocol_note": (
            "All ablations reuse the same cached Detector B object-state/rule channels. "
            "The category-conditioned oracle uses the ground-truth category only to select a rule family; "
            "the category-blind modes use no category label at scoring time."
        ),
        "calibrations": calibrations,
        "summary_rows": summary_rows,
        "modes": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    console.print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
