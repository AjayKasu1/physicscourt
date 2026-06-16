#!/usr/bin/env python3
"""Calibrate and evaluate Detector B with per-rule calibration."""

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

from evaluate_detector_a import VIOLATION_TAXONOMY, metrics, plot_examples
from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest
from physicscourt.pipeline.score_utils import (
    first_sustained_crossing,
    load_score_cache,
    median_filter_1d,
    save_score_cache,
    summarize_scores,
)

console = Console()


def cache_path(features_dir: Path, detector: str, clip: ClipRecord) -> Path:
    return features_dir / detector / f"{clip.clip_id}.npz"


def load_rule_names(cached: dict[str, Any]) -> list[str]:
    value = cached["rule_names_json"][0]
    return json.loads(str(value))


def fit_rule_stats(detector: str, records: list[ClipRecord], features_dir: Path, kernel: int) -> dict[str, Any]:
    calibration = [record for record in records if record.split == "calibration" and record.possible]
    arrays: list[np.ndarray] = []
    rule_names: list[str] | None = None
    for record in calibration:
        cached = load_score_cache(cache_path(features_dir, detector, record))
        names = load_rule_names(cached)
        if rule_names is None:
            rule_names = names
        elif names != rule_names:
            raise ValueError(f"Rule names differ in calibration clip {record.clip_id}")
        arrays.append(np.asarray(cached["rule_scores"], dtype=np.float32))
    if not arrays:
        raise ValueError("Cannot fit Detector B calibration without calibration clips.")

    joined = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays], axis=0)
    mean = joined.mean(axis=0).astype(np.float32)
    std = joined.std(axis=0, ddof=0).astype(np.float32)
    safe_std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    return {
        "split": "calibration",
        "normal_clip_count": len(calibration),
        "smoothing_kernel": kernel,
        "calibration_mode": "per_rule_zscore_then_max",
        "rule_names": rule_names or [],
        "rule_mean": mean.astype(float).tolist(),
        "rule_std": std.astype(float).tolist(),
        "rule_safe_std": safe_std.astype(float).tolist(),
        "num_values_per_rule": int(joined.shape[0]),
        "global_rule_score_min": float(joined.min()),
        "global_rule_score_max": float(joined.max()),
    }


def apply_rule_calibration(
    detector: str,
    records: list[ClipRecord],
    features_dir: Path,
    stats: dict[str, Any],
    kernel: int,
    onset_threshold: float,
    onset_consecutive: int,
    onset_prior_below: int,
) -> list[dict[str, Any]]:
    rule_names = list(stats["rule_names"])
    mean = np.asarray(stats["rule_mean"], dtype=np.float32)
    safe_std = np.asarray(stats["rule_safe_std"], dtype=np.float32)
    rows: list[dict[str, Any]] = []

    for record in records:
        path = cache_path(features_dir, detector, record)
        cached = load_score_cache(path)
        names = load_rule_names(cached)
        if names != rule_names:
            raise ValueError(f"Rule names differ in clip {record.clip_id}")

        rule_scores = np.asarray(cached["rule_scores"], dtype=np.float32)
        rule_calibrated = ((rule_scores - mean[None, :]) / safe_std[None, :]).astype(np.float32)
        raw = np.nanmax(rule_scores, axis=1).astype(np.float32)
        calibrated = np.nanmax(rule_calibrated, axis=1).astype(np.float32)
        top_rule_index = np.nanargmax(rule_calibrated, axis=1).astype(np.int32)
        smoothed = median_filter_1d(calibrated, kernel)
        summary = summarize_scores(smoothed)
        onset_frame = first_sustained_crossing(
            smoothed,
            onset_threshold,
            onset_consecutive,
            min_prior_below=onset_prior_below,
        )
        metadata = dict(cached["metadata"])
        metadata["detector"] = detector
        metadata["calibration_mode"] = stats["calibration_mode"]
        extra = {
            key: value
            for key, value in cached.items()
            if key
            not in {
                "raw_score",
                "calibrated_score",
                "smoothed_score",
                "metadata_json",
                "metadata",
                "rule_calibrated_scores",
                "top_rule_index",
            }
        }
        extra["rule_calibrated_scores"] = rule_calibrated
        extra["top_rule_index"] = top_rule_index
        save_score_cache(
            path,
            raw_score=raw,
            calibrated_score=calibrated,
            smoothed_score=smoothed,
            metadata=metadata,
            extra_arrays=extra,
        )
        rows.append(
            {
                "clip_id": record.clip_id,
                "pair_id": record.pair_id,
                "detector": detector,
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
                "top_rule_at_argmax": rule_names[int(top_rule_index[summary.predicted_frame])],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="li_state_rules")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "detector_b_report.json")
    parser.add_argument("--calibration-report", type=Path, default=ROOT / "results" / "detector_b_calibration_report.json")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "results" / "figures" / "detector_b_eval")
    parser.add_argument("--smooth-kernel", type=int, default=5)
    parser.add_argument("--localization-tolerance", type=int, default=12)
    parser.add_argument("--onset-threshold-z", type=float, default=0.5)
    parser.add_argument("--onset-consecutive-frames", type=int, default=5)
    parser.add_argument("--onset-prior-below-frames", type=int, default=5)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    created_at = datetime.now(timezone.utc).isoformat()
    stats = fit_rule_stats(args.detector, records, args.features_dir, args.smooth_kernel)
    rows = apply_rule_calibration(
        args.detector,
        records,
        args.features_dir,
        stats,
        args.smooth_kernel,
        args.onset_threshold_z,
        args.onset_consecutive_frames,
        args.onset_prior_below_frames,
    )
    calibration_report: dict[str, Any] = {
        "created_at": created_at,
        "calibration_split": "calibration",
        "normal_only": True,
        "onset_threshold_z": args.onset_threshold_z,
        "onset_consecutive_frames": args.onset_consecutive_frames,
        "onset_prior_below_frames": args.onset_prior_below_frames,
        "detectors": {args.detector: stats},
    }
    report: dict[str, Any] = {
        "created_at": created_at,
        "localization_tolerance_frames": args.localization_tolerance,
        "onset_threshold_z": args.onset_threshold_z,
        "onset_consecutive_frames": args.onset_consecutive_frames,
        "onset_prior_below_frames": args.onset_prior_below_frames,
        "violation_taxonomy": VIOLATION_TAXONOMY,
        "detectors": {
            args.detector: {
                "calibration_mode": stats["calibration_mode"],
                "metrics": metrics(rows, args.localization_tolerance),
                "examples": plot_examples(args.detector, rows, args.features_dir, args.figures_dir),
                "rows": rows,
            }
        },
    }

    args.calibration_report.parent.mkdir(parents=True, exist_ok=True)
    with args.calibration_report.open("w", encoding="utf-8") as fh:
        json.dump(calibration_report, fh, indent=2)
        fh.write("\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    console.print(f"wrote {args.calibration_report}")
    console.print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
