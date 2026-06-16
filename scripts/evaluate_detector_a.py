#!/usr/bin/env python3
"""Calibrate and evaluate Detector A cached scores."""

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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MPL_CONFIG_DIR = ROOT / "results" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest
from physicscourt.pipeline.score_utils import (
    fit_normal_stats,
    first_sustained_crossing,
    load_score_cache,
    median_filter_1d,
    save_score_cache,
    summarize_scores,
    zscore,
)

console = Console()

VIOLATION_TAXONOMY: dict[str, dict[str, str]] = {
    "continuity_teleportation": {
        "violation_type": "instantaneous",
        "preferred_localization": "argmax",
        "note": "A discontinuous jump is a brief event, so the peak surprise frame is meaningful.",
    },
    "gravity_support": {
        "violation_type": "sustained",
        "preferred_localization": "onset",
        "note": "Unsupported floating persists after the support edge; onset is more informative than plateau peak.",
    },
    "momentum_causality": {
        "violation_type": "instantaneous_dynamic",
        "preferred_localization": "argmax",
        "note": "The violation is a sudden causal/directional transition in motion.",
    },
    "object_permanence": {
        "violation_type": "sustained",
        "preferred_localization": "onset",
        "note": "The missing reappearance becomes an ongoing absence, so sustained surprise should localize by onset.",
    },
    "solidity": {
        "violation_type": "sustained_contact",
        "preferred_localization": "onset",
        "note": "Wall penetration starts at contact and may remain anomalous over subsequent frames.",
    },
    "spontaneous_vanishing": {
        "violation_type": "instantaneous",
        "preferred_localization": "argmax",
        "note": "The disappearance itself is a sudden event, even if absence persists afterward.",
    },
}


def cache_path(features_dir: Path, detector: str, clip: ClipRecord) -> Path:
    return features_dir / detector / f"{clip.clip_id}.npz"


def calibrate(detector: str, records: list[ClipRecord], features_dir: Path, kernel: int) -> dict[str, Any]:
    calibration = [record for record in records if record.split == "calibration" and record.possible]
    arrays = []
    for record in calibration:
        cached = load_score_cache(cache_path(features_dir, detector, record))
        arrays.append(cached["raw_score"])
    stats = fit_normal_stats(arrays)
    stats["split"] = "calibration"
    stats["normal_clip_count"] = len(calibration)
    stats["smoothing_kernel"] = kernel
    return stats


def apply_calibration(
    detector: str,
    records: list[ClipRecord],
    features_dir: Path,
    stats: dict[str, float],
    kernel: int,
    onset_threshold: float,
    onset_consecutive: int,
    onset_prior_below: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        path = cache_path(features_dir, detector, record)
        cached = load_score_cache(path)
        raw = cached["raw_score"]
        calibrated = zscore(raw, stats["mean"], stats["std"])
        smoothed = median_filter_1d(calibrated, kernel)
        summary = summarize_scores(smoothed)
        onset_frame = first_sustained_crossing(
            smoothed,
            onset_threshold,
            onset_consecutive,
            min_prior_below=onset_prior_below,
        )
        metadata = cached["metadata"]
        extra = {key: value for key, value in cached.items() if key not in {"raw_score", "calibrated_score", "smoothed_score", "metadata_json", "metadata"}}
        save_score_cache(path, raw_score=raw, calibrated_score=calibrated, smoothed_score=smoothed, metadata=metadata, extra_arrays=extra)
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
            }
        )
    return rows


def paired_accuracy(rows: list[dict[str, Any]], category: str) -> float:
    by_pair: dict[str, dict[bool, float]] = {}
    for row in rows:
        if row["split"] != "test" or row["category"] != category:
            continue
        by_pair.setdefault(row["pair_id"], {})[bool(row["possible"])] = float(row["clip_level_score"])
    hits = 0
    total = 0
    for values in by_pair.values():
        if True in values and False in values:
            total += 1
            hits += float(values[False] > values[True])
    return hits / total if total else float("nan")


def localization_stats(
    impossible_rows: list[dict[str, Any]],
    *,
    frame_key: str,
    tolerance: int,
    missing_counts_as_miss: bool = False,
) -> dict[str, Any]:
    errors: list[float] = []
    within = 0
    detected = 0
    total = len(impossible_rows)
    for row in impossible_rows:
        frame = row.get(frame_key)
        if frame is None:
            if missing_counts_as_miss:
                errors.append(float("nan"))
            continue
        detected += 1
        error = abs(int(frame) - int(row["violation_frame"]))
        errors.append(float(error))
        within += int(error <= tolerance)

    finite_errors = np.array([error for error in errors if np.isfinite(error)], dtype=np.float32)
    denominator = total if missing_counts_as_miss else detected
    return {
        "median_abs_error": float(np.median(finite_errors)) if finite_errors.size else float("nan"),
        "within_tolerance": float(within / denominator) if denominator else float("nan"),
        "detected_rate": float(detected / total) if total else float("nan"),
        "num_detected": int(detected),
        "num_total": int(total),
    }


def metrics(rows: list[dict[str, Any]], tolerance: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    categories = sorted({row["category"] for row in rows if row["split"] == "test"})
    for category in categories:
        subset = [row for row in rows if row["split"] == "test" and row["category"] == category]
        y_true = np.array([not row["possible"] for row in subset], dtype=np.int32)
        y_score = np.array([row["clip_level_score"] for row in subset], dtype=np.float32)
        impossible = [row for row in subset if not row["possible"]]
        argmax_loc = localization_stats(impossible, frame_key="argmax_frame", tolerance=tolerance)
        onset_loc = localization_stats(
            impossible,
            frame_key="onset_frame",
            tolerance=tolerance,
            missing_counts_as_miss=True,
        )
        taxonomy = VIOLATION_TAXONOMY.get(category, {})
        out[category] = {
            "violation_type": taxonomy.get("violation_type", "unspecified"),
            "preferred_localization": taxonomy.get("preferred_localization", "argmax"),
            "violation_type_note": taxonomy.get("note", ""),
            "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) == 2 else float("nan"),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "paired_accuracy": float(paired_accuracy(rows, category)),
            "argmax_localization_median_abs_error": argmax_loc["median_abs_error"],
            "argmax_localization_within_tolerance": argmax_loc["within_tolerance"],
            "onset_localization_median_abs_error": onset_loc["median_abs_error"],
            "onset_localization_within_tolerance": onset_loc["within_tolerance"],
            "onset_detected_rate": onset_loc["detected_rate"],
            "onset_num_detected": onset_loc["num_detected"],
            "localization_median_abs_error": argmax_loc["median_abs_error"],
            "localization_within_tolerance": argmax_loc["within_tolerance"],
            "num_test_clips": int(len(subset)),
            "num_impossible": int(len(impossible)),
        }
    return out


def plot_examples(detector: str, rows: list[dict[str, Any]], features_dir: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = [row for row in rows if row["split"] == "test"]
    impossible = [row for row in test_rows if not row["possible"]]
    possible = [row for row in test_rows if row["possible"]]

    def preferred_error(row: dict[str, Any]) -> float:
        taxonomy = VIOLATION_TAXONOMY.get(str(row["category"]), {})
        preferred = taxonomy.get("preferred_localization", "argmax")
        frame = row.get("onset_frame") if preferred == "onset" else row.get("argmax_frame")
        if frame is None:
            return float("inf")
        return float(abs(int(frame) - int(row["violation_frame"])))

    hit = min(
        impossible,
        key=preferred_error,
    )
    miss = max(
        impossible,
        key=preferred_error,
    )
    possible_twin = next((row for row in possible if row["pair_id"] == hit["pair_id"]), possible[0])
    examples = [("clear_hit", hit), ("miss", miss), ("possible_twin", possible_twin)]
    paths: list[str] = []
    for label, row in examples:
        cached = load_score_cache(features_dir / detector / f"{row['clip_id']}.npz")
        raw = cached["raw_score"]
        calibrated = cached["calibrated_score"]
        smoothed = cached["smoothed_score"]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(raw, color="#8aa1ff", alpha=0.35, label="raw")
        ax.plot(calibrated, color="#7dd3fc", alpha=0.65, label="calibrated")
        ax.plot(smoothed, color="#f97316", linewidth=2, label="smoothed")
        if row["violation_frame"] is not None:
            ax.axvline(int(row["violation_frame"]), color="#ef4444", linestyle="--", label="t*")
        ax.axvline(int(row["argmax_frame"]), color="#22c55e", linestyle=":", label="argmax")
        if row.get("onset_frame") is not None:
            ax.axvline(int(row["onset_frame"]), color="#a855f7", linestyle="-.", label="onset")
        ax.set_title(f"{detector} {label}: {row['clip_id']}")
        ax.set_xlabel("frame")
        ax.set_ylabel("score")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path = output_dir / f"{detector}_{label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+", default=["vjepa2", "dino_latent"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "detector_a_report.json")
    parser.add_argument("--calibration-report", type=Path, default=ROOT / "results" / "calibration_report.json")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "results" / "figures" / "detector_a")
    parser.add_argument("--smooth-kernel", type=int, default=5)
    parser.add_argument("--localization-tolerance", type=int, default=12)
    parser.add_argument("--onset-threshold-z", type=float, default=0.5)
    parser.add_argument("--onset-consecutive-frames", type=int, default=5)
    parser.add_argument("--onset-prior-below-frames", type=int, default=5)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    calibration_report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_split": "calibration",
        "normal_only": True,
        "onset_threshold_z": args.onset_threshold_z,
        "onset_consecutive_frames": args.onset_consecutive_frames,
        "onset_prior_below_frames": args.onset_prior_below_frames,
        "detectors": {},
    }
    report: dict[str, Any] = {
        "created_at": calibration_report["created_at"],
        "localization_tolerance_frames": args.localization_tolerance,
        "onset_threshold_z": args.onset_threshold_z,
        "onset_consecutive_frames": args.onset_consecutive_frames,
        "onset_prior_below_frames": args.onset_prior_below_frames,
        "violation_taxonomy": VIOLATION_TAXONOMY,
        "detectors": {},
    }
    for detector in args.detectors:
        stats = calibrate(detector, records, args.features_dir, args.smooth_kernel)
        calibration_report["detectors"][detector] = stats
        rows = apply_calibration(
            detector,
            records,
            args.features_dir,
            stats,
            args.smooth_kernel,
            args.onset_threshold_z,
            args.onset_consecutive_frames,
            args.onset_prior_below_frames,
        )
        report["detectors"][detector] = {
            "metrics": metrics(rows, args.localization_tolerance),
            "examples": plot_examples(detector, rows, args.features_dir, args.figures_dir),
            "rows": rows,
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
