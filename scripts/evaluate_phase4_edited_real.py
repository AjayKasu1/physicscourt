#!/usr/bin/env python3
"""Evaluate edited-real pairs with frozen synthetic calibration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
    first_sustained_crossing,
    load_score_cache,
    median_filter_1d,
    save_score_cache,
    summarize_scores,
    zscore,
)

DETECTOR_LABELS = {
    "vjepa2": "V-JEPA 2 0.3B",
    "dino_latent": "DINOv2-small baseline",
    "li_state_rules": "SSRD",
}


def cache_path(features_dir: Path, detector: str, clip: ClipRecord) -> Path:
    return features_dir / detector / f"{clip.clip_id}.npz"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_rule_names(cached: dict[str, Any]) -> list[str]:
    value = cached["rule_names_json"][0]
    return json.loads(str(value))


def score_detector_a_clip(
    detector: str,
    clip: ClipRecord,
    features_dir: Path,
    stats: dict[str, Any],
    *,
    kernel: int,
    onset_threshold: float,
    onset_consecutive: int,
    onset_prior_below: int,
) -> dict[str, Any]:
    path = cache_path(features_dir, detector, clip)
    cached = load_score_cache(path)
    raw = np.asarray(cached["raw_score"], dtype=np.float32)
    calibrated = zscore(raw, float(stats["mean"]), float(stats["std"]))
    smoothed = median_filter_1d(calibrated, kernel)
    summary = summarize_scores(smoothed)
    onset = first_sustained_crossing(
        smoothed,
        onset_threshold,
        onset_consecutive,
        min_prior_below=onset_prior_below,
    )
    extra = {
        key: value
        for key, value in cached.items()
        if key not in {"raw_score", "calibrated_score", "smoothed_score", "metadata_json", "metadata"}
    }
    metadata = dict(cached["metadata"])
    metadata["phase4_calibration_source"] = "synthetic_calibration_report"
    save_score_cache(
        path,
        raw_score=raw,
        calibrated_score=calibrated,
        smoothed_score=smoothed,
        metadata=metadata,
        extra_arrays=extra,
    )
    return clip_row(detector, clip, raw, calibrated, smoothed, summary.predicted_frame, onset)


def score_ssrd_clip(
    clip: ClipRecord,
    features_dir: Path,
    stats: dict[str, Any],
    *,
    kernel: int,
    onset_threshold: float,
    onset_consecutive: int,
    onset_prior_below: int,
) -> dict[str, Any]:
    detector = "li_state_rules"
    path = cache_path(features_dir, detector, clip)
    cached = load_score_cache(path)
    rule_names = list(stats["rule_names"])
    names = load_rule_names(cached)
    if names != rule_names:
        raise ValueError(f"Rule names differ in {clip.clip_id}")

    rule_scores = np.asarray(cached["rule_scores"], dtype=np.float32)
    mean = np.asarray(stats["rule_mean"], dtype=np.float32)
    safe_std = np.asarray(stats["rule_safe_std"], dtype=np.float32)
    rule_calibrated = ((rule_scores - mean[None, :]) / safe_std[None, :]).astype(np.float32)
    raw = np.nanmax(rule_scores, axis=1).astype(np.float32)
    calibrated = np.nanmax(rule_calibrated, axis=1).astype(np.float32)
    top_rule_index = np.nanargmax(rule_calibrated, axis=1).astype(np.int32)
    smoothed = median_filter_1d(calibrated, kernel)
    summary = summarize_scores(smoothed)
    onset = first_sustained_crossing(
        smoothed,
        onset_threshold,
        onset_consecutive,
        min_prior_below=onset_prior_below,
    )
    metadata = dict(cached["metadata"])
    metadata["detector"] = detector
    metadata["phase4_calibration_source"] = "synthetic_detector_b_calibration_report"
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
    row = clip_row(detector, clip, raw, calibrated, smoothed, summary.predicted_frame, onset)
    row["top_rule_at_argmax"] = rule_names[int(top_rule_index[summary.predicted_frame])]
    return row


def clip_row(
    detector: str,
    clip: ClipRecord,
    raw: np.ndarray,
    calibrated: np.ndarray,
    smoothed: np.ndarray,
    argmax_frame: int,
    onset_frame: int | None,
) -> dict[str, Any]:
    return {
        "clip_id": clip.clip_id,
        "pair_id": clip.pair_id,
        "detector": detector,
        "detector_label": DETECTOR_LABELS.get(detector, detector),
        "category": clip.category,
        "possible": clip.possible,
        "violation_frame": clip.violation_frame,
        "raw_max": float(np.nanmax(raw)) if raw.size else 0.0,
        "calibrated_max": float(np.nanmax(calibrated)) if calibrated.size else 0.0,
        "clip_level_score": float(smoothed[argmax_frame]) if smoothed.size else 0.0,
        "argmax_frame": int(argmax_frame),
        "onset_frame": int(onset_frame) if onset_frame is not None else None,
        "onset_detected": onset_frame is not None,
    }


def pair_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_detector_pair: dict[tuple[str, str], dict[bool, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["detector"]), str(row["pair_id"]))
        by_detector_pair.setdefault(key, {})[bool(row["possible"])] = row

    results = []
    for (detector, pair_id), values in sorted(by_detector_pair.items()):
        if True not in values or False not in values:
            continue
        possible = values[True]
        impossible = values[False]
        margin = float(impossible["clip_level_score"] - possible["clip_level_score"])
        t_star = impossible.get("violation_frame")
        onset = impossible.get("onset_frame")
        argmax = impossible.get("argmax_frame")
        results.append(
            {
                "detector": detector,
                "detector_label": DETECTOR_LABELS.get(detector, detector),
                "pair_id": pair_id,
                "category": impossible["category"],
                "possible_score": float(possible["clip_level_score"]),
                "impossible_score": float(impossible["clip_level_score"]),
                "margin_impossible_minus_possible": margin,
                "pair_correct": margin > 0.0,
                "pair_tied": margin == 0.0,
                "violation_frame": t_star,
                "impossible_argmax_frame": argmax,
                "impossible_onset_frame": onset,
                "argmax_abs_error": abs(int(argmax) - int(t_star)) if t_star is not None and argmax is not None else None,
                "onset_abs_error": abs(int(onset) - int(t_star)) if t_star is not None and onset is not None else None,
                "top_rule_at_argmax": impossible.get("top_rule_at_argmax"),
            }
        )
    return results


def write_pair_plots(
    rows: list[dict[str, Any]],
    features_dir: Path,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for pair in pair_results(rows):
        detector = str(pair["detector"])
        pair_id = str(pair["pair_id"])
        possible_row = next(row for row in rows if row["detector"] == detector and row["pair_id"] == pair_id and row["possible"])
        impossible_row = next(row for row in rows if row["detector"] == detector and row["pair_id"] == pair_id and not row["possible"])
        possible_cache = load_score_cache(features_dir / detector / f"{possible_row['clip_id']}.npz")
        impossible_cache = load_score_cache(features_dir / detector / f"{impossible_row['clip_id']}.npz")
        possible_score = np.asarray(possible_cache["smoothed_score"], dtype=np.float32)
        impossible_score = np.asarray(impossible_cache["smoothed_score"], dtype=np.float32)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(possible_score, color="#2563eb", label="possible")
        ax.plot(impossible_score, color="#dc2626", label="impossible")
        if pair["violation_frame"] is not None:
            ax.axvline(int(pair["violation_frame"]), color="black", linestyle=":", label="t*")
        if pair["impossible_argmax_frame"] is not None:
            ax.axvline(int(pair["impossible_argmax_frame"]), color="#f97316", linestyle="--", label="argmax")
        if pair["impossible_onset_frame"] is not None:
            ax.axvline(int(pair["impossible_onset_frame"]), color="#16a34a", linestyle="-.", label="onset")
        ax.set_title(f"{DETECTOR_LABELS.get(detector, detector)}: {pair_id}")
        ax.set_xlabel("frame")
        ax.set_ylabel("smoothed calibrated score")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path = figures_dir / f"{detector}_{pair_id}_score_overlay.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "edited_real_processed_manifest.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features_phase4_processed")
    parser.add_argument("--detector-a-calibration", type=Path, default=ROOT / "results" / "calibration_report.json")
    parser.add_argument("--detector-b-calibration", type=Path, default=ROOT / "results" / "detector_b_calibration_report.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "phase4_edited_real_report.json")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "results" / "figures" / "phase4_edited_real")
    parser.add_argument("--detectors", nargs="+", default=["vjepa2", "dino_latent", "li_state_rules"])
    parser.add_argument("--smooth-kernel", type=int, default=5)
    parser.add_argument("--onset-threshold-z", type=float, default=0.5)
    parser.add_argument("--onset-consecutive-frames", type=int, default=5)
    parser.add_argument("--onset-prior-below-frames", type=int, default=5)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    detector_a_calibration = load_json(args.detector_a_calibration)
    detector_b_calibration = load_json(args.detector_b_calibration)
    rows: list[dict[str, Any]] = []

    for detector in args.detectors:
        if detector in {"vjepa2", "dino_latent"}:
            stats = detector_a_calibration["detectors"][detector]
            for clip in records:
                rows.append(
                    score_detector_a_clip(
                        detector,
                        clip,
                        args.features_dir,
                        stats,
                        kernel=args.smooth_kernel,
                        onset_threshold=args.onset_threshold_z,
                        onset_consecutive=args.onset_consecutive_frames,
                        onset_prior_below=args.onset_prior_below_frames,
                    )
                )
        elif detector == "li_state_rules":
            stats = detector_b_calibration["detectors"]["li_state_rules"]
            for clip in records:
                rows.append(
                    score_ssrd_clip(
                        clip,
                        args.features_dir,
                        stats,
                        kernel=args.smooth_kernel,
                        onset_threshold=args.onset_threshold_z,
                        onset_consecutive=args.onset_consecutive_frames,
                        onset_prior_below=args.onset_prior_below_frames,
                    )
                )
        else:
            raise ValueError(f"Unknown detector {detector}")

    pairs = pair_results(rows)
    figures = write_pair_plots(rows, args.features_dir, args.figures_dir)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase4_edited_real_validation",
        "manifest": str(args.manifest),
        "features_dir": str(args.features_dir),
        "calibration_policy": "frozen synthetic calibration, no edited-real refit",
        "detectors": DETECTOR_LABELS,
        "rows": rows,
        "pair_results": pairs,
        "figures": figures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"wrote {args.report}")
    for item in pairs:
        status = "correct" if item["pair_correct"] else "wrong"
        print(
            f"{item['pair_id']} | {item['detector_label']}: {status}, "
            f"possible={item['possible_score']:.3f}, "
            f"impossible={item['impossible_score']:.3f}, "
            f"margin={item['margin_impossible_minus_possible']:.3f}, "
            f"onset={item['impossible_onset_frame']}, "
            f"argmax={item['impossible_argmax_frame']}"
        )


if __name__ == "__main__":
    main()
