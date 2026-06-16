#!/usr/bin/env python3
"""Correlate detector surprise with raw motion energy in the synthetic clips."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest
from physicscourt.pipeline.video_io import iter_video_frames_rgb


MOTION_METRICS = [
    "pixel_diff_mean",
    "pixel_diff_post_t",
    "pixel_diff_around_t",
    "flow_mean",
    "flow_post_t",
    "flow_around_t",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def pair_violation_frames(records: list[ClipRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for record in records:
        if record.violation_frame is not None:
            out[str(record.pair_id)] = int(record.violation_frame)
    return out


def resized_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def transition_slices(num_transitions: int, t_star: int | None, around_radius: int) -> dict[str, np.ndarray]:
    all_indices = np.arange(num_transitions, dtype=np.int32)
    if t_star is None:
        return {
            "mean": all_indices,
            "post_t": all_indices,
            "around_t": all_indices,
        }
    post_start = max(0, int(t_star) - 1)
    around_start = max(0, int(t_star) - around_radius)
    around_end = min(num_transitions, int(t_star) + around_radius)
    return {
        "mean": all_indices,
        "post_t": np.arange(post_start, num_transitions, dtype=np.int32),
        "around_t": np.arange(around_start, around_end, dtype=np.int32),
    }


def safe_slice_mean(values: np.ndarray, indices: np.ndarray) -> float:
    if values.size == 0 or indices.size == 0:
        return float("nan")
    valid = indices[(indices >= 0) & (indices < values.size)]
    if valid.size == 0:
        return float("nan")
    return float(values[valid].mean())


def compute_motion(record: ClipRecord, *, t_star: int | None, size: int, around_radius: int) -> dict[str, Any]:
    frames = [resized_rgb(frame, size) for frame in iter_video_frames_rgb(Path(record.video_path))]
    if len(frames) < 2:
        raise ValueError(f"Need at least two frames for motion energy: {record.clip_id}")

    pixel_diffs: list[float] = []
    flow_magnitudes: list[float] = []
    prev_rgb = frames[0].astype(np.float32) / 255.0
    prev_gray = to_gray(frames[0])
    for frame in frames[1:]:
        current_rgb = frame.astype(np.float32) / 255.0
        current_gray = to_gray(frame)
        pixel_diffs.append(float(np.abs(current_rgb - prev_rgb).mean()))
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitudes.append(float(magnitude.mean()))
        prev_rgb = current_rgb
        prev_gray = current_gray

    pixel_arr = np.asarray(pixel_diffs, dtype=np.float32)
    flow_arr = np.asarray(flow_magnitudes, dtype=np.float32)
    slices = transition_slices(len(pixel_arr), t_star, around_radius)
    return {
        "clip_id": record.clip_id,
        "pair_id": record.pair_id,
        "category": record.category,
        "possible": bool(record.possible),
        "t_star_for_motion": t_star,
        "resize": size,
        "around_radius": around_radius,
        "num_frames": len(frames),
        "num_transitions": int(pixel_arr.size),
        "pixel_diff_mean": safe_slice_mean(pixel_arr, slices["mean"]),
        "pixel_diff_post_t": safe_slice_mean(pixel_arr, slices["post_t"]),
        "pixel_diff_around_t": safe_slice_mean(pixel_arr, slices["around_t"]),
        "flow_mean": safe_slice_mean(flow_arr, slices["mean"]),
        "flow_post_t": safe_slice_mean(flow_arr, slices["post_t"]),
        "flow_around_t": safe_slice_mean(flow_arr, slices["around_t"]),
    }


def motion_cache_path(cache_dir: Path, record: ClipRecord) -> Path:
    return cache_dir / f"{record.clip_id}.json"


def motion_for_records(
    records: list[ClipRecord],
    *,
    cache_dir: Path,
    size: int,
    around_radius: int,
    force: bool,
) -> dict[str, dict[str, Any]]:
    tstars = pair_violation_frames(records)
    out: dict[str, dict[str, Any]] = {}
    test_records = [record for record in records if record.split == "test"]
    for index, record in enumerate(test_records, start=1):
        path = motion_cache_path(cache_dir, record)
        if path.exists() and not force:
            payload = load_json(path)
        else:
            payload = compute_motion(
                record,
                t_star=tstars.get(str(record.pair_id)),
                size=size,
                around_radius=around_radius,
            )
            write_json(path, payload)
        out[record.clip_id] = payload
        if index % 25 == 0:
            print(f"motion cached {index}/{len(test_records)}")
    return out


def detector_rows(report_path: Path, detector: str) -> list[dict[str, Any]]:
    report = load_json(report_path)
    return [row for row in report["detectors"][detector]["rows"] if row["split"] == "test"]


def detector_score_maps(detector_a_report: Path, detector_b_report: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for detector in ("vjepa2", "dino_latent"):
        out[detector] = {row["clip_id"]: row for row in detector_rows(detector_a_report, detector)}
    if detector_b_report is not None and detector_b_report.exists():
        out["li_state_rules"] = {
            row["clip_id"]: row
            for row in detector_rows(detector_b_report, "li_state_rules")
        }
    return out


def pearson_and_spearman(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {
            "n": int(x.size),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def clip_level_correlations(
    detector_maps: dict[str, dict[str, dict[str, Any]]],
    motion: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for detector, score_map in detector_maps.items():
        out[detector] = {}
        for metric in MOTION_METRICS:
            scores = []
            energies = []
            for clip_id, row in score_map.items():
                if clip_id not in motion:
                    continue
                scores.append(float(row["clip_level_score"]))
                energies.append(float(motion[clip_id][metric]))
            out[detector][metric] = pearson_and_spearman(energies, scores)
    return out


def pair_rows_for_detector(
    score_map: dict[str, dict[str, Any]],
    motion: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[bool, tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
    for clip_id, row in score_map.items():
        if clip_id not in motion:
            continue
        grouped[str(row["pair_id"])][bool(row["possible"])] = (row, motion[clip_id])

    rows = []
    for pair_id, pair in sorted(grouped.items()):
        if True not in pair or False not in pair:
            continue
        possible_row, possible_motion = pair[True]
        impossible_row, impossible_motion = pair[False]
        output = {
            "pair_id": pair_id,
            "category": impossible_row["category"],
            "possible_clip_id": possible_row["clip_id"],
            "impossible_clip_id": impossible_row["clip_id"],
            "detector_margin_impossible_minus_possible": float(
                impossible_row["clip_level_score"] - possible_row["clip_level_score"]
            ),
            "detector_correct": bool(
                float(impossible_row["clip_level_score"]) > float(possible_row["clip_level_score"])
            ),
        }
        for metric in MOTION_METRICS:
            output[f"{metric}_margin_impossible_minus_possible"] = float(
                impossible_motion[metric] - possible_motion[metric]
            )
            output[f"{metric}_motion_ranks_impossible_higher"] = bool(
                output[f"{metric}_margin_impossible_minus_possible"] > 0.0
            )
        rows.append(output)
    return rows


def pair_margin_correlations(
    detector_maps: dict[str, dict[str, dict[str, Any]]],
    motion: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for detector, score_map in detector_maps.items():
        rows = pair_rows_for_detector(score_map, motion)
        out[detector] = {
            "overall": summarize_pair_rows(rows),
            "by_category": {},
            "rows": rows,
        }
        for category in sorted({row["category"] for row in rows}):
            out[detector]["by_category"][category] = summarize_pair_rows(
                [row for row in rows if row["category"] == category]
            )
    return out


def summarize_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    detector_margins = [float(row["detector_margin_impossible_minus_possible"]) for row in rows]
    detector_impossible_higher = int(np.sum(np.asarray(detector_margins) > 0.0)) if rows else 0
    detector_possible_higher = int(np.sum(np.asarray(detector_margins) < 0.0)) if rows else 0
    detector_tied = int(np.sum(np.asarray(detector_margins) == 0.0)) if rows else 0
    out: dict[str, Any] = {
        "num_pairs": len(rows),
        "detector_paired_accuracy": float(np.mean([row["detector_correct"] for row in rows])) if rows else float("nan"),
        "detector_tie_half_paired_accuracy": float((detector_impossible_higher + 0.5 * detector_tied) / len(rows))
        if rows
        else float("nan"),
        "mean_detector_margin": float(np.mean(detector_margins)) if rows else float("nan"),
        "detector_margin_sign_counts": {
            "impossible_higher": detector_impossible_higher,
            "possible_higher": detector_possible_higher,
            "tied": detector_tied,
        },
        "motion_metrics": {},
    }
    for metric in MOTION_METRICS:
        margin_key = f"{metric}_margin_impossible_minus_possible"
        motion_sign_key = f"{metric}_motion_ranks_impossible_higher"
        motion_margins = [float(row[margin_key]) for row in rows]
        sign_agreement = [
            np.sign(row["detector_margin_impossible_minus_possible"]) == np.sign(row[margin_key])
            for row in rows
            if row["detector_margin_impossible_minus_possible"] != 0.0 and row[margin_key] != 0.0
        ]
        out["motion_metrics"][metric] = {
            "mean_motion_margin": float(np.mean(motion_margins)) if motion_margins else float("nan"),
            "motion_only_paired_accuracy": float(np.mean([row[motion_sign_key] for row in rows]))
            if rows
            else float("nan"),
            "detector_motion_sign_agreement": float(np.mean(sign_agreement)) if sign_agreement else float("nan"),
            "correlation": pearson_and_spearman(motion_margins, detector_margins),
        }
    return out


def compact_table(report: dict[str, Any], detector: str, metric: str) -> list[dict[str, Any]]:
    rows = []
    for category, payload in report["pair_margin_correlations"][detector]["by_category"].items():
        motion = payload["motion_metrics"][metric]
        rows.append(
            {
                "category": category,
                "detector_pair_acc": payload["detector_paired_accuracy"],
                "mean_detector_margin": payload["mean_detector_margin"],
                "motion_pair_acc": motion["motion_only_paired_accuracy"],
                "mean_motion_margin": motion["mean_motion_margin"],
                "sign_agreement": motion["detector_motion_sign_agreement"],
                "spearman_r": motion["correlation"]["spearman_r"],
                "spearman_p": motion["correlation"]["spearman_p"],
            }
        )
    return rows


def print_summary(report: dict[str, Any], detector: str, metric: str) -> None:
    overall = report["pair_margin_correlations"][detector]["overall"]
    motion = overall["motion_metrics"][metric]
    print(
        f"{detector} overall paired_acc={overall['detector_paired_accuracy']:.3f} "
        f"tie_half={overall['detector_tie_half_paired_accuracy']:.3f} "
        f"{metric}_motion_acc={motion['motion_only_paired_accuracy']:.3f} "
        f"sign_agree={motion['detector_motion_sign_agreement']:.3f} "
        f"spearman={motion['correlation']['spearman_r']:.3f} p={motion['correlation']['spearman_p']:.3g}"
    )
    print("category | detector_acc | motion_acc | sign_agree | mean_detector_margin | mean_motion_margin | spearman")
    for row in compact_table(report, detector, metric):
        print(
            f"{row['category']} | "
            f"{row['detector_pair_acc']:.3f} | "
            f"{row['motion_pair_acc']:.3f} | "
            f"{row['sign_agreement']:.3f} | "
            f"{row['mean_detector_margin']:.3f} | "
            f"{row['mean_motion_margin']:.6f} | "
            f"{row['spearman_r']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--detector-a-report", type=Path, default=ROOT / "results" / "detector_a_report.json")
    parser.add_argument("--detector-b-report", type=Path, default=ROOT / "results" / "detector_b_report.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "features" / "motion_energy")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "motion_correlation_report.json")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--around-radius", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    motion = motion_for_records(
        records,
        cache_dir=args.cache_dir,
        size=args.size,
        around_radius=args.around_radius,
        force=args.force,
    )
    detector_maps = detector_score_maps(
        args.detector_a_report,
        args.detector_b_report if args.detector_b_report.exists() else None,
    )
    report = {
        "created_at": utc_now(),
        "purpose": (
            "Test whether detector surprise, especially V-JEPA 2, tracks raw motion/change energy "
            "rather than physical implausibility."
        ),
        "motion_cache_dir": str(args.cache_dir),
        "motion_metrics": MOTION_METRICS,
        "resize": args.size,
        "around_radius": args.around_radius,
        "clip_level_correlations": clip_level_correlations(detector_maps, motion),
        "pair_margin_correlations": pair_margin_correlations(detector_maps, motion),
    }
    write_json(args.output, report)
    print(f"wrote {args.output}")
    print_summary(report, "vjepa2", "pixel_diff_post_t")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))
    main()
