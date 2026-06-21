#!/usr/bin/env python3
"""Evaluate cached V-JEPA 2 scores under different token reductions.

This script does not load V-JEPA 2. It reads caches produced by
``scripts/run_detector_a.py`` after ``src/physicscourt/detectors/lecun_vjepa.py``
has stored extra per-frame arrays such as ``series_l2_max`` and
``series_l2_topk``.

The original Detector A score remains ``raw_score`` and is reported as
``l2_mean``. New reductions answer one narrow fairness question: did spatial
mean pooling hide localized surprise that max or top-k pooling can recover?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.pipeline.score_utils import (  # noqa: E402
    load_score_cache,
    median_filter_1d,
    summarize_scores,
    zscore,
)

VARIANTS = ("l2_mean", "l2_max", "l2_topk", "cos_mean", "cos_max", "cos_topk")


def wilson_ci(hits: int, total: int, z_value: float = 1.96) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = hits / total
    denom = 1.0 + z_value * z_value / total
    center = (p + z_value * z_value / (2.0 * total)) / denom
    half = (
        z_value
        * math.sqrt((p * (1.0 - p) / total) + (z_value * z_value / (4.0 * total * total)))
        / denom
    )
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def sign_test_p(impossible_higher: int, possible_higher: int) -> float:
    total = impossible_higher + possible_higher
    if total == 0:
        return float("nan")

    def pmf(k: int) -> float:
        return math.comb(total, k) * (0.5**total)

    observed = pmf(impossible_higher)
    return float(min(1.0, sum(pmf(k) for k in range(total + 1) if pmf(k) <= observed * (1.0 + 1e-12))))


def auroc(scores: list[float], labels: list[int]) -> float:
    positive = [score for score, label in zip(scores, labels) if label == 1]
    negative = [score for score, label in zip(scores, labels) if label == 0]
    if not positive or not negative:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for rank_index in range(index, end + 1):
            ranks[order[rank_index]] = average_rank
        index = end + 1

    n_positive = len(positive)
    n_negative = len(negative)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    u_stat = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(u_stat / (n_positive * n_negative))


def series_for_variant(cached: dict[str, object], variant: str) -> np.ndarray | None:
    key = f"series_{variant}"
    if key in cached:
        return np.asarray(cached[key], dtype=np.float32)
    if variant == "l2_mean" and "raw_score" in cached:
        return np.asarray(cached["raw_score"], dtype=np.float32)
    return None


def read_caches(features_dir: Path, detector: str) -> list[dict[str, Any]]:
    cache_dir = features_dir / detector
    paths = sorted(cache_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no caches found under {cache_dir}")
    rows = []
    for path in paths:
        cached = load_score_cache(path)
        rows.append({"path": path, "cached": cached, "metadata": cached["metadata"]})
    return rows


def fit_calibration(cache_rows: list[dict[str, Any]], variant: str) -> dict[str, Any] | None:
    values = []
    for row in cache_rows:
        metadata = row["metadata"]
        if metadata.get("split") != "calibration" or not bool(metadata.get("possible")):
            continue
        series = series_for_variant(row["cached"], variant)
        if series is not None and series.size:
            values.append(series.reshape(-1))
    if not values:
        return None
    joined = np.concatenate(values).astype(np.float32)
    return {
        "mean": float(joined.mean()),
        "std": float(joined.std(ddof=0)),
        "num_values": int(joined.size),
        "min": float(joined.min()),
        "max": float(joined.max()),
    }


def score_variant(
    cache_rows: list[dict[str, Any]],
    variant: str,
    kernel: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    calibration = fit_calibration(cache_rows, variant)
    if calibration is None:
        return None

    rows = []
    for row in cache_rows:
        series = series_for_variant(row["cached"], variant)
        if series is None:
            continue
        metadata = row["metadata"]
        calibrated = zscore(series, calibration["mean"], calibration["std"])
        smoothed = median_filter_1d(calibrated, kernel)
        summary = summarize_scores(smoothed)
        rows.append(
            {
                "clip_id": metadata["clip_id"],
                "pair_id": metadata["pair_id"],
                "split": metadata["split"],
                "category": metadata["category"],
                "possible": bool(metadata["possible"]),
                "clip_level_score": summary.clip_level_score,
                "argmax_frame": summary.predicted_frame,
                "violation_frame": metadata.get("violation_frame"),
            }
        )
    return calibration, rows


def category_metrics(rows: list[dict[str, Any]], margin_epsilon: float) -> dict[str, Any]:
    by_category_pair: dict[str, dict[str, dict[bool, float]]] = defaultdict(lambda: defaultdict(dict))
    raw_by_category: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "test":
            continue
        category = str(row["category"])
        by_category_pair[category][str(row["pair_id"])][bool(row["possible"])] = float(row["clip_level_score"])
        raw_by_category[category].append((float(row["clip_level_score"]), 0 if row["possible"] else 1))

    out = {}
    for category, by_pair in sorted(by_category_pair.items()):
        impossible_higher = 0
        possible_higher = 0
        tied = 0
        margins = []
        for pair_scores in by_pair.values():
            if True not in pair_scores or False not in pair_scores:
                continue
            margin = pair_scores[False] - pair_scores[True]
            margins.append(float(margin))
            if margin > margin_epsilon:
                impossible_higher += 1
            elif margin < -margin_epsilon:
                possible_higher += 1
            else:
                tied += 1

        total = impossible_higher + possible_higher + tied
        non_tied = impossible_higher + possible_higher
        scores = [score for score, _ in raw_by_category[category]]
        labels = [label for _, label in raw_by_category[category]]
        out[category] = {
            "pairs": total,
            "counts": {
                "impossible_higher": impossible_higher,
                "possible_higher": possible_higher,
                "tied": tied,
            },
            "strict_accuracy": float(impossible_higher / total) if total else float("nan"),
            "tie_half_accuracy": float((impossible_higher + 0.5 * tied) / total) if total else float("nan"),
            "responsive_accuracy": float(impossible_higher / non_tied) if non_tied else float("nan"),
            "strict_wilson95": wilson_ci(impossible_higher, total),
            "sign_test_non_tied_p": sign_test_p(impossible_higher, possible_higher),
            "auroc": auroc(scores, labels),
            "mean_margin": float(np.mean(margins)) if margins else float("nan"),
            "median_margin": float(np.median(margins)) if margins else float("nan"),
        }
    return out


def print_matrix(report: dict[str, Any], metric: str) -> None:
    variants = [variant for variant in VARIANTS if variant in report["variants"]]
    categories = sorted(
        {
            category
            for variant_report in report["variants"].values()
            for category in variant_report["metrics_by_category"]
        }
    )
    header = "category".ljust(28) + "".join(variant.rjust(12) for variant in variants)
    print(header)
    print("-" * len(header))
    for category in categories:
        line = category.ljust(28)
        for variant in variants:
            value = report["variants"][variant]["metrics_by_category"].get(category, {}).get(metric, float("nan"))
            line += f"{value:11.3f} "
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features")
    parser.add_argument("--detector", default="vjepa2")
    parser.add_argument("--variant", choices=VARIANTS, default="l2_mean")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--metric", choices=["strict_accuracy", "tie_half_accuracy", "auroc"], default="tie_half_accuracy")
    parser.add_argument("--smooth-kernel", type=int, default=5)
    parser.add_argument("--margin-epsilon", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "vjepa2_reduction_report.json")
    args = parser.parse_args()

    cache_rows = read_caches(args.features_dir, args.detector)
    variants = VARIANTS if args.all else (args.variant,)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(args.features_dir),
        "detector": args.detector,
        "smooth_kernel": args.smooth_kernel,
        "margin_epsilon": args.margin_epsilon,
        "metric_note": "strict treats ties as misses; tie_half gives tied pairs half credit; responsive ignores ties.",
        "variants": {},
        "skipped_variants": [],
    }

    for variant in variants:
        result = score_variant(cache_rows, variant, args.smooth_kernel)
        if result is None:
            report["skipped_variants"].append(variant)
            continue
        calibration, rows = result
        report["variants"][variant] = {
            "calibration": calibration,
            "metrics_by_category": category_metrics(rows, args.margin_epsilon),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.all:
        print_matrix(report, args.metric)
        print(f"\nmetric: {args.metric}")
    else:
        variant_report = report["variants"].get(args.variant)
        if not variant_report:
            raise SystemExit(f"variant {args.variant} was not available in the cache")
        print(f"variant: {args.variant}")
        print("category".ljust(28) + "pairs  imp>  poss>  tie  strict  tiehalf  resp   auc")
        print("-" * 88)
        for category, metrics in variant_report["metrics_by_category"].items():
            counts = metrics["counts"]
            print(
                category.ljust(28)
                + f"{metrics['pairs']:5d} "
                + f"{counts['impossible_higher']:5d} "
                + f"{counts['possible_higher']:6d} "
                + f"{counts['tied']:4d} "
                + f"{metrics['strict_accuracy']:7.3f} "
                + f"{metrics['tie_half_accuracy']:8.3f} "
                + f"{metrics['responsive_accuracy']:6.3f} "
                + f"{metrics['auroc']:6.3f}"
            )

    if report["skipped_variants"]:
        print(f"\nskipped variants not found in caches: {', '.join(report['skipped_variants'])}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
