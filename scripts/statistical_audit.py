#!/usr/bin/env python3
"""Bootstrap uncertainty and complementarity statistics for cached detector scores."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def detector_rows(report: dict[str, Any], detector: str) -> list[dict[str, Any]]:
    payload = report["detectors"][detector]
    return list(payload["rows"])


def test_rows(rows: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == "test" and row["category"] == category]


def rows_by_pair(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["pair_id"])].append(row)
    return dict(out)


def score_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray([not bool(row["possible"]) for row in rows], dtype=np.int32)
    y_score = np.asarray([float(row["clip_level_score"]) for row in rows], dtype=np.float64)
    return y_true, y_score


def auc_for_rows(rows: list[dict[str, Any]]) -> float:
    y_true, y_score = score_arrays(rows)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def percentile_ci(values: list[float], confidence: float = 0.95) -> dict[str, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"low": float("nan"), "high": float("nan")}
    alpha = (1.0 - confidence) * 0.5
    return {
        "low": float(np.quantile(finite, alpha)),
        "high": float(np.quantile(finite, 1.0 - alpha)),
    }


def pair_stratified_bootstrap_auc(
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    by_pair = rows_by_pair(rows)
    pair_ids = sorted(by_pair)
    observed = auc_for_rows(rows)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        sampled_ids = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sampled_rows: list[dict[str, Any]] = []
        for pair_id in sampled_ids:
            sampled_rows.extend(by_pair[str(pair_id)])
        samples.append(auc_for_rows(sampled_rows))
    ci = percentile_ci(samples)
    return {
        "observed": observed,
        "ci95_low": ci["low"],
        "ci95_high": ci["high"],
        "n_bootstrap": n_bootstrap,
        "bootstrap_unit": "possible/impossible pair",
        "num_pairs": len(pair_ids),
        "num_clips": len(rows),
    }


def pair_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pair_id, pair_rows in rows_by_pair(rows).items():
        possible = next((row for row in pair_rows if bool(row["possible"])), None)
        impossible = next((row for row in pair_rows if not bool(row["possible"])), None)
        if possible is None or impossible is None:
            continue
        out[pair_id] = {
            "pair_id": pair_id,
            "category": impossible["category"],
            "possible_clip_id": possible["clip_id"],
            "impossible_clip_id": impossible["clip_id"],
            "possible_score": float(possible["clip_level_score"]),
            "impossible_score": float(impossible["clip_level_score"]),
            "margin_impossible_minus_possible": float(
                impossible["clip_level_score"] - possible["clip_level_score"]
            ),
            "correct": bool(float(impossible["clip_level_score"]) > float(possible["clip_level_score"])),
        }
    return out


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1))
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def complementarity(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    *,
    a_name: str,
    b_name: str,
) -> dict[str, Any]:
    a_pairs = pair_scores(a_rows)
    b_pairs = pair_scores(b_rows)
    shared = sorted(set(a_pairs) & set(b_pairs))
    counts = {
        "both_correct": 0,
        f"{a_name}_only": 0,
        f"{b_name}_only": 0,
        "neither_correct": 0,
    }
    examples: dict[str, list[str]] = {key: [] for key in counts}
    for pair_id in shared:
        a_correct = bool(a_pairs[pair_id]["correct"])
        b_correct = bool(b_pairs[pair_id]["correct"])
        if a_correct and b_correct:
            key = "both_correct"
        elif a_correct and not b_correct:
            key = f"{a_name}_only"
        elif b_correct and not a_correct:
            key = f"{b_name}_only"
        else:
            key = "neither_correct"
        counts[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(pair_id)
    a_only = counts[f"{a_name}_only"]
    b_only = counts[f"{b_name}_only"]
    total = len(shared)
    a_margins = np.asarray([float(a_pairs[pair_id]["margin_impossible_minus_possible"]) for pair_id in shared])
    b_margins = np.asarray([float(b_pairs[pair_id]["margin_impossible_minus_possible"]) for pair_id in shared])
    a_impossible_higher = int(np.sum(a_margins > 0.0))
    a_possible_higher = int(np.sum(a_margins < 0.0))
    a_tied = int(np.sum(a_margins == 0.0))
    b_impossible_higher = int(np.sum(b_margins > 0.0))
    b_possible_higher = int(np.sum(b_margins < 0.0))
    b_tied = int(np.sum(b_margins == 0.0))
    return {
        "comparison": f"{a_name}_vs_{b_name}",
        "unit": "possible/impossible pair",
        "counts": counts,
        "rates": {key: float(value / total) if total else float("nan") for key, value in counts.items()},
        "margin_sign_counts": {
            a_name: {
                "impossible_higher": a_impossible_higher,
                "possible_higher": a_possible_higher,
                "tied": a_tied,
            },
            b_name: {
                "impossible_higher": b_impossible_higher,
                "possible_higher": b_possible_higher,
                "tied": b_tied,
            },
        },
        "num_pairs": total,
        "mcnemar": {
            "a_only": a_only,
            "b_only": b_only,
            "discordant": a_only + b_only,
            "exact_two_sided_p": exact_mcnemar_pvalue(a_only, b_only),
            "null": "Both detectors have equal pair-level error rate on the shared pairs.",
        },
        "pair_accuracy": {
            a_name: float((counts["both_correct"] + a_only) / total) if total else float("nan"),
            b_name: float((counts["both_correct"] + b_only) / total) if total else float("nan"),
            f"{a_name}_tie_half": float((a_impossible_higher + 0.5 * a_tied) / total) if total else float("nan"),
            f"{b_name}_tie_half": float((b_impossible_higher + 0.5 * b_tied) / total) if total else float("nan"),
            "label_aware_oracle": float((counts["both_correct"] + a_only + b_only) / total) if total else float("nan"),
        },
        "example_pair_ids": examples,
    }


def rank01(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / float(values.size - 1)


def label_aware_oracle_auc(
    detector_rows_by_name: dict[str, list[dict[str, Any]]],
    category: str,
) -> dict[str, Any]:
    """Upper-bound AUC if an oracle could choose the best detector per labeled clip."""
    by_clip: dict[str, dict[str, Any]] = {}
    ranked_scores: dict[str, dict[str, float]] = {}
    for detector, rows in detector_rows_by_name.items():
        subset = test_rows(rows, category)
        scores = np.asarray([float(row["clip_level_score"]) for row in subset], dtype=np.float64)
        ranks = rank01(scores)
        ranked_scores[detector] = {str(row["clip_id"]): float(rank) for row, rank in zip(subset, ranks, strict=True)}
        for row in subset:
            by_clip.setdefault(str(row["clip_id"]), row)

    oracle_rows = []
    detectors = sorted(detector_rows_by_name)
    for clip_id, row in by_clip.items():
        available = [ranked_scores[name][clip_id] for name in detectors if clip_id in ranked_scores[name]]
        if not available:
            continue
        oracle_score = max(available) if not bool(row["possible"]) else min(available)
        oracle_rows.append({**row, "clip_level_score": oracle_score})
    return {
        "roc_auc": auc_for_rows(oracle_rows),
        "score_space": "within-category rank-normalized detector scores",
        "oracle_definition": (
            "Label-aware upper bound: impossible clips receive the max rank score across detectors, "
            "possible clips receive the min rank score. This is not a deployable detector."
        ),
    }


def build_report(
    *,
    detector_a_report: Path,
    detector_b_report: Path,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    a_report = load_json(detector_a_report)
    b_report = load_json(detector_b_report)
    rows_by_detector = {
        "vjepa2": detector_rows(a_report, "vjepa2"),
        "dino_latent": detector_rows(a_report, "dino_latent"),
        "li_state_rules": detector_rows(b_report, "li_state_rules"),
    }
    categories = sorted(
        {
            row["category"]
            for rows in rows_by_detector.values()
            for row in rows
            if row["split"] == "test"
        }
    )
    rng = np.random.default_rng(seed)
    auc_ci: dict[str, dict[str, Any]] = {}
    for detector, rows in rows_by_detector.items():
        auc_ci[detector] = {}
        for category in categories:
            auc_ci[detector][category] = pair_stratified_bootstrap_auc(
                test_rows(rows, category),
                rng=rng,
                n_bootstrap=n_bootstrap,
            )

    complementarity_by_category = {}
    for category in categories:
        complementarity_by_category[category] = {
            "vjepa2_vs_ssrd": complementarity(
                test_rows(rows_by_detector["vjepa2"], category),
                test_rows(rows_by_detector["li_state_rules"], category),
                a_name="vjepa2",
                b_name="ssrd",
            ),
            "dino_vs_ssrd": complementarity(
                test_rows(rows_by_detector["dino_latent"], category),
                test_rows(rows_by_detector["li_state_rules"], category),
                a_name="dino",
                b_name="ssrd",
            ),
            "oracle_ensemble_auc": label_aware_oracle_auc(
                {
                    "vjepa2": rows_by_detector["vjepa2"],
                    "li_state_rules": rows_by_detector["li_state_rules"],
                },
                category,
            ),
        }

    overall = {
        "vjepa2_vs_ssrd": complementarity(
            [row for row in rows_by_detector["vjepa2"] if row["split"] == "test"],
            [row for row in rows_by_detector["li_state_rules"] if row["split"] == "test"],
            a_name="vjepa2",
            b_name="ssrd",
        ),
        "dino_vs_ssrd": complementarity(
            [row for row in rows_by_detector["dino_latent"] if row["split"] == "test"],
            [row for row in rows_by_detector["li_state_rules"] if row["split"] == "test"],
            a_name="dino",
            b_name="ssrd",
        ),
    }
    all_oracle_rows = {
        name: [row for row in rows if row["split"] == "test"]
        for name, rows in rows_by_detector.items()
        if name in {"vjepa2", "li_state_rules"}
    }
    overall["oracle_ensemble_auc"] = label_aware_oracle_auc_for_rows(all_oracle_rows)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "detectors": {
            "vjepa2": "V-JEPA 2 latent-prediction detector",
            "dino_latent": "DINOv2 latent-extrapolation baseline",
            "li_state_rules": "SSRD: Spatial-State Rule Detector",
        },
        "notes": {
            "bootstrap": "AUC confidence intervals resample possible/impossible pairs with replacement within each category.",
            "complementarity": "A pair is correct when the impossible clip outscores its possible twin.",
            "mcnemar": "Exact two-sided McNemar test uses pair-level discordant correctness counts.",
            "oracle": (
                "The oracle ensemble is a label-aware upper bound, not a deployable method; it measures "
                "whether the two detectors carry complementary signal."
            ),
        },
        "auc_ci95_by_detector_category": auc_ci,
        "complementarity_by_category": complementarity_by_category,
        "overall_complementarity": overall,
    }


def label_aware_oracle_auc_for_rows(detector_rows_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_clip: dict[str, dict[str, Any]] = {}
    ranked_scores: dict[str, dict[str, float]] = {}
    for detector, rows in detector_rows_by_name.items():
        subset = [row for row in rows if row["split"] == "test"]
        scores = np.asarray([float(row["clip_level_score"]) for row in subset], dtype=np.float64)
        ranks = rank01(scores)
        ranked_scores[detector] = {str(row["clip_id"]): float(rank) for row, rank in zip(subset, ranks, strict=True)}
        for row in subset:
            by_clip.setdefault(str(row["clip_id"]), row)

    oracle_rows = []
    detectors = sorted(detector_rows_by_name)
    for clip_id, row in by_clip.items():
        available = [ranked_scores[name][clip_id] for name in detectors if clip_id in ranked_scores[name]]
        if not available:
            continue
        oracle_score = max(available) if not bool(row["possible"]) else min(available)
        oracle_rows.append({**row, "clip_level_score": oracle_score})
    return {
        "roc_auc": auc_for_rows(oracle_rows),
        "score_space": "overall rank-normalized detector scores",
        "oracle_definition": (
            "Label-aware upper bound: impossible clips receive the max rank score across detectors, "
            "possible clips receive the min rank score. This is not a deployable detector."
        ),
    }


def compact_console_summary(report: dict[str, Any]) -> str:
    lines = ["category | V-JEPA2 AUC [95% CI] | SSRD AUC [95% CI] | V-only | S-only | McNemar p"]
    for category, comp in report["complementarity_by_category"].items():
        v = report["auc_ci95_by_detector_category"]["vjepa2"][category]
        s = report["auc_ci95_by_detector_category"]["li_state_rules"][category]
        counts = comp["vjepa2_vs_ssrd"]["counts"]
        mcnemar = comp["vjepa2_vs_ssrd"]["mcnemar"]
        lines.append(
            f"{category} | "
            f"{v['observed']:.3f} [{v['ci95_low']:.3f}, {v['ci95_high']:.3f}] | "
            f"{s['observed']:.3f} [{s['ci95_low']:.3f}, {s['ci95_high']:.3f}] | "
            f"{counts['vjepa2_only']} | {counts['ssrd_only']} | {mcnemar['exact_two_sided_p']:.4f}"
        )
    overall = report["overall_complementarity"]["vjepa2_vs_ssrd"]
    lines.append(
        "overall paired complementarity | "
        f"both={overall['counts']['both_correct']} "
        f"vjepa2_only={overall['counts']['vjepa2_only']} "
        f"ssrd_only={overall['counts']['ssrd_only']} "
        f"neither={overall['counts']['neither_correct']} "
        f"mcnemar_p={overall['mcnemar']['exact_two_sided_p']:.4g}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-a-report", type=Path, default=ROOT / "results" / "detector_a_report.json")
    parser.add_argument("--detector-b-report", type=Path, default=ROOT / "results" / "detector_b_report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "statistical_audit_report.json")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    report = build_report(
        detector_a_report=args.detector_a_report,
        detector_b_report=args.detector_b_report,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.output}")
    print(compact_console_summary(report))


if __name__ == "__main__":
    main()
