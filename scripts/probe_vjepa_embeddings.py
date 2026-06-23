#!/usr/bin/env python3
"""Train simple linear probes on cached V-JEPA embeddings.

This is a representation audit, not a calibration-only detector. It tests
whether possible/impossible information is linearly readable from frozen
embeddings after the surprise readout has failed to expose it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def stable_fold(value: str, num_folds: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_folds


def load_rows(features_dir: Path, detector: str, feature_key: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((features_dir / detector).glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            if feature_key not in data:
                continue
            metadata = dict(data["metadata_json"][0])
            rows.append(
                {
                    "path": str(path),
                    "clip_id": str(metadata["clip_id"]),
                    "pair_id": str(metadata["pair_id"]),
                    "split": str(metadata["split"]),
                    "category": str(metadata["category"]),
                    "possible": bool(metadata["possible"]),
                    "label": 0 if bool(metadata["possible"]) else 1,
                    "x": np.asarray(data[feature_key], dtype=np.float32).reshape(-1),
                }
            )
    if not rows:
        raise SystemExit(f"no usable {feature_key} caches found under {features_dir / detector}")
    return rows


def make_model(c_value: float, max_iter: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=max_iter,
            solver="liblinear",
            random_state=1729,
        ),
    )


def safe_auc(labels: list[int], scores: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def paired_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, dict[bool, float]] = defaultdict(dict)
    for row in rows:
        by_pair[str(row["pair_id"])][bool(row["possible"])] = float(row["score"])
    impossible_higher = 0
    possible_higher = 0
    tied = 0
    for values in by_pair.values():
        if True not in values or False not in values:
            continue
        margin = values[False] - values[True]
        if margin > 1e-12:
            impossible_higher += 1
        elif margin < -1e-12:
            possible_higher += 1
        else:
            tied += 1
    total = impossible_higher + possible_higher + tied
    return {
        "pairs": int(total),
        "impossible_higher": int(impossible_higher),
        "possible_higher": int(possible_higher),
        "tied": int(tied),
        "tie_half_accuracy": float((impossible_higher + 0.5 * tied) / total) if total else float("nan"),
    }


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "overall": {
            "num_clips": int(len(predictions)),
            "auc": safe_auc([int(row["label"]) for row in predictions], [float(row["score"]) for row in predictions]),
            "paired": paired_accuracy(predictions),
        },
        "by_category": {},
    }
    categories = sorted({str(row["category"]) for row in predictions})
    for category in categories:
        subset = [row for row in predictions if row["category"] == category]
        out["by_category"][category] = {
            "num_clips": int(len(subset)),
            "auc": safe_auc([int(row["label"]) for row in subset], [float(row["score"]) for row in subset]),
            "paired": paired_accuracy(subset),
        }
    return out


def predict_with_train_test(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    c_value: float,
    max_iter: int,
    protocol: str,
    fold: str,
) -> list[dict[str, Any]]:
    if len({row["label"] for row in train_rows}) < 2:
        return []
    x_train = np.stack([row["x"] for row in train_rows], axis=0)
    y_train = np.array([row["label"] for row in train_rows], dtype=np.int32)
    x_test = np.stack([row["x"] for row in test_rows], axis=0)
    model = make_model(c_value, max_iter)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1].astype(np.float32)
    out = []
    for row, score in zip(test_rows, scores):
        item = {key: row[key] for key in ("clip_id", "pair_id", "category", "possible", "label")}
        item.update({"score": float(score), "protocol": protocol, "fold": fold})
        out.append(item)
    return out


def pair_cv_predictions(rows: list[dict[str, Any]], num_folds: int, c_value: float, max_iter: int) -> list[dict[str, Any]]:
    predictions = []
    for fold in range(num_folds):
        train_rows = [row for row in rows if stable_fold(row["pair_id"], num_folds) != fold]
        test_rows = [row for row in rows if stable_fold(row["pair_id"], num_folds) == fold]
        if not test_rows:
            continue
        predictions.extend(
            predict_with_train_test(rows, train_rows, test_rows, c_value, max_iter, "pair_cv", str(fold))
        )
    return predictions


def category_holdout_predictions(rows: list[dict[str, Any]], c_value: float, max_iter: int) -> list[dict[str, Any]]:
    predictions = []
    for category in sorted({row["category"] for row in rows}):
        train_rows = [row for row in rows if row["category"] != category]
        test_rows = [row for row in rows if row["category"] == category]
        predictions.extend(
            predict_with_train_test(rows, train_rows, test_rows, c_value, max_iter, "category_holdout", category)
        )
    return predictions


def load_surprise_baseline(path: Path | None, detector: str) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("detectors", {}).get(detector, {}).get("metrics", {})
    return {category: float(values.get("roc_auc", math.nan)) for category, values in metrics.items()}


def print_summary(report: dict[str, Any]) -> None:
    baseline = report.get("surprise_baseline_auc_by_category", {})
    for protocol, payload in report["protocols"].items():
        print(f"\nprotocol: {protocol}")
        print("category".ljust(30) + "probe_auc  surprise_auc  pairs  tiehalf")
        print("-" * 72)
        for category, values in payload["by_category"].items():
            paired = values["paired"]
            surprise = baseline.get(category, float("nan"))
            print(
                category.ljust(30)
                + f"{values['auc']:9.3f} "
                + f"{surprise:12.3f} "
                + f"{paired['pairs']:6d} "
                + f"{paired['tie_half_accuracy']:8.3f}"
            )
        overall = payload["overall"]
        print(
            "overall".ljust(30)
            + f"{overall['auc']:9.3f} "
            + f"{float('nan'):12.3f} "
            + f"{overall['paired']['pairs']:6d} "
            + f"{overall['paired']['tie_half_accuracy']:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=Path("results/features_vjepa21_probe_l4_fp32"))
    parser.add_argument("--detector", default="vjepa2_1")
    parser.add_argument("--feature-key", default="embedding_vector")
    parser.add_argument("--surprise-report", type=Path, default=None)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("results/vjepa21_linear_probe_l4_fp32_report.json"))
    args = parser.parse_args()

    rows = [row for row in load_rows(args.features_dir, args.detector, args.feature_key) if row["split"] == "test"]
    if len(rows) < 4:
        raise SystemExit("not enough test rows for a supervised probe")

    protocols = {
        "pair_cv": summarize_predictions(pair_cv_predictions(rows, args.num_folds, args.c, args.max_iter)),
        "category_holdout": summarize_predictions(category_holdout_predictions(rows, args.c, args.max_iter)),
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": args.detector,
        "features_dir": str(args.features_dir),
        "feature_key": args.feature_key,
        "num_test_rows": int(len(rows)),
        "note": (
            "Supervised representation audit. This is not calibration-only scoring. "
            "Pair-CV holds out matched pairs; category-holdout tests transfer to an unseen violation type."
        ),
        "surprise_baseline_auc_by_category": load_surprise_baseline(args.surprise_report, args.detector),
        "protocols": protocols,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
