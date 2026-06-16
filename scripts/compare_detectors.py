#!/usr/bin/env python3
"""Write a compact Detector A vs Detector B comparison report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def preferred_localization_within(metric: dict[str, Any]) -> float:
    preferred = metric.get("preferred_localization", "argmax")
    key = f"{preferred}_localization_within_tolerance"
    return float(metric.get(key, metric.get("localization_within_tolerance", float("nan"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-a-report", type=Path, default=ROOT / "results" / "detector_a_report.json")
    parser.add_argument("--detector-b-report", type=Path, default=ROOT / "results" / "detector_b_report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "head_to_head_report.json")
    args = parser.parse_args()

    detector_a = load_json(args.detector_a_report)["detectors"]
    detector_b = load_json(args.detector_b_report)["detectors"]["li_state_rules"]
    categories = sorted(detector_b["metrics"])
    rows: list[dict[str, Any]] = []
    wins = {"vjepa2": 0, "dino_latent": 0, "li_state_rules": 0}
    display_names = {
        "vjepa2": "V-JEPA 2 (facebook/vjepa2-vitl-fpc64-256)",
        "dino_latent": "DINOv2-small latent extrapolation baseline",
        "li_state_rules": "SSRD: Spatial-State Rule Detector",
    }

    for category in categories:
        metrics = {
            "vjepa2": detector_a["vjepa2"]["metrics"][category],
            "dino_latent": detector_a["dino_latent"]["metrics"][category],
            "li_state_rules": detector_b["metrics"][category],
        }
        aucs = {name: float(metric["roc_auc"]) for name, metric in metrics.items()}
        winner = max(aucs, key=aucs.get)
        wins[winner] += 1
        rows.append(
            {
                "category": category,
                "winner_by_auc": winner,
                "vjepa2": {
                    "roc_auc": aucs["vjepa2"],
                    "paired_accuracy": float(metrics["vjepa2"]["paired_accuracy"]),
                    "preferred_localization_within_12": preferred_localization_within(metrics["vjepa2"]),
                },
                "dino_latent": {
                    "roc_auc": aucs["dino_latent"],
                    "paired_accuracy": float(metrics["dino_latent"]["paired_accuracy"]),
                    "preferred_localization_within_12": preferred_localization_within(metrics["dino_latent"]),
                },
                "li_state_rules": {
                    "roc_auc": aucs["li_state_rules"],
                    "paired_accuracy": float(metrics["li_state_rules"]["paired_accuracy"]),
                    "preferred_localization_within_12": preferred_localization_within(metrics["li_state_rules"]),
                    "argmax_localization_within_12": float(metrics["li_state_rules"]["argmax_localization_within_tolerance"]),
                    "onset_localization_within_12": float(metrics["li_state_rules"]["onset_localization_within_tolerance"]),
                    "onset_detected_rate": float(metrics["li_state_rules"]["onset_detected_rate"]),
                },
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "display_names": display_names,
        "framing_note": (
            "SSRD is a project-built detector assembled from SAM 2, Depth Anything V2, CoTracker, "
            "and a hand-written calibrated physics-rule layer. It is not World Labs Marble and "
            "should not be attributed to Fei-Fei Li. Marble 1.1 / Marble 1.1 Plus is the public "
            "World Labs reference point for explicit 3D world generation."
        ),
        "fairness_disclosure": (
            "The comparison is asymmetric: V-JEPA 2 is a frozen external model, while SSRD is a "
            "project-built detector with an explicit rule bank, calibration layer, and more design "
            "degrees of freedom. Both detectors are calibrated only on the normal calibration split; "
            "no test clip statistics or labels are used for fitting."
        ),
        "protocol_note": (
            "Detector B scores are category-blind: all explicit rule families are run on every clip, "
            "each rule channel is z-scored using calibration-normal clips only, and clip-level score is the "
            "max across calibrated rule scores. Category labels are used only for reporting."
        ),
        "winner_counts_by_auc": wins,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
