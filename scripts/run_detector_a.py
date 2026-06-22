#!/usr/bin/env python3
"""Run Detector A feature scoring passes over a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import track

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.detectors.lecun_dino_baseline import DINOLatentExtrapolator
from physicscourt.detectors.lecun_vjepa import VJEPALatentPredictor
from physicscourt.detectors.lecun_vjepa21 import OfficialVJEPA21Predictor
from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest
from physicscourt.pipeline.score_utils import save_score_cache

console = Console()


def _load_model_id(config_path: Path, key: str) -> str:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return str(cfg["models"][key]["model_id"])


def _load_model_spec(config_path: Path, key: str) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return dict(cfg["models"][key])


def _detector(detector_name: str, config_path: Path, device: str, fp16: bool, vjepa_topk_frac: float) -> Any:
    if detector_name == "vjepa2":
        return VJEPALatentPredictor(
            model_id=_load_model_id(config_path, "vjepa2"),
            device=device,
            fp16=fp16,
            topk_frac=vjepa_topk_frac,
        )
    if detector_name == "vjepa2_1":
        spec = _load_model_spec(config_path, "vjepa2_1")
        return OfficialVJEPA21Predictor(
            hub_model=str(spec["hub_model"]),
            checkpoint_url=str(spec["checkpoint_url"]),
            checkpoint_key=str(spec.get("checkpoint_key", "target_encoder")),
            device=device,
            fp16=fp16,
            image_size=int(spec.get("image_size", 384)),
            window_frames=int(spec.get("window_frames", 64)),
            stride_frames=int(spec.get("stride_frames", 4)),
            topk_frac=vjepa_topk_frac,
        )
    if detector_name == "dino_latent":
        return DINOLatentExtrapolator(
            model_id=_load_model_id(config_path, "dino_latent"),
            device=device,
            fp16=fp16,
        )
    raise ValueError(f"Unknown detector {detector_name}")


def _cache_path(features_dir: Path, detector_name: str, clip: ClipRecord) -> Path:
    return features_dir / detector_name / f"{clip.clip_id}.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["vjepa2", "vjepa2_1", "dino_latent"], required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--models-config", type=Path, default=ROOT / "config" / "models.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features")
    parser.add_argument("--timing-report", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--vjepa-topk-frac", type=float, default=0.05)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]

    detector = _detector(args.detector, args.models_config, args.device, args.fp16, args.vjepa_topk_frac)
    timings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if args.timing_report is not None and args.timing_report.exists() and not args.force:
        with args.timing_report.open("r", encoding="utf-8") as fh:
            prior = json.load(fh)
        known_ids = {record.clip_id for record in records}
        timings = [item for item in prior.get("timings", []) if item.get("clip_id") in known_ids]
        errors = [item for item in prior.get("errors", []) if item.get("clip_id") in known_ids]
    timed_ids = {item["clip_id"] for item in timings}
    errored_ids = {item["clip_id"] for item in errors}
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": args.detector,
        "manifest": str(args.manifest),
        "limit": args.limit,
        "num_requested": len(records),
        "timings": timings,
        "errors": errors,
    }

    def write_timing_report() -> None:
        if args.timing_report is None:
            return
        args.timing_report.parent.mkdir(parents=True, exist_ok=True)
        with args.timing_report.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")

    try:
        for clip in track(records, description=f"{args.detector}"):
            if clip.clip_id in errored_ids and not args.force:
                continue
            out_path = _cache_path(args.features_dir, args.detector, clip)
            if out_path.exists() and not args.force:
                if clip.clip_id not in timed_ids:
                    timings.append(
                        {
                            "clip_id": clip.clip_id,
                            "seconds": None,
                            "num_frames": clip.num_frames,
                            "num_windows": None,
                            "cache_hit": True,
                        }
                    )
                    timed_ids.add(clip.clip_id)
                    write_timing_report()
                continue
            try:
                raw, extras, timing = detector.score_clip(clip)
            except Exception as exc:
                error = {
                    "clip_id": clip.clip_id,
                    "pair_id": clip.pair_id,
                    "detector": args.detector,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                errors.append(error)
                errored_ids.add(clip.clip_id)
                write_timing_report()
                if args.continue_on_error:
                    console.print({"skipped_clip": clip.clip_id, "error": str(exc)})
                    continue
                raise
            metadata = {
                "clip_id": clip.clip_id,
                "pair_id": clip.pair_id,
                "detector": args.detector,
                "split": clip.split,
                "category": clip.category,
                "possible": clip.possible,
                "violation_frame": clip.violation_frame,
                "video_path": clip.video_path,
                "num_frames": clip.num_frames,
                "fps": clip.fps,
            }
            save_score_cache(out_path, raw_score=raw, metadata=metadata, extra_arrays=extras)
            timings.append(asdict(timing))
            timed_ids.add(clip.clip_id)
            write_timing_report()
    finally:
        detector.close()

    write_timing_report()
    numeric_seconds = [item["seconds"] for item in timings if isinstance(item.get("seconds"), (int, float))]
    total_seconds = sum(numeric_seconds)
    console.print(
        {
            "detector": args.detector,
            "records": len(timings),
            "errors": len(errors),
            "seconds": total_seconds,
            "seconds_per_scored_clip": total_seconds / len(numeric_seconds) if numeric_seconds else None,
            "skipped_clips": [item["clip_id"] for item in errors],
        }
    )


if __name__ == "__main__":
    main()
