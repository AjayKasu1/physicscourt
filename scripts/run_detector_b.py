#!/usr/bin/env python3
"""Run Detector B staged caches and explicit rule scores."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
import traceback
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

from physicscourt.detectors.li_detector_b import (
    CoTrackerStage,
    DepthStage,
    SAM2Stage,
    clip_metadata,
    reconstruct_state_and_score,
    save_npz_cache,
)
from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest

console = Console()

STAGE_ORDER = ["li_cotracker", "li_sam2", "li_depth", "li_state_rules"]
DEFAULT_CLIP_TIMEOUT_SECONDS = {
    "li_cotracker": 300.0,
    "li_sam2": 90.0,
    "li_depth": 60.0,
    "li_state_rules": 30.0,
}


def load_models_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cache_path(features_dir: Path, stage: str, clip: ClipRecord) -> Path:
    return features_dir / stage / f"{clip.clip_id}.npz"


def default_timing_report(stage: str) -> Path:
    return ROOT / "results" / f"{stage}_timing.json"


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    tmp.replace(path)


def cleanup_partial_cache(path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".tmp")
    if partial.exists():
        partial.unlink()


def load_prior_report(path: Path, records: list[ClipRecord], force: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if force or not path.exists():
        return [], []
    with path.open("r", encoding="utf-8") as fh:
        prior = json.load(fh)
    known = {record.clip_id for record in records}
    timings = [item for item in prior.get("timings", []) if item.get("clip_id") in known]
    errors = [item for item in prior.get("errors", []) if item.get("clip_id") in known]
    return timings, errors


def build_model_stage(stage: str, cfg: dict[str, Any], device: str, fp16: bool) -> Any:
    models = cfg["models"]
    if stage == "li_cotracker":
        spec = models["cotracker"]
        return CoTrackerStage(
            model_id=str(spec["model_id"]),
            checkpoint_file=str(spec.get("checkpoint_file", "scaled_offline.pth")),
            device=device,
        )
    if stage == "li_sam2":
        return SAM2Stage(model_id=str(models["sam2"]["model_id"]), device=device, fp16=fp16)
    if stage == "li_depth":
        return DepthStage(model_id=str(models["depth_anything_v2"]["model_id"]), device=device, fp16=fp16)
    raise ValueError(f"{stage} is not a model stage")


def process_clip(stage: str, model: Any, clip: ClipRecord, features_dir: Path) -> tuple[dict[str, Any], Any]:
    if stage == "li_cotracker":
        arrays, timing = model.process_clip(clip)
    elif stage == "li_sam2":
        arrays, timing = model.process_clip(clip, cache_path(features_dir, "li_cotracker", clip))
    elif stage == "li_depth":
        arrays, timing = model.process_clip(clip)
    elif stage == "li_state_rules":
        arrays, timing = reconstruct_state_and_score(clip, features_dir)
    else:
        raise ValueError(f"Unknown Detector B stage {stage}")
    metadata = clip_metadata(clip, stage)
    if stage == "li_state_rules":
        metadata["detector"] = "li_state_rules"
    return arrays, timing


class ClipTimeoutError(TimeoutError):
    pass


class WorkerFailure(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(str(payload.get("error", "Detector B worker failed.")))
        self.payload = payload


def _worker_payload(
    *,
    status: str,
    stage: str,
    clip_id: str | None = None,
    timing: dict[str, Any] | None = None,
    error_type: str | None = None,
    error: str | None = None,
    traceback_tail: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "stage": stage}
    if clip_id is not None:
        payload["clip_id"] = clip_id
    if timing is not None:
        payload["timing"] = timing
    if error_type is not None:
        payload["error_type"] = error_type
    if error is not None:
        payload["error"] = error
    if traceback_tail is not None:
        payload["traceback_tail"] = traceback_tail
    return payload


def _model_stage_worker(
    *,
    stage: str,
    models_config: dict[str, Any],
    features_dir: str,
    device: str,
    fp16: bool,
    task_queue: Any,
    result_queue: Any,
) -> None:
    model = None
    try:
        features_path = Path(features_dir)
        model = build_model_stage(stage, models_config, device, fp16)
        result_queue.put(_worker_payload(status="ready", stage=stage))
        while True:
            task = task_queue.get()
            if task is None:
                break
            clip = task["clip"]
            out_path = Path(task["out_path"])
            try:
                arrays, timing = process_clip(stage, model, clip, features_path)
                save_npz_cache(out_path, clip_metadata(clip, stage), arrays)
                result_queue.put(
                    _worker_payload(
                        status="ok",
                        stage=stage,
                        clip_id=clip.clip_id,
                        timing=asdict(timing),
                    )
                )
            except Exception as exc:
                cleanup_partial_cache(out_path)
                result_queue.put(
                    _worker_payload(
                        status="failed",
                        stage=stage,
                        clip_id=clip.clip_id,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback_tail=traceback.format_exc(limit=8),
                    )
                )
    except Exception as exc:
        result_queue.put(
            _worker_payload(
                status="worker_failed",
                stage=stage,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback_tail=traceback.format_exc(limit=8),
            )
        )
    finally:
        if model is not None:
            model.close()


class ModelStageWorker:
    def __init__(
        self,
        *,
        stage: str,
        models_config: dict[str, Any],
        features_dir: Path,
        device: str,
        fp16: bool,
        model_start_timeout_seconds: float,
    ) -> None:
        self.stage = stage
        self.models_config = models_config
        self.features_dir = features_dir
        self.device = device
        self.fp16 = fp16
        self.model_start_timeout_seconds = model_start_timeout_seconds
        self.ctx = mp.get_context("spawn")
        self.task_queue: Any | None = None
        self.result_queue: Any | None = None
        self.process: mp.Process | None = None
        self.start()

    def start(self) -> None:
        self.task_queue = self.ctx.Queue(maxsize=1)
        self.result_queue = self.ctx.Queue(maxsize=1)
        self.process = self.ctx.Process(
            target=_model_stage_worker,
            kwargs={
                "stage": self.stage,
                "models_config": self.models_config,
                "features_dir": str(self.features_dir),
                "device": self.device,
                "fp16": self.fp16,
                "task_queue": self.task_queue,
                "result_queue": self.result_queue,
            },
        )
        self.process.start()
        try:
            payload = self._get_result(self.model_start_timeout_seconds)
        except Exception:
            self.terminate()
            raise
        if payload.get("status") == "ready":
            return
        self.terminate()
        raise WorkerFailure(payload)

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive() and self.task_queue is not None:
            try:
                self.task_queue.put(None, timeout=1)
            except Exception:
                pass
            self.process.join(30)
        if self.process.is_alive():
            self.terminate()
        self.process = None

    def terminate(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(10)
        self.process = None

    def run_clip(self, clip: ClipRecord, out_path: Path, timeout_seconds: float) -> dict[str, Any]:
        if self.process is None or not self.process.is_alive() or self.task_queue is None:
            raise WorkerFailure(
                _worker_payload(
                    status="worker_failed",
                    stage=self.stage,
                    clip_id=clip.clip_id,
                    error_type="ChildProcessError",
                    error="Detector B worker is not running.",
                )
            )
        self.task_queue.put({"clip": clip, "out_path": str(out_path)}, timeout=5)
        try:
            payload = self._get_result(timeout_seconds)
        except ClipTimeoutError:
            self.terminate()
            cleanup_partial_cache(out_path)
            raise
        if payload.get("status") == "ok":
            return payload["timing"]
        if payload.get("status") in {"failed", "worker_failed"}:
            raise WorkerFailure(payload)
        raise WorkerFailure(
            _worker_payload(
                status="worker_failed",
                stage=self.stage,
                clip_id=clip.clip_id,
                error_type="ProtocolError",
                error=f"Unexpected worker payload: {payload}",
            )
        )

    def _get_result(self, timeout_seconds: float) -> dict[str, Any]:
        assert self.result_queue is not None
        assert self.process is not None
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ClipTimeoutError(f"{self.stage} exceeded {timeout_seconds:.1f}s.")
            try:
                return self.result_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if not self.process.is_alive():
                    raise WorkerFailure(
                        _worker_payload(
                            status="worker_failed",
                            stage=self.stage,
                            error_type="ChildProcessError",
                            error=f"Detector B worker exited with code {self.process.exitcode} without returning a result.",
                        )
                    )


def missing_prerequisites(stage: str, clip: ClipRecord, features_dir: Path) -> list[str]:
    prereqs: dict[str, list[str]] = {
        "li_cotracker": [],
        "li_sam2": ["li_cotracker"],
        "li_depth": [],
        "li_state_rules": ["li_cotracker", "li_sam2", "li_depth"],
    }
    return [name for name in prereqs[stage] if not cache_path(features_dir, name, clip).exists()]


def run_stage(
    *,
    stage: str,
    records: list[ClipRecord],
    models_config: dict[str, Any],
    features_dir: Path,
    timing_report: Path,
    device: str,
    fp16: bool,
    force: bool,
    continue_on_error: bool,
    clip_timeout_seconds: float | None,
    model_start_timeout_seconds: float,
) -> dict[str, Any]:
    timings, errors = load_prior_report(timing_report, records, force)
    timed_ids = {item["clip_id"] for item in timings}
    timeout_seconds = clip_timeout_seconds or DEFAULT_CLIP_TIMEOUT_SECONDS[stage]
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "num_requested": len(records),
        "clip_timeout_seconds": timeout_seconds,
        "timings": timings,
        "errors": errors,
    }

    def heartbeat() -> None:
        write_report(timing_report, report)

    worker: ModelStageWorker | None = None

    def clear_error(clip_id: str) -> None:
        nonlocal errors
        errors[:] = [item for item in errors if item.get("clip_id") != clip_id]

    def append_error(clip: ClipRecord, error_type: str, message: str, **extra: Any) -> None:
        errors.append(
            {
                "clip_id": clip.clip_id,
                "stage": stage,
                "error_type": error_type,
                "error": message,
                **extra,
            }
        )

    def ensure_worker() -> ModelStageWorker:
        nonlocal worker
        if worker is None:
            worker = ModelStageWorker(
                stage=stage,
                models_config=models_config,
                features_dir=features_dir,
                device=device,
                fp16=fp16,
                model_start_timeout_seconds=model_start_timeout_seconds,
            )
        return worker

    try:
        for clip in track(records, description=stage):
            out_path = cache_path(features_dir, stage, clip)
            if out_path.exists() and not force:
                if clip.clip_id not in timed_ids:
                    timings.append(
                        {
                            "clip_id": clip.clip_id,
                            "stage": stage,
                            "seconds": None,
                            "num_frames": clip.num_frames,
                            "cache_hit": True,
                        }
                    )
                    timed_ids.add(clip.clip_id)
                    heartbeat()
                continue
            missing = missing_prerequisites(stage, clip, features_dir)
            if missing:
                message = f"Missing prerequisite caches: {', '.join(missing)}"
                append_error(clip, "MissingPrerequisite", message)
                heartbeat()
                if continue_on_error:
                    console.print({"skipped_clip": clip.clip_id, "stage": stage, "missing": missing})
                    continue
                raise RuntimeError(message)

            try:
                if stage == "li_state_rules":
                    arrays, timing = process_clip(stage, None, clip, features_dir)
                    save_npz_cache(out_path, clip_metadata(clip, stage), arrays)
                    timing_record = asdict(timing)
                else:
                    timing_record = ensure_worker().run_clip(clip, out_path, timeout_seconds)
            except ClipTimeoutError as exc:
                append_error(
                    clip,
                    "TimeoutError",
                    str(exc),
                    timeout_seconds=timeout_seconds,
                    skipped=True,
                )
                heartbeat()
                if continue_on_error:
                    console.print(
                        {
                            "skipped_clip": clip.clip_id,
                            "stage": stage,
                            "error": str(exc),
                            "timeout_seconds": timeout_seconds,
                        }
                    )
                    worker = None
                    continue
                raise
            except WorkerFailure as exc:
                payload = exc.payload
                append_error(
                    clip,
                    str(payload.get("error_type", "WorkerFailure")),
                    str(payload.get("error", exc)),
                    traceback_tail=payload.get("traceback_tail"),
                    skipped=True,
                )
                heartbeat()
                if continue_on_error:
                    console.print({"skipped_clip": clip.clip_id, "stage": stage, "error": str(exc)})
                    if payload.get("status") == "worker_failed":
                        worker = None
                    continue
                raise
            except Exception as exc:
                append_error(clip, type(exc).__name__, str(exc), skipped=True)
                heartbeat()
                if continue_on_error:
                    console.print({"skipped_clip": clip.clip_id, "stage": stage, "error": str(exc)})
                    continue
                raise
            clear_error(clip.clip_id)
            timings.append(timing_record)
            timed_ids.add(clip.clip_id)
            heartbeat()
    finally:
        if worker is not None:
            worker.close()

    heartbeat()
    seconds = [item["seconds"] for item in timings if isinstance(item.get("seconds"), (int, float))]
    summary = {
        "stage": stage,
        "records": len(timings),
        "errors": len(errors),
        "seconds": float(sum(seconds)),
        "seconds_per_scored_clip": float(sum(seconds) / len(seconds)) if seconds else None,
        "skipped_clips": [item["clip_id"] for item in errors],
        "timing_report": str(timing_report),
    }
    console.print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGE_ORDER + ["all"], required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--models-config", type=Path, default=ROOT / "config" / "models.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features")
    parser.add_argument("--timing-report", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--clip-timeout-seconds", type=float, default=None)
    parser.add_argument("--model-start-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]
    cfg = load_models_config(args.models_config)

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    summaries = []
    for stage in stages:
        timing_report = args.timing_report if args.timing_report is not None and len(stages) == 1 else default_timing_report(stage)
        summaries.append(
            run_stage(
                stage=stage,
                records=records,
                models_config=cfg,
                features_dir=args.features_dir,
                timing_report=timing_report,
                device=args.device,
                fp16=args.fp16,
                force=args.force,
                continue_on_error=args.continue_on_error,
                clip_timeout_seconds=args.clip_timeout_seconds,
                model_start_timeout_seconds=args.model_start_timeout_seconds,
            )
        )
    console.print({"phase": "phase3_detector_b", "summaries": summaries})


if __name__ == "__main__":
    main()
