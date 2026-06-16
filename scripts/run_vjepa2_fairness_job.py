#!/usr/bin/env python3
"""Resumable live V-JEPA 2 window/stride fairness probe."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physicscourt.detectors.lecun_vjepa import VJEPALatentPredictor
from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest
from physicscourt.utils.torch_runtime import clear_torch
from vjepa2_fairness_sweep import (
    endpoint_indices,
    frames_for_clip,
    load_model_id,
    resolve_local_model_path,
    selected_pairs,
)


DEFAULT_CATEGORIES = [
    "continuity_teleportation",
    "gravity_support",
    "object_permanence",
    "spontaneous_vanishing",
]


class ClipTimeoutError(TimeoutError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@contextmanager
def clip_timeout(seconds: int) -> Iterator[None]:
    if seconds <= 0:
        yield
        return

    def handler(signum: int, frame: object) -> None:
        raise ClipTimeoutError(f"clip work exceeded {seconds}s")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def native_window_frames(model_path: str) -> int:
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    return int(config.get("frames_per_clip", 64))


def normalized_window_frames(args: argparse.Namespace, model_path: str) -> list[int]:
    if args.window_frames:
        values = args.window_frames
    else:
        native = native_window_frames(model_path)
        values = [native, args.short_window_frames]
    out: list[int] = []
    for value in values:
        if value not in out:
            out.append(int(value))
    return out


def item_key(clip: ClipRecord, window_frames: int, stride: int, selection_mode: str) -> str:
    return f"{clip.clip_id}__w{window_frames:03d}__s{stride:02d}__{selection_mode}"


def item_output_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def item_error_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.error.json"


def pair_tstars(clips: list[ClipRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for clip in clips:
        if clip.violation_frame is not None:
            out[str(clip.pair_id)] = int(clip.violation_frame)
    return out


def build_work_items(
    clips: list[ClipRecord],
    window_frames: list[int],
    strides: list[int],
    selection_mode: str,
) -> list[dict[str, Any]]:
    tstars = pair_tstars(clips)
    items = []
    for window in window_frames:
        for stride in strides:
            for clip in clips:
                items.append(
                    {
                        "key": item_key(clip, window, stride, selection_mode),
                        "clip": clip,
                        "t_star": tstars.get(str(clip.pair_id), clip.num_frames // 2),
                        "window_frames": int(window),
                        "stride": int(stride),
                        "selection_mode": selection_mode,
                    }
                )
    return items


def nearest_endpoint(indices: list[int], target: int) -> int | None:
    if not indices:
        return None
    return min(indices, key=lambda item: (abs(item - target), item))


def selected_endpoint_indices(
    all_indices: list[int],
    *,
    clip: ClipRecord,
    window_frames: int,
    t_star: int,
    max_windows: int | None,
    selection_mode: str,
) -> list[int]:
    if max_windows is None or max_windows <= 0 or len(all_indices) <= max_windows:
        return all_indices

    if selection_mode == "nearest_t":
        selected = sorted(all_indices, key=lambda item: (abs(item - t_star), item))[:max_windows]
        return sorted(set(selected))

    if selection_mode != "around_and_post_t":
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    targets = [
        int(t_star),
        min(int(clip.num_frames) - 1, int(t_star) + int(window_frames) - 1),
    ]
    selected: list[int] = []
    for target in targets:
        endpoint = nearest_endpoint(all_indices, target)
        if endpoint is not None and endpoint not in selected:
            selected.append(endpoint)
    if len(selected) < max_windows:
        extras = sorted(all_indices, key=lambda item: (min(abs(item - target) for target in targets), item))
        for endpoint in extras:
            if endpoint not in selected:
                selected.append(endpoint)
            if len(selected) >= max_windows:
                break
    return sorted(set(selected[:max_windows]))


def score_item(
    detector: VJEPALatentPredictor,
    clip: ClipRecord,
    *,
    stride: int,
    t_star: int,
    max_windows: int | None,
    selection_mode: str,
) -> dict[str, Any]:
    frames = frames_for_clip(clip)
    all_indices = endpoint_indices(len(frames), detector.window_frames, stride)
    indices = selected_endpoint_indices(
        all_indices,
        clip=clip,
        window_frames=detector.window_frames,
        t_star=t_star,
        max_windows=max_windows,
        selection_mode=selection_mode,
    )
    l2_scores = []
    cosine_scores = []
    started = time.perf_counter()
    for end in indices:
        window = frames[end - detector.window_frames + 1 : end + 1]
        l2, cosine = detector._score_window(window)
        l2_scores.append(float(l2))
        cosine_scores.append(float(cosine))
    return {
        "created_at": utc_now(),
        "clip_id": clip.clip_id,
        "pair_id": clip.pair_id,
        "category": clip.category,
        "possible": bool(clip.possible),
        "violation_frame": clip.violation_frame,
        "t_star_for_selection": int(t_star),
        "selection_mode": selection_mode,
        "window_frames": int(detector.window_frames),
        "stride": int(stride),
        "num_frames": int(clip.num_frames),
        "num_windows": len(indices),
        "num_available_windows": len(all_indices),
        "live_window_subset": bool(len(indices) != len(all_indices)),
        "score_frame_indices": [int(value) for value in indices],
        "l2_window_scores": l2_scores,
        "cosine_window_scores": cosine_scores,
        "l2_clip_max": float(max(l2_scores)) if l2_scores else 0.0,
        "cosine_clip_max": float(max(cosine_scores)) if cosine_scores else 0.0,
        "seconds": float(time.perf_counter() - started),
    }


def paired_accuracy(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    pairs: dict[str, dict[bool, dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(str(row["pair_id"]), {})[bool(row["possible"])] = row
    correct = 0
    total = 0
    margins = []
    for pair_id, pair in sorted(pairs.items()):
        if True not in pair or False not in pair:
            continue
        total += 1
        possible_score = float(pair[True][score_key])
        impossible_score = float(pair[False][score_key])
        margin = impossible_score - possible_score
        correct += int(margin > 0.0)
        margins.append(
            {
                "pair_id": pair_id,
                "category": pair[False]["category"],
                "margin_impossible_minus_possible": float(margin),
                "possible_clip_id": pair[True]["clip_id"],
                "impossible_clip_id": pair[False]["clip_id"],
            }
        )
    return {
        "paired_accuracy": float(correct / total) if total else float("nan"),
        "num_pairs": total,
        "num_correct_pairs": correct,
        "margins": margins,
    }


def auc(rows: list[dict[str, Any]], score_key: str) -> float:
    y_true = np.asarray([not bool(row["possible"]) for row in rows], dtype=np.int32)
    y_score = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for window in sorted({int(row["window_frames"]) for row in rows}):
        window_rows = [row for row in rows if int(row["window_frames"]) == window]
        metrics[str(window)] = {}
        for stride in sorted({int(row["stride"]) for row in window_rows}):
            stride_rows = [row for row in window_rows if int(row["stride"]) == stride]
            stride_payload: dict[str, Any] = {}
            for category in sorted({str(row["category"]) for row in stride_rows}):
                category_rows = [row for row in stride_rows if str(row["category"]) == category]
                stride_payload[category] = {}
                for score_key in ("l2_clip_max", "cosine_clip_max"):
                    paired = paired_accuracy(category_rows, score_key)
                    stride_payload[category][score_key] = {
                        "roc_auc": auc(category_rows, score_key),
                        "paired_accuracy": paired["paired_accuracy"],
                        "num_pairs": paired["num_pairs"],
                        "num_clips": len(category_rows),
                        "num_correct_pairs": paired["num_correct_pairs"],
                    }
            metrics[str(window)][str(stride)] = stride_payload
    return metrics


def load_item_rows(cache_dir: Path, selection_mode: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(cache_dir.glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        payload = load_json(path)
        if payload is not None and "clip_id" in payload and payload.get("selection_mode") == selection_mode:
            rows.append(payload)
    return rows


def load_error_rows(cache_dir: Path, selection_mode: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(cache_dir.glob("*.error.json")):
        payload = load_json(path)
        if payload is not None and payload.get("selection_mode") == selection_mode:
            rows.append(payload)
    return rows


def progress_payload(
    *,
    args: argparse.Namespace,
    model_id: str,
    model_path: str,
    window_frames: list[int],
    strides: list[int],
    clips: list[ClipRecord],
    work_items: list[dict[str, Any]],
    cache_dir: Path,
    active_device: str | None,
    active_dtype: str | None,
    status: str,
) -> dict[str, Any]:
    rows = load_item_rows(cache_dir, args.selection_mode)
    errors = load_error_rows(cache_dir, args.selection_mode)
    completed_keys = {item_key_from_row(row) for row in rows}
    error_keys = {str(row["key"]) for row in errors if "key" in row}
    pending = [
        item["key"]
        for item in work_items
        if item["key"] not in completed_keys and item["key"] not in error_keys
    ]
    timings = [
        {
            "key": item_key_from_row(row),
            "clip_id": row["clip_id"],
            "category": row["category"],
            "possible": row["possible"],
            "window_frames": row["window_frames"],
            "stride": row["stride"],
            "seconds": row["seconds"],
            "num_windows": row["num_windows"],
        }
        for row in rows
    ]
    return {
        "created_or_updated_at": utc_now(),
        "status": status,
        "purpose": (
            "Targeted live V-JEPA 2 fairness probe over selected categories, window lengths, "
            "and strides. This is label-informed around t* and is a diagnostic upper-bound probe, "
            "not the deployable Phase 2 detector."
        ),
        "python": sys.executable,
        "platform_machine": platform.machine(),
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "model_id": model_id,
        "local_model_path": model_path,
        "requested_device": args.device,
        "requested_fp16": bool(args.fp16),
        "active_device": active_device,
        "active_dtype": active_dtype,
        "timeout_seconds": int(args.timeout_seconds),
        "max_live_windows_per_clip": args.max_live_windows_per_clip,
        "selection_mode": args.selection_mode,
        "selection_mode_note": (
            "around_and_post_t samples a window endpoint near t* plus a later endpoint whose window starts "
            "near t* when possible. This directly tests whether Phase 2 ties came from pre-t* max aggregation."
        ),
        "categories": args.categories,
        "pairs_per_category": int(args.pairs_per_category),
        "num_selected_clips": len(clips),
        "window_frames": window_frames,
        "strides": strides,
        "cache_dir": str(cache_dir),
        "progress": {
            "total_work_items": len(work_items),
            "completed": len(completed_keys),
            "errors": len(error_keys),
            "pending": len(pending),
            "last_completed_key": timings[-1]["key"] if timings else None,
        },
        "timings": timings,
        "errors": errors,
        "metrics": compute_metrics(rows),
    }


def item_key_from_row(row: dict[str, Any]) -> str:
    selection_mode = str(row.get("selection_mode", "legacy_nearest_t"))
    return f"{row['clip_id']}__w{int(row['window_frames']):03d}__s{int(row['stride']):02d}__{selection_mode}"


def write_progress(
    *,
    args: argparse.Namespace,
    model_id: str,
    model_path: str,
    window_frames: list[int],
    strides: list[int],
    clips: list[ClipRecord],
    work_items: list[dict[str, Any]],
    cache_dir: Path,
    active_device: str | None,
    active_dtype: str | None,
    status: str,
) -> None:
    payload = progress_payload(
        args=args,
        model_id=model_id,
        model_path=model_path,
        window_frames=window_frames,
        strides=strides,
        clips=clips,
        work_items=work_items,
        cache_dir=cache_dir,
        active_device=active_device,
        active_dtype=active_dtype,
        status=status,
    )
    atomic_write_json(args.report, payload)


def should_skip(item: dict[str, Any], cache_dir: Path, *, force: bool, retry_errors: bool) -> bool:
    output = item_output_path(cache_dir, item["key"])
    error = item_error_path(cache_dir, item["key"])
    if force:
        return False
    if output.exists():
        return True
    if error.exists() and not retry_errors:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--models-config", type=Path, default=ROOT / "config" / "models.yaml")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "features" / "vjepa2_fairness_live_postt")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "vjepa2_fairness_live_report.json")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--pairs-per-category", type=int, default=3)
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--window-frames", type=int, nargs="*", default=None)
    parser.add_argument("--short-window-frames", type=int, default=16)
    parser.add_argument("--max-live-windows-per-clip", type=int, default=2)
    parser.add_argument("--selection-mode", choices=["around_and_post_t", "nearest_t"], default="around_and_post_t")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--limit-work-items", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    clips = selected_pairs(records, args.categories, args.pairs_per_category)
    model_id = load_model_id(args.models_config)
    model_path = resolve_local_model_path(model_id)
    windows = normalized_window_frames(args, model_path)
    strides = [int(value) for value in args.strides]
    work_items = build_work_items(clips, windows, strides, args.selection_mode)
    if args.limit_work_items is not None:
        work_items = work_items[: int(args.limit_work_items)]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    write_progress(
        args=args,
        model_id=model_id,
        model_path=model_path,
        window_frames=windows,
        strides=strides,
        clips=clips,
        work_items=work_items,
        cache_dir=args.cache_dir,
        active_device=None,
        active_dtype=None,
        status="dry_run" if args.dry_run else "running",
    )
    print(f"work_items={len(work_items)} selected_clips={len(clips)} windows={windows} strides={strides}")
    print(f"heartbeat={args.report}")
    if args.dry_run:
        return

    detector: VJEPALatentPredictor | None = None
    active_window: int | None = None
    active_device: str | None = None
    active_dtype: str | None = None
    try:
        for item in work_items:
            if should_skip(item, args.cache_dir, force=args.force, retry_errors=args.retry_errors):
                continue
            window = int(item["window_frames"])
            if detector is None or active_window != window:
                if detector is not None:
                    detector.close()
                    detector = None
                try:
                    detector = VJEPALatentPredictor(
                        model_id=model_path,
                        device=args.device,
                        fp16=args.fp16,
                        window_frames=window,
                        stride_frames=4,
                    )
                except Exception as exc:
                    clip = item["clip"]
                    error_payload = {
                        "created_at": utc_now(),
                        "key": item["key"],
                        "clip_id": clip.clip_id,
                        "category": clip.category,
                        "possible": bool(clip.possible),
                        "selection_mode": str(item["selection_mode"]),
                        "window_frames": int(item["window_frames"]),
                        "stride": int(item["stride"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "stage": "model_init",
                    }
                    atomic_write_json(item_error_path(args.cache_dir, item["key"]), error_payload)
                    write_progress(
                        args=args,
                        model_id=model_id,
                        model_path=model_path,
                        window_frames=windows,
                        strides=strides,
                        clips=clips,
                        work_items=work_items,
                        cache_dir=args.cache_dir,
                        active_device=active_device,
                        active_dtype=active_dtype,
                        status="failed_model_init",
                    )
                    raise
                active_window = window
                active_device = detector.device.type
                active_dtype = str(detector.dtype).replace("torch.", "")
                write_progress(
                    args=args,
                    model_id=model_id,
                    model_path=model_path,
                    window_frames=windows,
                    strides=strides,
                    clips=clips,
                    work_items=work_items,
                    cache_dir=args.cache_dir,
                    active_device=active_device,
                    active_dtype=active_dtype,
                    status="running",
                )
            clip = item["clip"]
            print(f"scoring {item['key']}")
            try:
                with clip_timeout(int(args.timeout_seconds)):
                    row = score_item(
                        detector,
                        clip,
                        stride=int(item["stride"]),
                        t_star=int(item["t_star"]),
                        max_windows=args.max_live_windows_per_clip,
                        selection_mode=str(item["selection_mode"]),
                    )
                atomic_write_json(item_output_path(args.cache_dir, item["key"]), row)
            except ClipTimeoutError as exc:
                error_payload = {
                    "created_at": utc_now(),
                    "key": item["key"],
                    "clip_id": clip.clip_id,
                    "category": clip.category,
                    "possible": bool(clip.possible),
                    "selection_mode": str(item["selection_mode"]),
                    "window_frames": int(item["window_frames"]),
                    "stride": int(item["stride"]),
                    "error_type": "timeout",
                    "error": str(exc),
                }
                atomic_write_json(item_error_path(args.cache_dir, item["key"]), error_payload)
                if detector is not None:
                    detector.close()
                    detector = None
                    active_window = None
                    clear_torch()
            except Exception as exc:
                error_payload = {
                    "created_at": utc_now(),
                    "key": item["key"],
                    "clip_id": clip.clip_id,
                    "category": clip.category,
                    "possible": bool(clip.possible),
                    "selection_mode": str(item["selection_mode"]),
                    "window_frames": int(item["window_frames"]),
                    "stride": int(item["stride"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                atomic_write_json(item_error_path(args.cache_dir, item["key"]), error_payload)
            finally:
                write_progress(
                    args=args,
                    model_id=model_id,
                    model_path=model_path,
                    window_frames=windows,
                    strides=strides,
                    clips=clips,
                    work_items=work_items,
                    cache_dir=args.cache_dir,
                    active_device=active_device,
                    active_dtype=active_dtype,
                    status="running",
                )
    finally:
        if detector is not None:
            detector.close()
        clear_torch()

    final_status = "complete"
    write_progress(
        args=args,
        model_id=model_id,
        model_path=model_path,
        window_frames=windows,
        strides=strides,
        clips=clips,
        work_items=work_items,
        cache_dir=args.cache_dir,
        active_device=active_device,
        active_dtype=active_dtype,
        status=final_status,
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
