#!/usr/bin/env python3
"""Run one local inference smoke test per Phase 0 model.

The script intentionally loads and releases one model at a time. That is slower,
but it is the only honest default for the first target machine: an 8 GB M2.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
import numpy as np
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.utils.hf_cache import dir_size_bytes, format_bytes, repo_cache_dir
from physicscourt.utils.media import make_smoke_media
from physicscourt.utils.torch_runtime import (
    clear_torch,
    measured_runtime,
    select_device,
    system_summary,
    tensor_batch_to_device,
)

console = Console()


class SmokeFailure(RuntimeError):
    pass


def _dtype_attempts(device: torch.device, fp16: bool) -> list[torch.dtype]:
    if fp16 and device.type in {"cuda", "mps"}:
        return [torch.float16, torch.float32]
    return [torch.float32]


def _record_cache(model_id: str) -> dict[str, Any]:
    cache = repo_cache_dir(model_id)
    size = dir_size_bytes(cache)
    return {
        "model_id": model_id,
        "cache_path": str(cache),
        "cache_exists": cache.exists(),
        "cache_size_bytes": size,
        "cache_size": format_bytes(size),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    tmp.replace(path)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name}")


def _attempt_worker(
    fn: Callable[[str, torch.device, torch.dtype], dict[str, Any]],
    model_id: str,
    device_type: str,
    dtype_name: str,
    result_queue: Any,
) -> None:
    device = torch.device(device_type)
    dtype = _dtype_from_name(dtype_name)
    try:
        with measured_runtime() as timing:
            output = fn(model_id, device, dtype)
        result_queue.put({"status": "ok", "timing": timing, "output": output})
    except Exception as exc:
        result_queue.put(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc(limit=8),
            }
        )
    finally:
        clear_torch()


def _run_attempt_in_child(
    fn: Callable[[str, torch.device, torch.dtype], dict[str, Any]],
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    timeout_seconds: int,
) -> dict[str, Any]:
    dtype_name = str(dtype).replace("torch.", "")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_attempt_worker, args=(fn, model_id, device.type, dtype_name, result_queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        return {
            "status": "failed",
            "error_type": "TimeoutError",
            "error": f"Smoke test exceeded {timeout_seconds} seconds.",
            "timing": {"timeout_seconds": timeout_seconds},
        }

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {
            "status": "failed",
            "error_type": "ChildProcessError",
            "error": f"Smoke worker exited with code {process.exitcode} without returning a result.",
        }
    return result


def _run_with_fallback(
    name: str,
    model_id: str,
    requested_device: str,
    fp16: bool,
    fn: Callable[[str, torch.device, torch.dtype], dict[str, Any]],
    timeout_seconds: int,
    stop_after_timeout: bool,
) -> dict[str, Any]:
    preferred = select_device(requested_device)
    devices = [preferred]
    if preferred.type != "cpu":
        devices.append(torch.device("cpu"))

    attempts: list[dict[str, Any]] = []
    for device in devices:
        for dtype in _dtype_attempts(device, fp16):
            clear_torch()
            attempt: dict[str, Any] = {
                "device": device.type,
                "dtype": str(dtype).replace("torch.", ""),
            }
            result = _run_attempt_in_child(fn, model_id, device, dtype, timeout_seconds)
            if result["status"] == "ok":
                cache_record = _record_cache(model_id)
                attempt.update(
                    {
                        "status": "ok",
                        "timing": result["timing"],
                        "output": result["output"],
                    }
                )
                attempt.update(cache_record)
                clear_torch()
                return {"name": name, "status": "ok", "attempts": attempts + [attempt], **cache_record}

            attempt.update(result)
            attempts.append(attempt)
            clear_torch()
            if result.get("error_type") == "TimeoutError" and stop_after_timeout:
                return {"name": name, "status": "failed", "attempts": attempts, **_record_cache(model_id)}
    return {"name": name, "status": "failed", "attempts": attempts, **_record_cache(model_id)}


def smoke_dino(model_id: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModel

    media = make_smoke_media(size=384, frames=16)
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id,
        dtype=dtype if device.type != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = processor(images=media.image, return_tensors="pt")
    inputs = tensor_batch_to_device(inputs, device, dtype if device.type != "cpu" else None)
    with torch.inference_mode():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().float().cpu()

    del outputs, model, processor, inputs
    return {"embedding_shape": list(embedding.shape), "embedding_norm": float(embedding.norm().item())}


def smoke_depth(model_id: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    media = make_smoke_media(size=384, frames=16)
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id,
        dtype=dtype if device.type != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = processor(images=media.image, return_tensors="pt")
    inputs = tensor_batch_to_device(inputs, device, dtype if device.type != "cpu" else None)
    with torch.inference_mode():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.detach().float().cpu()

    del outputs, model, processor, inputs
    return {
        "depth_shape": list(depth.shape),
        "depth_min": float(depth.min().item()),
        "depth_max": float(depth.max().item()),
    }


def smoke_sam2(model_id: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from transformers import Sam2Model, Sam2Processor

    media = make_smoke_media(size=384, frames=16)
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(
        model_id,
        dtype=dtype if device.type != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = processor(images=media.image, input_boxes=media.prompt_box_xyxy, return_tensors="pt")
    inputs = tensor_batch_to_device(inputs, device, dtype if device.type != "cpu" else None)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]

    del outputs, model, processor, inputs
    return {"mask_shape": list(masks.shape), "mask_mean": float(masks.float().mean().item())}


def smoke_vjepa2(model_id: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from transformers import AutoVideoProcessor, VJEPA2Model

    media = make_smoke_media(size=384, frames=16)
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = VJEPA2Model.from_pretrained(
        model_id,
        dtype=dtype if device.type != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    first = media.video_frames[0]
    inputs = processor(first, return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"]
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(1)
    pixel_values = pixel_values.repeat(1, 16, 1, 1, 1)
    batch = {"pixel_values_videos": pixel_values}
    batch = tensor_batch_to_device(batch, device, dtype if device.type != "cpu" else None)

    with torch.inference_mode():
        features = model.get_vision_features(**batch).detach().float().cpu()

    feature_shape = list(features.shape)
    del features, model, processor, inputs, batch
    return {"vision_feature_shape": feature_shape}


def smoke_cotracker(model_id: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from cotracker.predictor import CoTrackerPredictor

    media = make_smoke_media(size=96, frames=8)
    arrays = []
    for frame in media.video_frames[:8]:
        resized = frame.resize((96, 96))
        arrays.append(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1))
    video = torch.from_numpy(np.stack(arrays, axis=0)).unsqueeze(0)
    video = video.to(device=device, dtype=dtype if device.type != "cpu" else torch.float32)
    queries = torch.tensor([[[0.0, 28.0, 52.0], [0.0, 48.0, 52.0], [0.0, 68.0, 52.0]]], device=device)
    if device.type != "cpu":
        queries = queries.to(dtype=dtype)

    checkpoint = hf_hub_download(repo_id=model_id, filename="scaled_offline.pth")
    model = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
    model.to(device=device, dtype=dtype if device.type != "cpu" else torch.float32)
    model.eval()

    with torch.inference_mode():
        tracks, visibility = model(video, queries=queries)
        tracks_cpu = tracks.detach().float().cpu()
        visibility_cpu = visibility.detach().cpu()

    del tracks, visibility, model, video, queries
    return {
        "checkpoint": checkpoint,
        "tracks_shape": list(tracks_cpu.shape),
        "visibility_shape": list(visibility_cpu.shape),
        "first_track_xy": [float(v) for v in tracks_cpu[0, 0, 0].tolist()],
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--only", choices=["dino_latent", "depth_anything_v2", "sam2", "vjepa2", "cotracker"], default=None)
    parser.add_argument("--offline", action="store_true", help="Require local Hugging Face cache only.")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cfg = load_config(args.config)
    models = cfg["models"]
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase0",
        "hardware_policy": cfg["hardware"],
        "system": system_summary(),
        "models": [],
        "fallbacks": [],
        "blocking": [],
        "completed": False,
    }
    write_report(report, args.report)

    runners: list[tuple[str, str, Callable[[str, torch.device, torch.dtype], dict[str, Any]], int, bool]] = [
        (
            "dino_latent",
            models["dino_latent"]["model_id"],
            smoke_dino,
            int(models["dino_latent"].get("smoke_timeout_seconds", 180)),
            bool(models["dino_latent"].get("stop_after_timeout", False)),
        ),
        (
            "depth_anything_v2",
            models["depth_anything_v2"]["model_id"],
            smoke_depth,
            int(models["depth_anything_v2"].get("smoke_timeout_seconds", 180)),
            bool(models["depth_anything_v2"].get("stop_after_timeout", False)),
        ),
        (
            "sam2",
            models["sam2"]["model_id"],
            smoke_sam2,
            int(models["sam2"].get("smoke_timeout_seconds", 240)),
            bool(models["sam2"].get("stop_after_timeout", False)),
        ),
        (
            "vjepa2",
            models["vjepa2"]["model_id"],
            smoke_vjepa2,
            int(models["vjepa2"].get("smoke_timeout_seconds", 240)),
            bool(models["vjepa2"].get("stop_after_timeout", True)),
        ),
        (
            "cotracker",
            models["cotracker"]["model_id"],
            smoke_cotracker,
            int(models["cotracker"].get("smoke_timeout_seconds", 300)),
            bool(models["cotracker"].get("stop_after_timeout", False)),
        ),
    ]

    for name, model_id, runner, timeout_seconds, stop_after_timeout in runners:
        if args.only is not None and args.only != name:
            continue
        console.rule(f"[bold]Smoke: {name}")
        result = _run_with_fallback(
            name,
            model_id,
            args.device,
            args.fp16,
            runner,
            timeout_seconds=timeout_seconds,
            stop_after_timeout=stop_after_timeout,
        )
        report["models"].append(result)
        if result["status"] != "ok":
            if name == "vjepa2":
                report["fallbacks"].append(
                    {
                        "from": "vjepa2",
                        "to": "dino_latent",
                        "reason": "V-JEPA 2 did not complete the standalone Phase 0 smoke test inside the 8 GB timeout envelope.",
                        "hardware_note": "DINOv2 latent extrapolation is Detector A's primary path for this machine profile.",
                    }
                )
            else:
                report["blocking"].append(
                    {
                        "model": name,
                        "reason": f"{name} failed its Phase 0 smoke test.",
                    }
                )
        write_report(report, args.report)
        console.print(result["status"], result.get("attempts", [])[-1].get("device", "n/a"))

    report["completed"] = not report["blocking"]
    write_report(report, args.report)
    console.print(f"Wrote {args.report}")
    if report["blocking"]:
        raise SystemExit("Phase 0 blocked; see results/environment_report.json")


if __name__ == "__main__":
    main()
