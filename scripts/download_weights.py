#!/usr/bin/env python3
"""Prefetch Phase 0 checkpoints into the default Hugging Face cache."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.utils.hf_cache import dir_size_bytes, format_bytes, repo_cache_dir
from physicscourt.utils.torch_runtime import clear_torch

console = Console()
IGNORE_PATTERNS = ["original/*"]


def _cache_record(name: str, model_id: str, status: str, error: str | None = None) -> dict[str, Any]:
    cache = repo_cache_dir(model_id)
    size = dir_size_bytes(cache)
    record: dict[str, Any] = {
        "name": name,
        "model_id": model_id,
        "status": status,
        "cache_path": str(cache),
        "cache_exists": cache.exists(),
        "cache_size_bytes": size,
        "cache_size": format_bytes(size),
    }
    if error:
        record["error"] = error
    return record


def prefetch(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    model_id = spec["model_id"]
    try:
        if name == "dino_latent":
            from huggingface_hub import snapshot_download
            from transformers import AutoImageProcessor, AutoModel

            snapshot_download(repo_id=model_id, ignore_patterns=IGNORE_PATTERNS)
            AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            AutoModel.from_pretrained(model_id, low_cpu_mem_usage=True)
        elif name == "depth_anything_v2":
            from huggingface_hub import snapshot_download
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            snapshot_download(repo_id=model_id, ignore_patterns=IGNORE_PATTERNS)
            AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            AutoModelForDepthEstimation.from_pretrained(model_id, low_cpu_mem_usage=True)
        elif name == "sam2":
            from huggingface_hub import snapshot_download
            from transformers import Sam2Model, Sam2Processor

            snapshot_download(repo_id=model_id, ignore_patterns=IGNORE_PATTERNS)
            Sam2Processor.from_pretrained(model_id)
            Sam2Model.from_pretrained(model_id, low_cpu_mem_usage=True)
        elif name == "vjepa2":
            from huggingface_hub import snapshot_download
            from transformers import AutoVideoProcessor, VJEPA2Model

            snapshot_download(repo_id=model_id, ignore_patterns=IGNORE_PATTERNS)
            AutoVideoProcessor.from_pretrained(model_id)
            VJEPA2Model.from_pretrained(model_id, low_cpu_mem_usage=True)
        elif name == "cotracker":
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id=model_id, filename=spec.get("checkpoint_file", "scaled_offline.pth"))
        else:
            return _cache_record(name, model_id, "skipped", f"Unknown model key {name}")
        clear_torch()
        return _cache_record(name, model_id, "ok")
    except Exception as exc:
        clear_torch()
        return _cache_record(name, model_id, "failed", f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--only", choices=["dino_latent", "depth_anything_v2", "sam2", "vjepa2", "cotracker"], default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": [],
    }
    for name, spec in cfg["models"].items():
        if args.only is not None and args.only != name:
            continue
        model_id = spec["model_id"]
        console.rule(f"[bold]Prefetch: {name}")
        record = prefetch(name, spec)
        report["models"].append(record)
        console.print(record)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    console.print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
