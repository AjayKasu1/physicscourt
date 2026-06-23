#!/usr/bin/env python3
"""Cache pooled frozen V-JEPA 2.1 embeddings for a linear probe.

This script intentionally does not compute a prediction-surprise score. It
extracts one pooled representation per clip from the frozen V-JEPA 2.1 encoder
so a tiny supervised probe can test whether possible/impossible information is
present in the representation at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from rich.progress import track

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.detectors.lecun_vjepa21 import _clean_backbone_key  # noqa: E402
from physicscourt.pipeline.clip_dataset import ClipRecord, load_manifest  # noqa: E402
from physicscourt.pipeline.video_io import iter_video_frames_rgb  # noqa: E402
from physicscourt.utils.torch_runtime import clear_torch, select_device  # noqa: E402


def load_model_spec(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return dict(cfg["models"]["vjepa2_1"])


def pair_violation_frames(records: list[ClipRecord]) -> dict[str, int]:
    out = {}
    for record in records:
        if record.violation_frame is not None:
            out[record.pair_id] = int(record.violation_frame)
    return out


def read_window(record: ClipRecord, window_frames: int, pair_tstar: int | None) -> tuple[list[Image.Image], int, int]:
    frames = [Image.fromarray(frame) for frame in iter_video_frames_rgb(Path(record.video_path))]
    if not frames:
        raise RuntimeError(f"no frames found in {record.video_path}")

    while len(frames) < window_frames:
        frames.append(frames[-1].copy())

    anchor = int(pair_tstar if pair_tstar is not None else len(frames) // 2)
    start = anchor - window_frames // 2
    start = max(0, min(start, len(frames) - window_frames))
    end = start + window_frames
    return frames[start:end], start, end - 1


def preprocess(frames: list[Image.Image], image_size: int) -> torch.Tensor:
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    stacked = torch.stack([transform(frame.convert("RGB")) for frame in frames], dim=1)
    return stacked.unsqueeze(0)


class VJEPA21EncoderCache:
    def __init__(self, spec: dict[str, Any], device: str, fp16: bool) -> None:
        self.device = select_device(device)
        self.dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        self.image_size = int(spec.get("image_size", 384))
        self.window_frames = int(spec.get("window_frames", 64))
        self.hub_model = str(spec["hub_model"])
        self.checkpoint_url = str(spec["checkpoint_url"])
        self.checkpoint_key = str(spec.get("checkpoint_key", "target_encoder"))
        repo = str(spec.get("repo", "facebookresearch/vjepa2"))

        encoder, predictor = torch.hub.load(repo, self.hub_model, pretrained=False, trust_repo=True)
        del predictor
        state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url, map_location="cpu")
        if self.checkpoint_key not in state_dict:
            raise KeyError(f"checkpoint key {self.checkpoint_key!r} not found in {self.checkpoint_url}")
        encoder.load_state_dict(_clean_backbone_key(state_dict[self.checkpoint_key]), strict=True)
        del state_dict

        self.encoder = encoder.to(device=self.device, dtype=self.dtype)
        self.encoder.eval()
        self.patch_size = int(getattr(self.encoder, "patch_size", 16))
        self.grid_size = self.image_size // self.patch_size
        self.tokens_per_tubelet = self.grid_size * self.grid_size

    def close(self) -> None:
        del self.encoder
        clear_torch()

    def encode(self, frames: list[Image.Image]) -> dict[str, np.ndarray]:
        pixel_values = preprocess(frames, self.image_size).to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            tokens = self.encoder(pixel_values, training=True).float().squeeze(0)
            target_tokens = tokens[-self.tokens_per_tubelet :]
            context_tokens = tokens[: -self.tokens_per_tubelet]

            all_mean = tokens.mean(dim=0)
            context_mean = context_tokens.mean(dim=0)
            target_mean = target_tokens.mean(dim=0)
            delta_mean = target_mean - context_mean
            vector = torch.cat([all_mean, context_mean, target_mean, delta_mean], dim=0)

        out = {
            "embedding_all_mean": all_mean.detach().cpu().numpy().astype(np.float32),
            "embedding_context_mean": context_mean.detach().cpu().numpy().astype(np.float32),
            "embedding_target_mean": target_mean.detach().cpu().numpy().astype(np.float32),
            "embedding_delta_mean": delta_mean.detach().cpu().numpy().astype(np.float32),
            "embedding_vector": vector.detach().cpu().numpy().astype(np.float32),
            "embedding_dim": np.array([int(all_mean.numel())], dtype=np.int32),
            "tokens_per_tubelet": np.array([int(self.tokens_per_tubelet)], dtype=np.int32),
            "token_count": np.array([int(tokens.shape[0])], dtype=np.int32),
        }
        del pixel_values, tokens, target_tokens, context_tokens, all_mean, context_mean, target_mean, delta_mean, vector
        return out


def cache_path(features_dir: Path, clip: ClipRecord) -> Path:
    return features_dir / "vjepa2_1" / f"{clip.clip_id}.npz"


def save_embedding_cache(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = dict(arrays)
    payload["metadata_json"] = np.array([metadata], dtype=object)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--models-config", type=Path, default=ROOT / "config" / "models.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features_vjepa21_probe_l4_fp32")
    parser.add_argument("--timing-report", type=Path, default=ROOT / "results" / "vjepa21_probe_embeddings_l4_fp32_timing.json")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]
    tstar_by_pair = pair_violation_frames(records)

    spec = load_model_spec(args.models_config)
    encoder = VJEPA21EncoderCache(spec, args.device, args.fp16)
    timings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if args.timing_report.exists() and not args.force:
        with args.timing_report.open("r", encoding="utf-8") as fh:
            prior = json.load(fh)
        timings = list(prior.get("timings", []))
        errors = list(prior.get("errors", []))
    completed = {str(item.get("clip_id")) for item in timings if item.get("clip_id")}
    failed = {str(item.get("clip_id")) for item in errors if item.get("clip_id")}

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": "vjepa2_1",
        "feature_kind": "pooled_encoder_embeddings",
        "manifest": str(args.manifest),
        "features_dir": str(args.features_dir),
        "timings": timings,
        "errors": errors,
    }

    def write_report() -> None:
        args.timing_report.parent.mkdir(parents=True, exist_ok=True)
        args.timing_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        for record in track(records, description="vjepa21_probe_embeddings"):
            if record.clip_id in completed and not args.force and cache_path(args.features_dir, record).exists():
                continue
            if record.clip_id in failed and not args.force:
                continue
            started = time.perf_counter()
            try:
                pair_tstar = tstar_by_pair.get(record.pair_id)
                frames, start_frame, end_frame = read_window(record, encoder.window_frames, pair_tstar)
                arrays = encoder.encode(frames)
                metadata = record.to_manifest()
                metadata.update(
                    {
                        "feature_kind": "pooled_encoder_embeddings",
                        "model": "vjepa2_1",
                        "window_start_frame": int(start_frame),
                        "window_end_frame": int(end_frame),
                        "pair_violation_frame": int(pair_tstar) if pair_tstar is not None else None,
                    }
                )
                save_embedding_cache(cache_path(args.features_dir, record), arrays, metadata)
                timings.append(
                    {
                        "clip_id": record.clip_id,
                        "seconds": float(time.perf_counter() - started),
                        "window_start_frame": int(start_frame),
                        "window_end_frame": int(end_frame),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"clip_id": record.clip_id, "error": repr(exc)})
                if not args.continue_on_error:
                    raise
            write_report()
    finally:
        encoder.close()
        write_report()

    print(
        {
            "records": len(timings),
            "errors": len(errors),
            "features_dir": str(args.features_dir),
            "timing_report": str(args.timing_report),
        }
    )


if __name__ == "__main__":
    main()
