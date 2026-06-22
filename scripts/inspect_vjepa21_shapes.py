#!/usr/bin/env python3
"""Inspect raw V-JEPA 2.1 predictor and target shapes on one real clip.

Run this before a full V-JEPA 2.1 pass. V-JEPA 2.1's predictor may emit a
multi-head deep-supervision tensor rather than a single target-encoder tensor.
If so, predictor-error surprise is not directly comparable with the V-JEPA 2
Hugging Face predictor readout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.detectors.lecun_vjepa21 import _clean_backbone_key  # noqa: E402
from physicscourt.pipeline.clip_dataset import load_manifest  # noqa: E402
from physicscourt.pipeline.video_io import iter_video_frames_rgb  # noqa: E402
from physicscourt.utils.torch_runtime import select_device  # noqa: E402


def load_spec(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return dict(cfg["models"]["vjepa2_1"])


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
    return torch.stack([transform(frame.convert("RGB")) for frame in frames], dim=1).unsqueeze(0)


def first_window(manifest_path: Path, window_frames: int) -> tuple[str, torch.Tensor]:
    records = load_manifest(manifest_path)
    record = next(item for item in records if item.split == "test")
    window: deque[Image.Image] = deque(maxlen=window_frames)
    for frame_rgb in iter_video_frames_rgb(Path(record.video_path)):
        window.append(Image.fromarray(frame_rgb))
        if len(window) == window_frames:
            return record.clip_id, list(window)
    raise RuntimeError(f"clip {record.clip_id} has fewer than {window_frames} frames")


def tensor_shape(value: Any) -> Any:
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [tensor_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_shape(item) for key, item in value.items()}
    return str(type(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "models.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "vjepa21_shape_probe.json")
    args = parser.parse_args()

    spec = load_spec(args.config)
    device = select_device(args.device)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    image_size = int(spec.get("image_size", 384))
    window_frames = int(spec.get("window_frames", 64))

    clip_id, frames = first_window(args.manifest, window_frames)
    pixel_values = preprocess(frames, image_size).to(device=device, dtype=dtype)

    encoder, predictor = torch.hub.load(
        "facebookresearch/vjepa2",
        str(spec["hub_model"]),
        pretrained=False,
        trust_repo=True,
    )
    state_dict = torch.hub.load_state_dict_from_url(str(spec["checkpoint_url"]), map_location="cpu")
    encoder.load_state_dict(_clean_backbone_key(state_dict[str(spec.get("checkpoint_key", "target_encoder"))]), strict=True)
    predictor.load_state_dict(_clean_backbone_key(state_dict["predictor"]), strict=True)
    encoder.to(device=device, dtype=dtype).eval()
    predictor.to(device=device, dtype=dtype).eval()

    with torch.inference_mode():
        target_full = encoder(pixel_values, training=True)
        token_count = int(target_full.shape[1])
        patch_size = int(getattr(encoder, "patch_size", 16))
        grid_size = image_size // patch_size
        tokens_per_tubelet = grid_size * grid_size
        target = torch.arange(token_count - tokens_per_tubelet, token_count, device=device).unsqueeze(0)
        context = torch.arange(0, token_count - tokens_per_tubelet, device=device).unsqueeze(0)
        context_tokens = encoder(pixel_values, masks=[context], training=True)
        predictor_output = predictor(context_tokens, [context], [target], mod="video")

    x_pred = predictor_output[0] if isinstance(predictor_output, tuple) else predictor_output
    actual = target_full[:, target.squeeze(0), :]
    report = {
        "clip_id": clip_id,
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "hub_model": spec["hub_model"],
        "checkpoint_url": spec["checkpoint_url"],
        "pixel_values_shape": list(pixel_values.shape),
        "target_full_shape": list(target_full.shape),
        "context_tokens_shape": tensor_shape(context_tokens),
        "predictor_output_type": type(predictor_output).__name__,
        "predictor_output_shape": tensor_shape(predictor_output),
        "x_pred_shape": tensor_shape(x_pred),
        "actual_shape": list(actual.shape),
        "direct_predictor_error_compatible": bool(torch.is_tensor(x_pred) and tuple(x_pred.shape) == tuple(actual.shape)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
