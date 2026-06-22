"""Detector A follow-up: official V-JEPA 2.1 masked latent prediction error.

This uses Meta's official ``facebookresearch/vjepa2`` PyTorch Hub code path
instead of the Hugging Face VJEPA2Model wrapper used for V-JEPA 2. The initial
PhysicsCourt run used V-JEPA 2. This detector exists only for the V-JEPA 2.1
follow-up audit.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from physicscourt.pipeline.clip_dataset import ClipRecord
from physicscourt.pipeline.video_io import iter_video_frames_rgb
from physicscourt.utils.torch_runtime import clear_torch, select_device


@dataclass(frozen=True)
class DetectorTiming:
    clip_id: str
    seconds: float
    num_frames: int
    num_windows: int


def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


class OfficialVJEPA21Predictor:
    """Scores masked future-tubelet prediction error with V-JEPA 2.1."""

    name = "vjepa2_1"
    reduction_names = ("l2_mean", "l2_max", "l2_topk", "cos_mean", "cos_max", "cos_topk")

    def __init__(
        self,
        hub_model: str,
        checkpoint_url: str,
        device: str = "auto",
        fp16: bool = False,
        image_size: int = 384,
        window_frames: int = 64,
        stride_frames: int = 4,
        topk_frac: float = 0.05,
        checkpoint_key: str = "target_encoder",
        repo: str = "facebookresearch/vjepa2",
    ) -> None:
        self.hub_model = hub_model
        self.checkpoint_url = checkpoint_url
        self.device = select_device(device)
        self.dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        self.image_size = int(image_size)
        self.window_frames = int(window_frames)
        self.stride_frames = int(stride_frames)
        self.topk_frac = float(topk_frac)
        self.checkpoint_key = checkpoint_key

        self.encoder, self.predictor = torch.hub.load(
            repo,
            hub_model,
            pretrained=False,
            trust_repo=True,
        )
        self._load_checkpoint()
        self.encoder.to(device=self.device, dtype=self.dtype)
        self.predictor.to(device=self.device, dtype=self.dtype)
        self.encoder.eval()
        self.predictor.eval()

        self.patch_size = int(getattr(self.encoder, "patch_size", 16))
        self.tubelet_size = int(getattr(self.encoder, "tubelet_size", 2))
        self.grid_size = self.image_size // self.patch_size
        self.tokens_per_tubelet = self.grid_size * self.grid_size

    def _load_checkpoint(self) -> None:
        state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url, map_location="cpu")
        if self.checkpoint_key not in state_dict:
            raise KeyError(f"checkpoint key {self.checkpoint_key!r} not found in {self.checkpoint_url}")
        if "predictor" not in state_dict:
            raise KeyError(f"predictor weights not found in {self.checkpoint_url}")
        self.encoder.load_state_dict(_clean_backbone_key(state_dict[self.checkpoint_key]), strict=True)
        self.predictor.load_state_dict(_clean_backbone_key(state_dict["predictor"]), strict=True)
        del state_dict

    def close(self) -> None:
        del self.encoder
        del self.predictor
        clear_torch()

    def score_clip(self, clip: ClipRecord) -> tuple[np.ndarray, dict[str, np.ndarray], DetectorTiming]:
        started = time.perf_counter()
        window: deque[Image.Image] = deque(maxlen=self.window_frames)
        frame_count = 0
        endpoint_frames: list[int] = []
        window_values: dict[str, list[float]] = {name: [] for name in self.reduction_names}

        for frame_rgb in iter_video_frames_rgb(Path(clip.video_path)):
            window.append(Image.fromarray(frame_rgb))
            frame_index = frame_count
            frame_count += 1
            if len(window) < self.window_frames:
                continue
            if (frame_index - (self.window_frames - 1)) % self.stride_frames != 0 and frame_index != clip.num_frames - 1:
                continue
            reductions = self._score_window(list(window))
            endpoint_frames.append(frame_index)
            for name in self.reduction_names:
                window_values[name].append(reductions[name])

        if not endpoint_frames:
            raw = np.zeros((frame_count,), dtype=np.float32)
            score_frames = np.zeros((0,), dtype=np.int32)
            series_arrays = {name: raw.copy() for name in self.reduction_names}
            window_arrays = {name: np.zeros((0,), dtype=np.float32) for name in self.reduction_names}
        else:
            x_axis = np.array(endpoint_frames, dtype=np.float32)
            frame_grid = np.arange(frame_count, dtype=np.float32)
            score_frames = np.array(endpoint_frames, dtype=np.int32)
            series_arrays = {}
            window_arrays = {}
            for name, values in window_values.items():
                y_axis = np.array(values, dtype=np.float32)
                series_arrays[name] = np.interp(frame_grid, x_axis, y_axis, left=y_axis[0], right=y_axis[-1]).astype(
                    np.float32
                )
                window_arrays[name] = y_axis
            raw = series_arrays["l2_mean"]

        timing = DetectorTiming(
            clip_id=clip.clip_id,
            seconds=float(time.perf_counter() - started),
            num_frames=frame_count,
            num_windows=len(endpoint_frames),
        )
        extras = {
            "score_frame_indices": score_frames,
            "window_scores": window_arrays["l2_mean"],
            "window_cosine_scores": window_arrays["cos_mean"],
        }
        extras.update({f"series_{name}": series for name, series in series_arrays.items()})
        extras.update({f"window_{name}": scores for name, scores in window_arrays.items()})
        return raw, extras, timing

    def _score_window(self, frames: list[Image.Image]) -> dict[str, float]:
        pixel_values = self._preprocess(frames).to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            target_full = self.encoder(pixel_values, training=True)
            token_count = int(target_full.shape[1])
            target = torch.arange(token_count - self.tokens_per_tubelet, token_count, device=self.device).unsqueeze(0)
            context = torch.arange(0, token_count - self.tokens_per_tubelet, device=self.device).unsqueeze(0)
            context_tokens = self.encoder(pixel_values, masks=[context], training=True)
            pred_out = self.predictor(context_tokens, [context], [target], mod="video")
            pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            actual = target_full[:, target.squeeze(0), :]
            if pred.shape != actual.shape:
                raise RuntimeError(
                    f"V-JEPA 2.1 predictor/target shape mismatch: pred={tuple(pred.shape)} "
                    f"actual={tuple(actual.shape)}. Use a non-distilled checkpoint such as ViT-g."
                )
            diff = pred.float() - actual.float()
            l2_tokens = torch.linalg.vector_norm(diff, dim=-1).reshape(-1)
            cosine_tokens = (1.0 - torch.nn.functional.cosine_similarity(pred.float(), actual.float(), dim=-1)).reshape(
                -1
            )
            l2_mean, l2_max, l2_topk = self._reduce_tokens(l2_tokens)
            cos_mean, cos_max, cos_topk = self._reduce_tokens(cosine_tokens)

        del pixel_values, target_full, target, context, context_tokens, pred_out, pred, actual, diff, l2_tokens
        del cosine_tokens
        return {
            "l2_mean": l2_mean,
            "l2_max": l2_max,
            "l2_topk": l2_topk,
            "cos_mean": cos_mean,
            "cos_max": cos_max,
            "cos_topk": cos_topk,
        }

    def _preprocess(self, frames: list[Image.Image]) -> torch.Tensor:
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        stacked = torch.stack([transform(frame.convert("RGB")) for frame in frames], dim=1)
        return stacked.unsqueeze(0)

    def _reduce_tokens(self, values: torch.Tensor) -> tuple[float, float, float]:
        k = max(1, int(round(float(values.numel()) * self.topk_frac)))
        return (
            float(values.mean().detach().cpu().item()),
            float(values.max().detach().cpu().item()),
            float(values.topk(k).values.mean().detach().cpu().item()),
        )
