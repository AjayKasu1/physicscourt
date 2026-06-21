"""Detector A primary: V-JEPA 2 masked latent prediction error.

The historical Detector A score is the spatial mean L2 prediction error over
the target tubelet. That stays as ``raw_score`` for backward compatibility. We
also cache max and top-k token reductions from the same forward pass so the
readout can be audited without changing the model.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoVideoProcessor, VJEPA2Model

from physicscourt.pipeline.clip_dataset import ClipRecord
from physicscourt.pipeline.video_io import iter_video_frames_rgb
from physicscourt.utils.torch_runtime import clear_torch, select_device


@dataclass(frozen=True)
class DetectorTiming:
    clip_id: str
    seconds: float
    num_frames: int
    num_windows: int


class VJEPALatentPredictor:
    """Scores future-tubelet prediction error in V-JEPA latent space."""

    name = "vjepa2"
    reduction_names = ("l2_mean", "l2_max", "l2_topk", "cos_mean", "cos_max", "cos_topk")

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        fp16: bool = True,
        window_frames: int = 16,
        stride_frames: int = 4,
        topk_frac: float = 0.05,
    ) -> None:
        self.model_id = model_id
        self.device = select_device(device)
        self.dtype = torch.float16 if fp16 and self.device.type in {"cuda", "mps"} else torch.float32
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.topk_frac = float(topk_frac)
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = VJEPA2Model.from_pretrained(model_id, dtype=self.dtype, low_cpu_mem_usage=True)
        self.model.to(self.device)
        self.model.eval()

        grid = self.model.config.crop_size // self.model.config.patch_size
        self.tokens_per_tubelet = int(grid * grid)

    def close(self) -> None:
        del self.model
        del self.processor
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
            x = np.array(endpoint_frames, dtype=np.float32)
            score_frames = np.array(endpoint_frames, dtype=np.int32)
            series_arrays = {}
            window_arrays = {}
            frame_grid = np.arange(frame_count, dtype=np.float32)
            for name, values in window_values.items():
                y = np.array(values, dtype=np.float32)
                series_arrays[name] = np.interp(frame_grid, x, y, left=y[0], right=y[-1]).astype(np.float32)
                window_arrays[name] = y
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
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values_videos"].to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            encoded = self.model.get_vision_features(pixel_values)
            token_count = encoded.shape[1]
            target = torch.arange(
                token_count - self.tokens_per_tubelet,
                token_count,
                device=self.device,
            ).unsqueeze(0)
            context = torch.arange(0, token_count - self.tokens_per_tubelet, device=self.device).unsqueeze(0)
            output = self.model(
                pixel_values_videos=pixel_values,
                context_mask=[context],
                target_mask=[target],
                skip_predictor=False,
            )
            pred = output.predictor_output.last_hidden_state.float()
            actual = output.predictor_output.target_hidden_state.float()
            diff = pred - actual
            l2_tokens = torch.linalg.vector_norm(diff, dim=-1).reshape(-1)
            cosine_tokens = (1.0 - torch.nn.functional.cosine_similarity(pred, actual, dim=-1)).reshape(-1)
            l2_mean, l2_max, l2_topk = self._reduce_tokens(l2_tokens)
            cos_mean, cos_max, cos_topk = self._reduce_tokens(cosine_tokens)
        del inputs, pixel_values, encoded, output, pred, actual, diff, l2_tokens, cosine_tokens
        return {
            "l2_mean": l2_mean,
            "l2_max": l2_max,
            "l2_topk": l2_topk,
            "cos_mean": cos_mean,
            "cos_max": cos_max,
            "cos_topk": cos_topk,
        }

    def _reduce_tokens(self, values: torch.Tensor) -> tuple[float, float, float]:
        k = max(1, int(round(float(values.numel()) * self.topk_frac)))
        mean_value = float(values.mean().detach().cpu().item())
        max_value = float(values.max().detach().cpu().item())
        topk_value = float(values.topk(k).values.mean().detach().cpu().item())
        return mean_value, max_value, topk_value
