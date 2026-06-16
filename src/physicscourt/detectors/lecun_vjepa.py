"""Detector A primary: V-JEPA 2 masked latent prediction error."""

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

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        fp16: bool = True,
        window_frames: int = 16,
        stride_frames: int = 4,
    ) -> None:
        self.model_id = model_id
        self.device = select_device(device)
        self.dtype = torch.float16 if fp16 and self.device.type in {"cuda", "mps"} else torch.float32
        self.window_frames = window_frames
        self.stride_frames = stride_frames
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
        l2_scores: list[float] = []
        cosine_scores: list[float] = []

        for frame_rgb in iter_video_frames_rgb(Path(clip.video_path)):
            window.append(Image.fromarray(frame_rgb))
            frame_index = frame_count
            frame_count += 1
            if len(window) < self.window_frames:
                continue
            if (frame_index - (self.window_frames - 1)) % self.stride_frames != 0 and frame_index != clip.num_frames - 1:
                continue
            l2, cosine = self._score_window(list(window))
            endpoint_frames.append(frame_index)
            l2_scores.append(l2)
            cosine_scores.append(cosine)

        if not endpoint_frames:
            raw = np.zeros((frame_count,), dtype=np.float32)
            score_frames = np.zeros((0,), dtype=np.int32)
            window_scores = np.zeros((0,), dtype=np.float32)
            cosine_window_scores = np.zeros((0,), dtype=np.float32)
        else:
            x = np.array(endpoint_frames, dtype=np.float32)
            y = np.array(l2_scores, dtype=np.float32)
            raw = np.interp(np.arange(frame_count, dtype=np.float32), x, y, left=y[0], right=y[-1]).astype(np.float32)
            score_frames = np.array(endpoint_frames, dtype=np.int32)
            window_scores = y
            cosine_window_scores = np.array(cosine_scores, dtype=np.float32)

        timing = DetectorTiming(
            clip_id=clip.clip_id,
            seconds=float(time.perf_counter() - started),
            num_frames=frame_count,
            num_windows=len(endpoint_frames),
        )
        extras = {
            "score_frame_indices": score_frames,
            "window_scores": window_scores,
            "window_cosine_scores": cosine_window_scores,
        }
        return raw, extras, timing

    def _score_window(self, frames: list[Image.Image]) -> tuple[float, float]:
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
            l2 = torch.linalg.vector_norm(diff, dim=-1).mean()
            cosine = 1.0 - torch.nn.functional.cosine_similarity(pred, actual, dim=-1).mean()
            score = float(l2.detach().cpu().item())
            cosine_score = float(cosine.detach().cpu().item())
        del inputs, pixel_values, encoded, output, pred, actual, diff, l2, cosine
        return score, cosine_score
