"""Detector A baseline: DINOv2 latent linear extrapolation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from physicscourt.pipeline.clip_dataset import ClipRecord
from physicscourt.pipeline.video_io import iter_video_frames_rgb
from physicscourt.utils.torch_runtime import clear_torch, select_device, tensor_batch_to_device


@dataclass(frozen=True)
class DetectorTiming:
    clip_id: str
    seconds: float
    num_frames: int
    num_batches: int


class DINOLatentExtrapolator:
    """Cheap control: predict next DINO embedding by linear extrapolation."""

    name = "dino_latent"

    def __init__(self, model_id: str, device: str = "auto", fp16: bool = True, batch_size: int = 12) -> None:
        self.model_id = model_id
        self.device = select_device(device)
        self.dtype = torch.float16 if fp16 and self.device.type in {"cuda", "mps"} else torch.float32
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype, low_cpu_mem_usage=True)
        self.model.to(self.device)
        self.model.eval()

    def close(self) -> None:
        del self.model
        del self.processor
        clear_torch()

    def score_clip(self, clip: ClipRecord, history: int = 4) -> tuple[np.ndarray, dict[str, np.ndarray], DetectorTiming]:
        started = time.perf_counter()
        embeddings: list[np.ndarray] = []
        batch: list[Image.Image] = []
        num_batches = 0

        for frame_rgb in iter_video_frames_rgb(Path(clip.video_path)):
            batch.append(Image.fromarray(frame_rgb))
            if len(batch) == self.batch_size:
                embeddings.extend(self._embed_batch(batch))
                batch = []
                num_batches += 1
        if batch:
            embeddings.extend(self._embed_batch(batch))
            num_batches += 1

        emb = np.stack(embeddings, axis=0).astype(np.float32)
        raw = np.zeros((emb.shape[0],), dtype=np.float32)
        for idx in range(1, emb.shape[0]):
            if idx < history:
                pred = emb[idx - 1]
            else:
                slope = (emb[idx - 1] - emb[idx - history]) / float(history - 1)
                pred = emb[idx - 1] + slope
            raw[idx] = float(np.linalg.norm(emb[idx] - pred))
        if raw.size > 1:
            raw[0] = raw[1]

        timing = DetectorTiming(
            clip_id=clip.clip_id,
            seconds=float(time.perf_counter() - started),
            num_frames=int(emb.shape[0]),
            num_batches=num_batches,
        )
        extras = {"embedding_dim": np.array([emb.shape[1]], dtype=np.int32)}
        return raw, extras, timing

    def _embed_batch(self, frames: list[Image.Image]) -> list[np.ndarray]:
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = tensor_batch_to_device(inputs, self.device, self.dtype if self.device.type != "cpu" else None)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().float().cpu().numpy()
        del inputs, outputs
        return [row for row in embedding]
