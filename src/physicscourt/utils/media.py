"""Tiny synthetic media fixtures for smoke tests.

The benchmark generator comes later. These helpers exist only to put each model
through one realistic tensor path without pulling in dataset code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SmokeMedia:
    image: Image.Image
    video_frames: List[Image.Image]
    prompt_box_xyxy: list[list[list[int]]]
    prompt_point: list[list[list[list[int]]]]
    prompt_label: list[list[list[int]]]


def make_smoke_media(size: int = 384, frames: int = 16) -> SmokeMedia:
    rng = np.random.default_rng(1729)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    gradient = np.linspace(28, 92, size, dtype=np.uint8)
    base[:, :, 0] = gradient[:, None]
    base[:, :, 1] = 42
    base[:, :, 2] = gradient[None, :]

    pil_frames: list[Image.Image] = []
    radius = max(18, size // 16)
    y = int(size * 0.55)
    start_x = int(size * 0.22)
    end_x = int(size * 0.78)

    for idx in range(frames):
        canvas = Image.fromarray(base.copy(), mode="RGB")
        draw = ImageDraw.Draw(canvas)
        jitter = int(rng.integers(-2, 3))
        x = int(np.interp(idx, [0, frames - 1], [start_x, end_x])) + jitter
        draw.rectangle(
            [int(size * 0.62), int(size * 0.22), int(size * 0.66), int(size * 0.78)],
            fill=(170, 176, 185),
        )
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(235, 92, 72))
        draw.line([0, int(size * 0.72), size, int(size * 0.72)], fill=(95, 114, 122), width=3)
        pil_frames.append(canvas)

    first_x = start_x
    box = [[[first_x - radius, y - radius, first_x + radius, y + radius]]]
    point = [[[[first_x, y]]]]
    label = [[[1]]]
    return SmokeMedia(
        image=pil_frames[0],
        video_frames=pil_frames,
        prompt_box_xyxy=box,
        prompt_point=point,
        prompt_label=label,
    )

