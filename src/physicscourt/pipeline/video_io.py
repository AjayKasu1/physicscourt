"""Streaming video I/O for PhysicsCourt clips.

Phase 1 uses OpenCV's local encoder so synthetic generation does not depend on
an external ffmpeg binary. Later overlay export can still use ffmpeg when it is
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int


def write_video_mp4(frames_rgb: Iterable[np.ndarray], path: Path, fps: int) -> VideoProbe:
    """Write RGB uint8 frames to an MP4 file without materializing the sequence twice."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    width = 0
    height = 0
    frame_count = 0

    try:
        for frame in frames_rgb:
            if frame.dtype != np.uint8:
                raise ValueError(f"Expected uint8 frame, got {frame.dtype}")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 RGB frame, got shape {frame.shape}")

            if writer is None:
                height, width = int(frame.shape[0]), int(frame.shape[1])
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"OpenCV could not open MP4 writer for {path}")
            elif frame.shape[0] != height or frame.shape[1] != width:
                raise ValueError("All video frames must have the same dimensions")

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise ValueError("Cannot write an empty video")
    return VideoProbe(path=path, fps=float(fps), width=width, height=height, frame_count=frame_count)


def iter_video_frames_rgb(path: Path) -> Iterator[np.ndarray]:
    """Yield RGB frames from a video file one at a time."""

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video {path}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_frame_rgb(path: Path, frame_index: int) -> np.ndarray:
    """Read one RGB frame by index."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok:
            raise IndexError(f"Could not read frame {frame_index} from {path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def probe_video(path: Path) -> VideoProbe:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video {path}")
    try:
        return VideoProbe(
            path=path,
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()

