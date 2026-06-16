"""Score calibration and metric helpers for detector outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScoreSummary:
    clip_level_score: float
    predicted_frame: int


def median_filter_1d(values: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 1:
        return values.astype(np.float32, copy=True)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pad = kernel_size // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    for idx in range(len(values)):
        out[idx] = float(np.median(padded[idx : idx + kernel_size]))
    return out


def zscore(raw: np.ndarray, mean: float, std: float) -> np.ndarray:
    safe_std = std if std > 1e-8 else 1.0
    return ((raw.astype(np.float32) - mean) / safe_std).astype(np.float32)


def summarize_scores(smoothed: np.ndarray) -> ScoreSummary:
    if smoothed.size == 0:
        return ScoreSummary(clip_level_score=0.0, predicted_frame=0)
    frame = int(np.argmax(smoothed))
    return ScoreSummary(clip_level_score=float(smoothed[frame]), predicted_frame=frame)


def first_sustained_crossing(
    values: np.ndarray,
    threshold: float,
    min_consecutive: int,
    min_prior_below: int = 0,
) -> int | None:
    """Return the first sustained upward threshold crossing."""
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")
    if min_prior_below < 0:
        raise ValueError("min_prior_below must be non-negative")
    run_length = 0
    below_run_length = min_prior_below if min_prior_below == 0 else 0
    for index, value in enumerate(values.astype(np.float32)):
        if float(value) >= threshold:
            if below_run_length >= min_prior_below:
                run_length += 1
            else:
                run_length = 0
            if run_length >= min_consecutive:
                return index - min_consecutive + 1
        else:
            below_run_length += 1
            run_length = 0
    return None


def fit_normal_stats(score_arrays: Iterable[np.ndarray]) -> dict[str, float]:
    values = [arr.astype(np.float32).reshape(-1) for arr in score_arrays if arr.size]
    if not values:
        raise ValueError("Cannot fit calibration stats without scores")
    joined = np.concatenate(values)
    return {
        "mean": float(joined.mean()),
        "std": float(joined.std(ddof=0)),
        "num_values": int(joined.size),
        "min": float(joined.min()),
        "max": float(joined.max()),
    }


def save_score_cache(
    path: Path,
    *,
    raw_score: np.ndarray,
    calibrated_score: np.ndarray | None = None,
    smoothed_score: np.ndarray | None = None,
    metadata: dict[str, object],
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, object] = {
        "raw_score": raw_score.astype(np.float32),
        "metadata_json": np.array([metadata], dtype=object),
    }
    if calibrated_score is not None:
        arrays["calibrated_score"] = calibrated_score.astype(np.float32)
    if smoothed_score is not None:
        arrays["smoothed_score"] = smoothed_score.astype(np.float32)
    if extra_arrays:
        arrays.update(extra_arrays)
    np.savez_compressed(path, **arrays)


def load_score_cache(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        loaded: dict[str, object] = {key: data[key] for key in data.files}
    if "metadata_json" in loaded:
        loaded["metadata"] = loaded["metadata_json"][0]
    return loaded
