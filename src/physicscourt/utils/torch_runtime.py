"""Torch runtime helpers for 8 GB Apple Silicon discipline."""

from __future__ import annotations

import gc
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Iterator

import psutil
import torch


@dataclass
class RuntimeSnapshot:
    rss_mb: float
    mps_allocated_mb: float | None
    mps_driver_allocated_mb: float | None

    def to_json(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass
class RuntimeDelta:
    seconds: float
    rss_delta_mb: float
    rss_peak_estimate_mb: float
    mps_allocated_delta_mb: float | None
    mps_driver_allocated_delta_mb: float | None

    def to_json(self) -> dict[str, float | None]:
        return asdict(self)


def system_summary() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
    }


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this Python environment.")
    return torch.device(requested)


def clear_torch() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def snapshot() -> RuntimeSnapshot:
    process = psutil.Process()
    rss = process.memory_info().rss / (1024**2)
    mps_allocated = None
    mps_driver = None
    if torch.backends.mps.is_available():
        try:
            mps_allocated = torch.mps.current_allocated_memory() / (1024**2)
        except Exception:
            mps_allocated = None
        try:
            mps_driver = torch.mps.driver_allocated_memory() / (1024**2)
        except Exception:
            mps_driver = None
    return RuntimeSnapshot(rss_mb=rss, mps_allocated_mb=mps_allocated, mps_driver_allocated_mb=mps_driver)


@contextmanager
def measured_runtime() -> Iterator[dict[str, Any]]:
    clear_torch()
    start = snapshot()
    start_time = time.perf_counter()
    holder: dict[str, Any] = {"start": start.to_json()}
    try:
        yield holder
    finally:
        end = snapshot()
        seconds = time.perf_counter() - start_time
        holder["end"] = end.to_json()
        holder["delta"] = RuntimeDelta(
            seconds=seconds,
            rss_delta_mb=end.rss_mb - start.rss_mb,
            rss_peak_estimate_mb=max(start.rss_mb, end.rss_mb),
            mps_allocated_delta_mb=(
                None
                if start.mps_allocated_mb is None or end.mps_allocated_mb is None
                else end.mps_allocated_mb - start.mps_allocated_mb
            ),
            mps_driver_allocated_delta_mb=(
                None
                if start.mps_driver_allocated_mb is None or end.mps_driver_allocated_mb is None
                else end.mps_driver_allocated_mb - start.mps_driver_allocated_mb
            ),
        ).to_json()


def tensor_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype | None) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            if dtype is not None and getattr(value, "is_floating_point", lambda: False)():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved
