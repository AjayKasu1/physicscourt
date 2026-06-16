"""Small Hugging Face cache helpers used by the Phase 0 hardware report."""

from __future__ import annotations

import os
from pathlib import Path


def default_hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()


def repo_cache_dir(model_id: str) -> Path:
    return default_hf_home() / "hub" / f"models--{model_id.replace('/', '--')}"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

