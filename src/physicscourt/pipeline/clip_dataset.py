"""Manifest records for clips used by the benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    pair_id: str
    source: str
    split: str
    category: str
    infant_construct: str
    possible: bool
    violation_kind: str
    violation_frame: int | None
    video_path: str
    width: int
    height: int
    fps: int
    num_frames: int
    prompt_box_xywh: list[int]
    prompt_point_xy: list[int]
    object_color_rgb: list[int]
    background_variant: str
    background_source: str
    photorealism_gap_control: bool
    seed: int
    notes: str = ""

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(path: Path, records: list[ClipRecord], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "metadata": metadata,
        "records": [record.to_manifest() for record in records],
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def load_manifest(path: Path) -> list[ClipRecord]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    records = []
    for record in payload["records"]:
        video_path = Path(record["video_path"])
        if not video_path.is_absolute():
            record = dict(record)
            record["video_path"] = str(ROOT / video_path)
        records.append(ClipRecord(**record))
    return records
