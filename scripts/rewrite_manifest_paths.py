#!/usr/bin/env python3
"""Rewrite copied manifest video paths to this checkout without changing clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def relative_from_data(path_text: str) -> Path:
    path = Path(path_text)
    parts = path.parts
    if "data" in parts:
        index = parts.index("data")
        return Path(*parts[index:])
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/manifests/synthetic_manifest.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/synthetic_manifest_local.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)

    root = args.root.resolve()
    records = payload.get("records", [])
    missing = []
    for record in records:
        rel = relative_from_data(str(record["video_path"]))
        new_path = root / rel
        record["video_path"] = str(new_path)
        if not new_path.exists():
            missing.append(str(new_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)

    print(f"wrote {args.output}")
    print(f"records={len(records)} missing_videos={len(missing)}")
    if missing:
        for path in missing[:10]:
            print(f"missing: {path}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
