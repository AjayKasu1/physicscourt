#!/usr/bin/env python3
"""Generate the Phase 1 synthetic benchmark clips and manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physicscourt.synthetic import CATEGORIES, SyntheticConfig, generate_dataset

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "synthetic_generator" / "generated")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--montage", type=Path, default=ROOT / "results" / "synthetic_montage.jpg")
    parser.add_argument("--pairs-per-category", type=int, default=10)
    parser.add_argument("--calibration-per-category", type=int, default=2)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--no-photo-control", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Generate a tiny dataset under results/tmp.")
    args = parser.parse_args()

    if args.smoke:
        args.output_root = ROOT / "results" / "tmp" / "synthetic_smoke"
        args.manifest = ROOT / "results" / "tmp" / "synthetic_smoke_manifest.yaml"
        args.montage = ROOT / "results" / "tmp" / "synthetic_smoke_montage.jpg"
        args.pairs_per_category = 1
        args.calibration_per_category = 1
        args.duration_seconds = 1.5
        args.size = 192

    config = SyntheticConfig(
        width=args.size,
        height=args.size,
        fps=args.fps,
        duration_seconds=args.duration_seconds,
        seed=args.seed,
    )
    records = generate_dataset(
        output_root=args.output_root,
        manifest_path=args.manifest,
        config=config,
        pairs_per_category=args.pairs_per_category,
        calibration_per_category=args.calibration_per_category,
        photo_controls=not args.no_photo_control,
        overwrite=args.overwrite,
        montage_path=args.montage,
    )

    table = Table(title="Synthetic Phase 1")
    table.add_column("split")
    table.add_column("possible")
    table.add_column("impossible")
    table.add_column("total")
    for split in ("calibration", "test"):
        split_records = [record for record in records if record.split == split]
        possible = sum(1 for record in split_records if record.possible)
        impossible = sum(1 for record in split_records if not record.possible)
        table.add_row(split, str(possible), str(impossible), str(len(split_records)))
    console.print(table)
    console.print(f"categories: {', '.join(CATEGORIES)}")
    console.print(f"manifest: {args.manifest}")
    console.print(f"montage: {args.montage}")
    console.print(f"videos: {args.output_root}")


if __name__ == "__main__":
    main()

