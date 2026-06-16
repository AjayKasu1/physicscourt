from __future__ import annotations

from pathlib import Path

from physicscourt.pipeline.clip_dataset import load_manifest
from physicscourt.pipeline.video_io import probe_video, read_frame_rgb
from physicscourt.synthetic import CATEGORIES, SyntheticConfig, SyntheticGenerator, generate_dataset


def test_pairs_are_identical_before_violation() -> None:
    generator = SyntheticGenerator(SyntheticConfig(width=192, height=192, duration_seconds=1.5))
    for idx, category in enumerate(CATEGORIES):
        spec = generator.make_spec(category, f"pair_{idx}", "test", 10_000 + idx, "plain")
        generator.assert_pair_identity_until_violation(spec)


def test_smoke_dataset_manifest_and_video_readback(tmp_path: Path) -> None:
    output_root = tmp_path / "clips"
    manifest_path = tmp_path / "manifest.yaml"
    records = generate_dataset(
        output_root=output_root,
        manifest_path=manifest_path,
        config=SyntheticConfig(width=192, height=192, duration_seconds=1.0, fps=12, seed=19),
        pairs_per_category=1,
        calibration_per_category=1,
        photo_controls=False,
        overwrite=True,
    )

    loaded = load_manifest(manifest_path)
    assert len(loaded) == len(records)
    assert len([record for record in loaded if record.split == "test" and record.possible]) == len(CATEGORIES)
    assert len([record for record in loaded if record.split == "test" and not record.possible]) == len(CATEGORIES)
    assert len([record for record in loaded if record.split == "calibration"]) == len(CATEGORIES)
    assert all(record.possible for record in loaded if record.split == "calibration")

    first = loaded[0]
    probe = probe_video(Path(first.video_path))
    assert probe.width == 192
    assert probe.height == 192
    assert probe.frame_count == 12
    frame = read_frame_rgb(Path(first.video_path), 0)
    assert frame.shape == (192, 192, 3)

