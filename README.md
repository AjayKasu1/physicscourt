# PhysicsCourt

PhysicsCourt is a reproducible, inference-only benchmark for testing two ways of detecting physically impossible events in video:

- **V-JEPA 2 latent prediction detector:** surprise in pretrained video representation space.
- **SSRD:** explicit object state from segmentation, depth, and point tracks, checked against physical constraints.

## Scientific Framing

This benchmark operationalizes a current disagreement about world models. One side, associated here with LeCun-style JEPA systems, argues that physical common sense should emerge from self-supervised prediction in abstract latent space, where irrelevant pixel detail can be discarded. The other side, associated here with Fei-Fei Li's World Labs and its Marble line of spatial world models, argues that spatial intelligence needs explicit 3D/4D state that respects geometry, objects, and dynamics. PhysicsCourt tests the disagreement with a violation-of-expectation setup: possible and impossible videos are matched until a known violation frame, and detectors are judged by whether they are more surprised by the impossible twin.

The two hypotheses are deliberately steelmanned:

- **H_LeCun:** latent prediction error from a pretrained video world model is a robust, rule-free detector of physical violations.
- **H_Li:** explicit reconstructed state plus symbolic physics checks is more interpretable and less likely to confuse statistical novelty with physical impossibility.

This repository is being built phase by phase. Phase 0 records the hardware and model viability before any benchmark code depends on a model.

The completed run and final interpretation are summarized in [RESULTS.md](RESULTS.md).

### World Labs / Marble Anchor

The explicit-geometry side is motivated by Fei-Fei Li's World Labs work, not implemented by it. PhysicsCourt does not run a private World Labs model. As of this run, the relevant public World Labs reference point is **Marble 1.1 / Marble 1.1 Plus**, released April 2, 2026. Marble generates navigable 3D environments as Gaussian splats, with Marble 1.1 improving visual quality and Marble 1.1 Plus adding auto-expansion for larger environments. That is the explicit-spatial-geometry commitment in its literal product form.

We call Detector B **Spatial-State Rule Detector (SSRD)**. It is a detector built in this project from SAM 2, Depth Anything V2, CoTracker, and a hand-written calibrated physics-rule layer. SSRD is not Marble, is not a World Labs model, and should not be attributed to Fei-Fei Li. The comparison is **V-JEPA 2 vs. SSRD: latent prediction vs. explicit object-state reasoning**, not a product benchmark between Meta and World Labs.

Fairness disclosure: the comparison is asymmetric by design. V-JEPA 2 is a
frozen external model and SSRD is a method built in this project. SSRD has more
degrees of freedom because it includes an explicit rule bank and calibration
layer, while V-JEPA 2 contributes a single latent prediction surprise score.
Both detectors are calibrated only on the 12 normal calibration clips; no test
clip statistics or labels are used to fit either detector.

References for this framing:

- NYU Shanghai RITS summary of Marble 1.1 / 1.1 Plus: <https://rits.shanghai.nyu.edu/ai/world-labs-releases-marble-1-1-auto-expanding-3d-world-generation/>
- World Labs Marble overview: <https://www.worldlabs.ai/blog/marble-world-model>
- World Labs Spark 2.0 renderer note: <https://www.worldlabs.ai/blog/spark-2.0>

## License And Acknowledgments

PhysicsCourt's source code and project-written documentation are released under
the [MIT License](LICENSE). The repo uses public research models including
V-JEPA 2, DINOv2, SAM 2, Depth Anything V2, and CoTracker. Their checkpoints
are downloaded into the user's local cache and are not redistributed here. See
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for the third-party model note.

## Reproduce

Fresh-clone command sequence:

```bash
cd /path/to/EmbodiedAI
git clone https://github.com/AjayKasu1/physicscourt.git
cd physicscourt
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pip install -r requirements.txt
make download-weights
make phase0
make generate-synthetic
make phase2-vjepa-first5
```

After the first-five V-JEPA 2 timing gate is accepted:

```bash
make phase2-vjepa
make phase2-dino
make phase2-evaluate
make vjepa2-fairness
make vjepa2-fairness-live
# If a foreground GUI Terminal reports MPS available, this faster variant writes
# to separate MPS cache/report paths and should not be run concurrently with the
# CPU fallback:
make vjepa2-fairness-live-mps
make phase3-b-first5
make phase3-b-evaluate
make compare-detectors
make detector-b-ablation
make statistical-audit
make motion-correlation
make visual-audit
```

For a reproducible cloud rerun on Google Cloud L4, use the fp32-only runbook in
[docs/gcp_l4_reproduce.md](docs/gcp_l4_reproduce.md). That path copies the
existing Mac-generated MP4s instead of regenerating clips, rewrites only
manifest paths for the VM checkout, and checks whether L4/fp32 reproduces the
V-JEPA 2 tie/near-chance result before running the fairness gate.

## Hardware Profile

The first target machine is a 2022 13-inch MacBook Pro with an M2 chip, 8 GB unified memory, and roughly 84 GB free disk at project start. That is below the original 16-32 GB assumption, so the default path is intentionally conservative:

- short side capped at 384 px for model inputs
- one model loaded at a time, with `del`, `gc.collect()`, and `torch.mps.empty_cache()` between stages
- MPS first, CPU fallback if a model or op fails
- fp16 on MPS only when the smoke test proves it is safe
- compressed feature caches and generated artifacts capped by policy at about 25 GB

For V-JEPA 2, the default is the smallest published V-JEPA 2 checkpoint on Hugging Face: `facebook/vjepa2-vitl-fpc64-256` (ViT-L, 0.3B params, 1.3 GB safetensors). After separating unbounded prefetch from bounded offline smoke testing, this checkpoint runs on the 8 GB machine. Detector A uses V-JEPA 2 as the primary model, with DINOv2 latent extrapolation kept as the cheap latent baseline and fallback.

Phase 0 demonstrates that V-JEPA 2 can run on this M2 through MPS/fp16. If a
later long-running fairness job falls back to CPU, treat that as runtime
device-availability volatility on the 8 GB profile rather than evidence that
the hardware cannot run V-JEPA 2 on MPS. CPU/fp32 is the safer full fairness path;
the MPS target is an optional foreground acceleration path only when MPS is
reliably available.

## Phase 0 Model Choices

| Role | Default checkpoint | Loader path |
| --- | --- | --- |
| Latent image baseline | `facebook/dinov2-small` | `AutoImageProcessor` + `AutoModel` |
| Latent video attempt | `facebook/vjepa2-vitl-fpc64-256` | `AutoVideoProcessor` + `VJEPA2Model` |
| Monocular depth | `depth-anything/Depth-Anything-V2-Small-hf` | `AutoImageProcessor` + `AutoModelForDepthEstimation` |
| Segmentation | `facebook/sam2.1-hiera-tiny` | `Sam2Processor` + `Sam2Model` |
| Point tracking | `facebook/cotracker3` / `scaled_offline.pth` | `CoTrackerPredictor` |

CoTracker is deliberately not stubbed. If a compatible local CoTracker loader is unavailable, Phase 0 stops and records that as a blocking environment issue for Detector B rather than silently replacing point tracks with fake data.

## Commands

Use the Python 3.11 runtime available on this machine. Run `make download-weights` before `make phase0` on a fresh cache; smoke tests are cache-only and bounded.

```bash
cd /path/to/EmbodiedAI/physicscourt
make phase0
```

Useful individual commands:

```bash
make smoke
make download-weights
make clean-cache
```

The Phase 0 report is written to:

```text
results/environment_report.json
```

## Disk Policy

Repo-owned generated artifacts live under `results/`, `.cache/`, and generated data folders. `make clean-cache` removes repo-owned caches and rendered artifacts. It does not delete the global Hugging Face cache because those files may be shared across projects.

## Current Scientific Status

The benchmark has completed Detector A, Detector B, ablations, visual audit, statistical uncertainty, motion-energy audit, the V-JEPA 2 fairness sweep, and a clean Google Cloud L4/fp32 reproduction run. The current result is in [RESULTS.md](RESULTS.md). The L4 run used the same copied MP4 clips, reproduced the V-JEPA 2 0.3B tie/near-chance result, and showed that a ViT-g 1B V-JEPA 2 larger-model check does not rescue the latent predictor under the current scoring harness. SSRD beats the tested V-JEPA 2 scores on this synthetic suite, with the caveats documented in the results.

## Phase 0 Result On This Machine

`results/environment_report.json`, `results/vjepa_offline_report.json`, and `results/cotracker_report.json` record the final hardware gate on the 8 GB M2 profile:

| Model stage | Result | Device / dtype | Smoke time |
| --- | --- | --- | --- |
| DINOv2-small latent embedding | pass | MPS / fp16 | 8.51 s |
| Depth Anything V2 Small | pass | MPS / fp16 | 4.09 s |
| SAM2 tiny | pass | MPS / fp16 | 5.11 s |
| V-JEPA 2 `facebook/vjepa2-vitl-fpc64-256` | pass | MPS / fp16 | 10.80 s |
| CoTracker | pass | MPS / fp32 | 1.33 s |

The final Detector A path is V-JEPA-2-primary. DINOv2-small remains the cheap latent extrapolation baseline and fallback. The earlier V-JEPA 2 timeout was a download/cache artifact, not an inference result. Going forward, weight prefetch is unbounded and smoke tests run with `HF_HUB_OFFLINE=1`.

### CoTracker Follow-Up

CoTracker is installed from the official repository at pinned commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`, with weights fetched from `facebook/cotracker3` / `scaled_offline.pth`. The bounded offline smoke test is written to `results/cotracker_report.json`. On this machine, CoTracker fp16 on MPS fails with a mixed Half/Float grid error, then fp32 on MPS passes in 1.33 s on the tiny smoke clip. Detector B uses CoTracker on MPS/fp32 by default.

## Phase 1 Synthetic Dataset

Phase 1 generated the synthetic controlled benchmark with OpenCV/NumPy rendering only:

- Manifest: `data/manifests/synthetic_manifest.yaml`
- Videos: `data/synthetic_generator/generated/`
- Montage: `results/synthetic_montage.jpg`
- Test split: 120 possible clips and 120 impossible clips
- Calibration split: 12 normal-only clips
- Categories: object permanence, solidity, continuity/teleportation, gravity/support, spontaneous vanishing, momentum/causality
- Photo-control labels: procedural real-image-statistics controls are labeled `procedural_photo_like`, not real photos

The generated MP4 set is about 18 MB. Each clip is 512x512, 24 fps, 72 frames. Matched possible/impossible pairs are pixel-identical before their recorded violation frame.

Generation is bit-reproducible from the manifest seed policy: the dataset root seed is `1729`, category/pair seeds are deterministic offsets from that root, and calibration uses a fixed `+500000` offset. Re-running `make generate-synthetic` with the same code and OpenCV encoder rewrites the same scenarios, prompts, violation frames, and manifest records.

## Phase 4 Edited-Real Starter Pairs

Phase 4 has started with two user-provided edited-real pairs:

- Raw manifest: `data/manifests/edited_real_manifest.yaml`
- Scoring manifest: `data/manifests/edited_real_processed_manifest.yaml`
- Object permanence: a ball rolls behind a box and either reappears or never reappears.
- Continuity/teleportation: a ball rolls smoothly left or skips the middle path and reappears left.
- Processed scoring clips are 512x288, 72 frames, 12 fps.

These pairs are not enough to make a Phase 4 claim. They are starter validation
examples and a scaffold for adding more edited-real pairs across the six
PhysicsCourt categories.

Initial frozen-detector result:

- Object permanence: SSRD is correct; V-JEPA 2 0.3B and DINOv2-small are wrong.
- Continuity/teleportation: no detector cleanly catches the edit. V-JEPA 2 and DINOv2 rank the possible clip as more anomalous; SSRD ties the pair because a late absence rule saturates both clips.

See `results/phase4_edited_real_report.json` and the overlays under
`results/figures/phase4_edited_real/`.

A CUDA fp32 rerun on a GCP L4 VM is also included:

- Report: `results/phase4_edited_real_l4_fp32_report.json`
- Overlays: `results/figures/phase4_edited_real_l4_fp32/`
- Timings: `results/phase4_l4_fp32_*_timing.json`

The L4 rerun confirms the same qualitative outcome: SSRD catches the
object-permanence edit, while all three detectors miss the teleport edit.
