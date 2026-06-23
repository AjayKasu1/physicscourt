# PhysicsCourt Results

This document summarizes the current PhysicsCourt run. The original target run
was on a 2022 13-inch MacBook Pro M2 with 8 GB unified memory. The key
head-to-head was then reproduced on a Google Cloud `g2-standard-8` VM with one
NVIDIA L4, CUDA, and fp32 throughout, using the exact same copied MP4 clips.
All model inference was local, cache-only after prefetch, and run one model at
a time.

## Experimental Setup

PhysicsCourt uses 252 synthetic 512x512 MP4 clips:

- 240 test clips arranged as 120 possible/impossible twins.
- 12 calibration clips, all normal/possible.
- 6 categories: object permanence, solidity, continuity/teleportation,
  gravity/support, spontaneous vanishing, and momentum/causality.

Possible/impossible twins are matched before the known violation frame `t*`.
The evaluation asks whether each detector assigns a higher anomaly score to the
impossible twin, whether it separates possible/impossible clips by ROC-AUC, and
whether it localizes the violation. Sustained violations are judged with onset
localization as well as argmax localization.

Calibration is fit only on the 12 calibration-normal clips. No test clip
statistics are used for normalization.

## Detectors

**Detector A:** V-JEPA 2 latent prediction error is the primary
latent predictor, using `facebook/vjepa2-vitl-fpc64-256`. DINOv2 latent
extrapolation is kept as a cheap non-video baseline.

**Detector B:** Spatial-State Rule Detector (SSRD). SAM 2, Depth Anything V2,
and CoTracker reconstruct object state. A hand-written physics-rule bank then
scores the reconstructed state. The final SSRD score is category-blind: all
rule families run on every clip, each rule channel is z-scored using
calibration-normal clips only, and the per-frame score is the max across
calibrated rule channels. Category labels are used only for reporting.

SSRD is original to this project. It is not Marble 1.1, not a World Labs model,
and not a method authored by Fei-Fei Li. The public World Labs / Fei-Fei Li
anchor for the spatial-intelligence motivation is Marble 1.1 / Marble 1.1
Plus, released April 2, 2026: a generative world model that produces navigable
3D Gaussian-splat environments, with the Plus variant adding auto-expansion for
larger worlds. Marble is the clearest product expression of the explicit
geometry worldview, while SSRD is the reproducible detector built here for
these MP4 clips.

Fairness disclosure: this is an asymmetric comparison. V-JEPA 2 is a frozen
external model embodying the latent prediction approach. SSRD is a detector
assembled in this project from off-the-shelf perception components plus our
rule bank, calibration, and scoring layer. That gives SSRD more design
degrees of freedom than V-JEPA 2. The guardrail is calibration discipline: both
detectors are fit only on the 12 normal calibration clips, and no test clip
statistics or labels influence either score.

Version note: this document pins the World Labs reference point to Marble 1.1 /
1.1 Plus as reported for April 2, 2026. See the NYU Shanghai RITS summary and
World Labs' Marble/Spark materials linked from the README.

## V-JEPA 2 Fairness Gate

The V-JEPA 2 fairness sweep is complete. Five of six original V-JEPA 2 category
AUCs were at or near chance, which could have indicated a harness/scoring issue
rather than a real limitation. The repo includes two checks for that:

```bash
make vjepa2-fairness
make vjepa2-fairness-live
```

The first command writes `results/vjepa2_fairness_report.json`, checking cached
score variants, frame mapping, temporal liveness, and a small live stride
sweep. The second command writes the resumable heartbeat/report
`results/vjepa2_fairness_live_report.json` and caches per-clip/window/stride
live probes under `results/features/vjepa2_fairness_live_postt/`.

Fairness status:

- Cached all-clip variants do **not** rescue the weak V-JEPA 2 result. L2
  endpoint, center, and approximate target-start mappings are effectively
  unchanged; cosine scoring is worse.
- The live probe confirms the predictor path is live:
  `VJEPA2Model(..., skip_predictor=False).predictor_output` responds to
  temporal order changes. On the probed object-permanence clip, original,
  reversed, shuffled, and static windows produce different prediction errors.
- The live probe currently runs on CPU/fp32 when the process cannot acquire
  MPS. This is runtime/device-availability volatility on the 8 GB profile, not
  an M2 hardware limitation: Phase 0 recorded V-JEPA 2 on MPS/fp16. CPU/fp32 is
  the safer full fairness path, while `make vjepa2-fairness-live-mps`
  remains available only for a foreground environment that reliably reports
  MPS available.
- The broad live stride/window sweep completed on CPU/fp32 with 144/144 work
  items and 0 errors. It covered continuity, gravity, object permanence, and
  spontaneous vanishing; 3 pairs per category; strides 2/4/8; and window
  lengths 64 and 16. It sampled both a window endpoint near `t*` and a later
  endpoint whose window starts near `t*` when possible, directly testing
  whether Phase 2 ties came from max aggregation over shared pre-`t*` content.
- The targeted sweep helped object permanence and produced a gravity/support
  diagnostic signal, but it did not broadly rescue V-JEPA 2. It remains a
  diagnostic gate, not a replacement benchmark: the per-category AUC table
  below comes from the full copied-clip L4/fp32 rerun.

## Cloud Reproduction And Scale Control

The clean reproducibility run used the exact Mac-generated clips copied to a
Google Cloud `g2-standard-8` VM with one NVIDIA L4. The manifest paths were
rewritten for the VM, but the MP4 bytes were not regenerated. All cloud
detector runs used fp32: no `--fp16`.

The V-JEPA 2 0.3B rerun completed 252/252 clips with 0 errors in 33m37s
(`8.00 s/clip`). Its pair signs were 26 impossible-higher, 19 possible-higher,
and 75 tied. This closely reproduces the Mac CPU/fp32 result
(32/15/73), so the high tie count and near-chance V-JEPA 2 result are not an
8 GB Mac/MPS artifact and not an fp16 quantization artifact.

The ViT-g 1B larger-model check, `facebook/vjepa2-vitg-fpc64-256`, also completed
252/252 clips with 0 errors in 1h34m26s (`22.94 s/clip`). It did not rescue the
latent predictor: pair signs were 16 impossible-higher, 22 possible-higher, and
82 tied, with near-chance AUCs in every category. In this setup, scaling V-JEPA
2 from 0.3B to 1B did not improve the benchmark result.

## V-JEPA 2.1 Follow-Up

After the main run, we added a bounded V-JEPA 2.1 follow-up on the same copied
synthetic clips. This is not part of the original Detector A gate, but it
checks whether the newer dense V-JEPA 2.1 recipe changes the conclusion.

The matched-size control was `facebook/vjepa2-vitg-fpc64-384` and the follow-up
model was V-JEPA 2.1 ViT-g/384 from the official Meta `facebookresearch/vjepa2`
code path. Both ran on the same GCP L4 VM in fp32. The V-JEPA 2 ViT-g/384 pass
completed 252/252 clips with 0 errors in 4h11m35s (`59.90 s/clip`). The V-JEPA
2.1 ViT-g/384 pass completed 252/252 clips with 0 errors in 7h25m35s
(`106.09 s/clip`).

This comparison is suggestive, not clean. The two models are matched for size
class and 384px input resolution, but they are not scored through an identical
prediction object. V-JEPA 2 uses the Hugging Face `VJEPA2Model` predictor output.
V-JEPA 2.1 uses the official `torch.hub` code path, where the predictor returns
a tuple and the first output is compared against the hierarchical target encoder
state. The shape probe verified that the V-JEPA 2.1 direct predictor error is
shape-compatible for ViT-g/384, but the prediction object is structurally
different from the Hugging Face V-JEPA 2 output. A cleaner future test would
score both models through one identical encoder-feature readout that avoids both
predictors.

The AUC audit keeps the claim narrow. Under the native `l2_mean` predictor-error
score, V-JEPA 2.1 stays near chance on all six categories. Across the six
exploratory token reductions, V-JEPA 2.1 shows one directional hint on solidity
(`l2_topk` AUC 0.620), which is the category where a denser localized feature
recipe might plausibly help. That hint is not a win: it is one cell among many,
uses a different readout from the native mean score, and is not strong enough to
change the project conclusion. Several max/top-k reductions also become strongly
wrong-signed, including V-JEPA 2.1 AUC 0.000 on continuity and spontaneous
vanishing. Those inverted cells are better treated as confounded readout behavior
than as physics understanding.

Summary AUCs from the follow-up audit:

| Category | V-JEPA 2 ViT-g/384 `l2_mean` | V-JEPA 2 ViT-g/384 best exploratory AUC | V-JEPA 2.1 ViT-g/384 `l2_mean` | V-JEPA 2.1 ViT-g/384 best exploratory AUC |
| --- | ---: | ---: | ---: | ---: |
| continuity_teleportation | 0.509 | 0.704 | 0.480 | 0.480 |
| gravity_support | 0.497 | 0.500 | 0.470 | 0.470 |
| momentum_causality | 0.562 | 0.605 | 0.480 | 0.605 |
| object_permanence | 0.713 | 0.713 | 0.515 | 0.520 |
| solidity | 0.450 | 0.499 | 0.515 | 0.620 |
| spontaneous_vanishing | 0.613 | 0.654 | 0.530 | 0.530 |

The "best exploratory" column is descriptive only. It is not used to pick a
winner because it searches across six reductions per category.

The defensible conclusion is scoped: under the predictor-error latent-surprise
readouts tested here, V-JEPA 2.1 does not rescue PhysicsCourt. This does not show
that V-JEPA 2.1's representation lacks physical information. It shows that this
surprise readout does not reliably turn that representation into possible versus
impossible discrimination on these clips.

## L4/Fp32 Head-To-Head

Clean cloud comparison by per-category ROC-AUC:

| Category | V-JEPA 2 0.3B | V-JEPA 2 ViT-g 1B | DINO baseline | SSRD | AUC winner |
| --- | ---: | ---: | ---: | ---: | --- |
| continuity_teleportation | 0.494 | 0.495 | 0.726 | 0.941 | SSRD |
| gravity_support | 0.500 | 0.495 | 0.231 | 0.093 | V-JEPA 2 0.3B, but chance-level |
| momentum_causality | 0.497 | 0.505 | 0.498 | 0.475 | V-JEPA 2 ViT-g, but chance-level |
| object_permanence | 0.778 | 0.545 | 0.356 | 1.000 | SSRD |
| solidity | 0.466 | 0.491 | 0.566 | 0.950 | SSRD |
| spontaneous_vanishing | 0.576 | 0.534 | 0.315 | 0.903 | SSRD |

In the L4/fp32 report, SSRD wins 4 of 6 categories by ROC-AUC. DINO wins none.
The two nominal V-JEPA 2 wins are not strong positive detections:
gravity_support is exactly chance for V-JEPA 2 0.3B and inverted for SSRD,
while momentum_causality is near chance for all methods. V-JEPA 2 ViT-g 1B
does not become competitive; its strongest AUC is only 0.545 on object
permanence.

## Statistical Audit

`results/l4_fp32_reports/statistical_audit_l4_fp32_report.json` adds
pair-stratified bootstrap confidence intervals, pair-level complementarity
counts, a label-aware oracle upper bound, and exact McNemar tests. The
bootstrap unit is the matched possible/impossible pair.

| Category | V-JEPA 2 AUC 95% CI | SSRD AUC 95% CI | V-JEPA 2-only pairs | SSRD-only pairs | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| continuity_teleportation | 0.494 [0.468, 0.499] | 0.941 [0.846, 1.000] | 0 | 19 | 0.0000 |
| gravity_support | 0.500 [0.500, 0.500] | 0.093 [0.000, 0.202] | 0 | 0 | 1.0000 |
| momentum_causality | 0.497 [0.474, 0.514] | 0.475 [0.385, 0.565] | 1 | 8 | 0.0391 |
| object_permanence | 0.778 [0.675, 0.901] | 1.000 [1.000, 1.000] | 0 | 3 | 0.2500 |
| solidity | 0.466 [0.404, 0.496] | 0.950 [0.850, 1.000] | 0 | 18 | 0.0000 |
| spontaneous_vanishing | 0.576 [0.514, 0.664] | 0.903 [0.763, 1.000] | 0 | 12 | 0.0005 |

Across all 120 test pairs, the complementarity matrix for V-JEPA 2 vs. SSRD is:
25 both correct, 1 V-JEPA-2-only, 60 SSRD-only, and 34 neither correct
(`p = 5.38e-17`, exact McNemar). V-JEPA 2 contributes just one unique correct
pair under the L4/fp32 harness.

The strict paired matrix also exposed a metric trap and a harness warning.
V-JEPA 2 is not cleanly anti-predictive across all pairs. Its L4/fp32 margin
signs are 26 impossible-higher, 19 possible-higher, and 75 tied. Strict paired
accuracy counts ties as misses; tie-half paired accuracy is the more honest
summary for this diagnostic. Among non-ties, V-JEPA 2 0.3B ranks the impossible
clip higher in 26/45 cases. A better reading is: V-JEPA 2 often gives the
possible and impossible twins the same score, but when the current score
responds, it usually responds in the right direction. The 75 ties persisted on
a clean CUDA/fp32 machine, so this is a result that repeats on another machine,
not just a Mac harness worry.

## Motion-Energy Audit

`results/motion_correlation_report.json` tests whether surprise is merely
tracking visual change. It computes per-clip pixel-difference energy and
Farneback optical-flow magnitude, then correlates detector pair margins with
motion-energy pair margins.

The simple motion-only rule is itself strongly anti-physical on these clips:
for post-`t*` pixel change, the impossible clip has more motion in only 0.033
of pairs; for post-`t*` optical flow, only 0.117. Many violations make the
impossible clip calmer than its possible twin.

V-JEPA 2 does **not** behave like a simple "more motion = more surprise"
detector here. Its pair margins are negatively correlated with motion margins:
post-`t*` pixel-change Spearman `r = -0.318` (`p = 3.9e-4`) and post-`t*`
flow Spearman `r = -0.316` (`p = 4.4e-4`). DINO goes the other way: post-`t*`
pixel-change Spearman `r = 0.362` and flow Spearman `r = 0.293`. This makes
DINO the cleaner "motion energy" baseline. So DINO's weakness is not
just that it is a cheap baseline; it is being pulled toward a nuisance axis
that is anti-aligned with physical violation labels in this suite. V-JEPA 2
appears more insensitive/tied than motion-driven under the current harness.

The motion audit weakens the strongest version of the "V-JEPA 2 is inverted
because it tracks motion" hypothesis. It still supports a subtler result:
visual motion energy is badly misaligned with physical violation labels in this
suite, and simple latent baselines can be drawn toward that nuisance axis.

The L4-confirmed scientific result is:

1. Explicit object-state rules are much stronger than latent prediction
   on continuity, object permanence, solidity, and spontaneous vanishing in this
   controlled synthetic setting.
2. V-JEPA 2's earlier clean signal on object permanence is real, but it is not
   unique. SSRD separates object permanence perfectly on these synthetic clips,
   where the violation is defined as mask disappearance and non-return; that
   cell is likely partly circular and should not be oversold as a real-footage
   permanence solution.
3. Momentum/causality remains unsolved by both approaches.
4. Gravity/support exposes an SSRD calibration failure: the raw geometry
   rules can detect it, but category-blind per-rule z-scoring lets unrelated
   high-z rule channels dominate possible clips, inverting the category.

## Detector B Ablation

All ablations reuse the same cached SAM 2, Depth Anything, CoTracker, and rule
channels. They differ only in how rule scores are selected and calibrated.

| Category | Category-conditioned oracle | Category-blind raw max | Category-blind std z-max | Category-blind robust MAD z-max |
| --- | ---: | ---: | ---: | ---: |
| continuity_teleportation | 0.676 | 0.791 | 0.941 | 0.867 |
| gravity_support | 0.904 | 0.905 | 0.093 | 0.048 |
| momentum_causality | 0.515 | 0.482 | 0.475 | 0.491 |
| object_permanence | 1.000 | 0.485 | 1.000 | 0.487 |
| solidity | 0.925 | 0.906 | 0.950 | 0.906 |
| spontaneous_vanishing | 0.925 | 0.278 | 0.903 | 0.276 |

This ablation is the key interpretability result for SSRD. Giving the
rule system the category label is not the whole story: category-blind raw max
already works well for continuity, gravity, and solidity. But scale matters.
Per-rule calibration rescues object permanence and spontaneous vanishing while
breaking gravity/support. There is no single winner in this table: raw-max
keeps gravity, while std z-max keeps the sparse absence categories. We treat
that tradeoff as part of the result rather than tuning around it.

The robust MAD column was a pre-committed single fix attempt, not a tuned
winner-selection sweep. It replaces per-rule mean/std with calibration-normal
median/MAD and applies the same category-blind max to all categories. It does
not rescue gravity: gravity AUC falls from 0.093 to 0.048. It also collapses
object permanence and spontaneous vanishing back near raw-max behavior. The
reason is informative rather than mysterious: the rule channels are sparse
violation alarms, so on normal clips they are mostly silent by construction.
With normal-only calibration, 17 of 18 rules have zero MAD, so the robust scale
falls back to 1.0 for almost every channel and behaves like raw max. The std
z-score version is no safer: it estimates scale from rare normal-clip hiccups,
so an unrelated high-z channel can dominate a possible gravity clip and invert
the category. The gravity failure is not a small-sample nuisance to patch away
with more normal clips. It exposes a structural limitation: normal-only z-score
calibration is the wrong tool for sparse hand-written violation rules. We freeze
this as an SSRD limitation rather than continue test-guided aggregation search.

## Localization

The project uncovered an important metric issue. Argmax localization can be the
wrong statistic for sustained violations. For object permanence, surprise starts
when an object should reappear and then remains elevated; argmax often lands at
the end of the plateau. We report onset localization alongside
argmax. This mirrors the violation-of-expectation logic from infant cognition:
surprise can be sustained rather than instantaneous.

The final strict SSRD score has a second localization caveat: per-rule
calibration can produce excellent clip-level separation while shifting the
highest calibrated rule response away from `t*`. The visual audit should be
consulted before interpreting localization in isolation.

## Visual Audit

The audit sheets render actual MP4 frames, not just metadata. For each category,
the audit includes an SSRD hit and miss, possible/impossible twins, frames
around `t*`, and score overlays for V-JEPA 2, DINO, and SSRD.

- `results/figures/visual_audit/continuity_teleportation_contact_sheet.png`
- `results/figures/visual_audit/continuity_teleportation_score_overlay.png`
- `results/figures/visual_audit/gravity_support_contact_sheet.png`
- `results/figures/visual_audit/gravity_support_score_overlay.png`
- `results/figures/visual_audit/momentum_causality_contact_sheet.png`
- `results/figures/visual_audit/momentum_causality_score_overlay.png`
- `results/figures/visual_audit/object_permanence_contact_sheet.png`
- `results/figures/visual_audit/object_permanence_score_overlay.png`
- `results/figures/visual_audit/solidity_contact_sheet.png`
- `results/figures/visual_audit/solidity_score_overlay.png`
- `results/figures/visual_audit/spontaneous_vanishing_contact_sheet.png`
- `results/figures/visual_audit/spontaneous_vanishing_score_overlay.png`

## Phase 4 Edited-Real Starter Pairs

Phase 4 currently has two edited-real starter pairs. This is not enough to
claim edited-real generalization, but it is a useful check outside the
synthetic generator.

Both pairs are scored with frozen synthetic calibration and no edited-real
refit. The processed scoring clips are 512x288, 12 fps, 72 frames.

Pair setup:

| Pair | Raw input | Processed t* | Notes |
| --- | --- | ---: | --- |
| Object permanence | 1080x608, 30 fps, 180 frames | 32 | ball rolls behind a box; edited twin never reappears |
| Continuity/teleportation | raw crop from 1280x720/1276x718 sources | 16 | ball skips the middle path and reappears left |

Pair results:

| Pair | Detector | Pair result | Possible score | Impossible score | Margin | Argmax frame | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Object permanence | V-JEPA 2 0.3B | wrong | -2.115 | -2.497 | -0.381 | 18 | ranks the possible clip as more anomalous |
| Object permanence | DINOv2-small | wrong | -1.014 | -1.016 | -0.002 | 15 | essentially tied, slight possible-higher ranking |
| Object permanence | SSRD | correct | 37.151 | 55.505 | 18.354 | 59 | top rule is `permanence_absence_run` |
| Continuity/teleportation | V-JEPA 2 0.3B | wrong | -1.901 | -2.118 | -0.217 | 0 | ranks the possible clip as more anomalous |
| Continuity/teleportation | DINOv2-small | wrong | 0.104 | -0.200 | -0.304 | 3 | ranks the possible clip as more anomalous |
| Continuity/teleportation | SSRD | tied | 55.505 | 55.505 | 0.000 | 46 | late `permanence_absence_run` saturates both twins |

The object-permanence edited-real pair matches the synthetic story: SSRD
separates the edited absence, while V-JEPA 2 and DINOv2 do not. The SSRD peak
is late (`argmax=59` vs. `t*=32`), which matches the sustained absence behavior
seen in the synthetic object-permanence clips.

The continuity/teleportation edited-real pair is a negative result. V-JEPA 2
and DINOv2 both score the possible clip higher. SSRD does not reverse the pair,
but it also does not detect the teleport: both twins hit the same late absence
score after the ball leaves the frame, driven by `permanence_absence_run`
rather than a continuity rule. That means this pair should be counted as
missed by all three detectors under strict paired accuracy.

### GCP L4 fp32 Check

The same two edited-real pairs were rerun on a GCP `g2-standard-8` VM with an
NVIDIA L4 GPU using CUDA fp32 outputs. This separates the result from the
MacBook M2 memory constraint and MPS variability.

L4 pair results:

| Pair | Detector | Pair result | Possible score | Impossible score | Margin |
| --- | --- | --- | ---: | ---: | ---: |
| Object permanence | V-JEPA 2 0.3B | wrong | -2.459 | -2.576 | -0.117 |
| Object permanence | DINOv2-small | wrong | -0.947 | -1.001 | -0.054 |
| Object permanence | SSRD | correct | 37.120 | 55.505 | 18.385 |
| Continuity/teleportation | V-JEPA 2 0.3B | wrong | -1.909 | -2.876 | -0.967 |
| Continuity/teleportation | DINOv2-small | wrong | 0.027 | -0.036 | -0.063 |
| Continuity/teleportation | SSRD | tied | 55.505 | 55.505 | 0.000 |

The L4 rerun confirms the same qualitative outcome as the Mac run. The
object-permanence edit is caught only by SSRD. The teleport edit is missed by
all three detectors. The L4 run also confirms the runtime benefit of CUDA:
V-JEPA 2 scored all four processed edited-real clips in about 31 seconds,
compared with several minutes on the Mac path.

Artifacts:

- `results/phase4_edited_real_report.json`
- `results/figures/phase4_edited_real/`
- `results/phase4_edited_real_l4_fp32_report.json`
- `results/figures/phase4_edited_real_l4_fp32/`
- `results/phase4_l4_fp32_*_timing.json`

## Limitations

The largest limitation is that the main benchmark is still synthetic. The
possible/impossible twins are carefully controlled, and Phase 4 currently has
only two edited-real starter pairs. Until the same conclusions survive on a
larger edited-real split, the result should be read as a controlled probe, not
an external validity claim about arbitrary video.

The object-permanence SSRD result is likely partly circular. On these synthetic
clips, the violation is defined by object mask disappearance and non-return,
and SSRD directly measures mask presence/area. The perfect AUC is useful as a
pipeline sanity check, but it is the cell most likely to shrink on real footage
with occlusion, shadows, segmentation noise, and ambiguous reappearance.

Calibration is structurally hard for SSRD. Both detectors obey the normal-only
calibration rule, but SSRD's rule channels are designed to be silent on normal
clips and fire on violations. That means normal-only clips often contain no
meaningful spread from which to estimate per-rule scale. The L4 MAD ablation
confirmed this: 17 of 18 rule channels had zero MAD on calibration normals.
More normal clips would mostly add more zeros unless the rule definitions or
calibration protocol changed. SSRD is especially sensitive to cross-rule score
scale, as shown by the gravity/support inversion after
category-blind per-rule calibration.

The comparison is intentionally asymmetric. V-JEPA 2 is a frozen external model
and SSRD is a project-built pipeline with explicit perception components,
rules, and calibration choices. This makes SSRD interpretable and controllable,
but gives it more design degrees of freedom than the frozen latent predictor.

Detector B also inherits perception ambiguities. SAM 2 masks, CoTracker tracks,
and monocular Depth Anything estimates are local 2D/relative-depth signals, not
metric 3D reconstructions. Depth scale ambiguity and track/mask failures can
make some rule violations look cleaner or noisier than the underlying scene.
The SAM2 stage also emits a Hugging Face compatibility warning because the
checkpoint advertises `sam2_video` while the current implementation uses the
Transformers `Sam2Model` path. The stage completed all clips with 0 errors, but
this remains an implementation caveat; a stricter future version should use the
official SAM2 video predictor API.

V-JEPA 2 is not a child-like learner trained only on these simple physics
events. It is pretrained on internet-scale video, so any success or failure may
reflect training-distribution biases, domain mismatch with flat synthetic
clips, or harness choices rather than a pure verdict on latent prediction world
models.

The V-JEPA 2.1 follow-up adds another harness caveat. The comparison to V-JEPA 2
ViT-g/384 is matched on model scale and input resolution, but not on an identical
prediction object. The V-JEPA 2 run uses the Hugging Face predictor output, while
the V-JEPA 2.1 run uses the official Meta predictor path and a hierarchical
target state. Because both runs remain weak or modest under their native
predictor-error readouts, this mismatch does not create a false positive win for
V-JEPA 2.1. It does mean the result should not be read as a clean recipe-only
comparison or as evidence that the V-JEPA 2.1 representation itself lacks
physical information.

## What Would Change Our Minds

The LeCun-style latent prediction view gains support if a better scoring
scheme, predictor target, or latent model separates the same possible/impossible
twins without category labels, handcrafted rules, or test-statistic leakage.
The completed post-`t*` sweep and the 1B ViT-g larger-model check did not do that
under the current harness. The later V-JEPA 2.1 follow-up also did not rescue the
predictor-error readout. The clean next test would score V-JEPA 2 and V-JEPA 2.1
through one identical encoder-feature readout, then check whether the newer
dense recipe moves the same possible/impossible pairs by AUC.

The explicit object-state view gains support if SSRD keeps separating
violations after the synthetic-only weakness is removed: first on an
edited-real split, then on less controlled real footage, while preserving
normal-only calibration and category-blind scoring. It also gains support if
SSRD's rule-level errors remain interpretable enough to explain failures like
gravity/support without overfitting to the test set. The current gravity result
is a useful counterweight: explicit rules can be brittle because sparse
violation alarms do not calibrate naturally on violation-free normals.

The motion-confound result would change if a better latent model or scoring
scheme separated physical impossibility from gross visual change on these
matched twins. In that case, the benchmark would be showing that the first
V-JEPA 2 harness was too weak, not that latent prediction lacks the needed
signal.

## Reproducible Artifacts

Primary reports:

- `results/environment_report.json`
- `results/detector_a_report.json`
- `results/detector_b_report.json`
- `results/head_to_head_report.json`
- `results/statistical_audit_report.json`
- `results/motion_correlation_report.json`
- `results/detector_b_ablation_report.json`
- `results/visual_audit_report.json`

Cloud reproduction reports checked into this repo:

- `results/l4_fp32_reports/detector_a_l4_fp32_report.json`
- `results/l4_fp32_reports/detector_a_vjepa2_vitg_l4_fp32_report.json`
- `results/l4_fp32_reports/detector_b_l4_fp32_report.json`
- `results/l4_fp32_reports/head_to_head_l4_fp32_report.json`
- `results/l4_fp32_reports/statistical_audit_l4_fp32_report.json`
- `results/l4_fp32_reports/detector_b_ablation_l4_fp32_report.json`
- `results/l4_fp32_reports/detector_a_vjepa2_vitg384_l4_fp32_report.json`
- `results/l4_fp32_reports/detector_a_vjepa21_l4_fp32_report.json`
- `results/l4_fp32_reports/vjepa2_vitg384_reductions_l4_fp32_report.json`
- `results/l4_fp32_reports/vjepa21_reductions_l4_fp32_report.json`
- `results/l4_fp32_reports/vjepa21_shape_probe_l4_fp32.json`

Regenerate the final analysis from cached features:

```bash
make final-results
make test
```

## Bottom Line

On these copied synthetic clips, in a clean L4/fp32 environment, SSRD
beats V-JEPA 2 0.3B on pair-level correctness and on four
of six AUC categories. V-JEPA 2 0.3B still shows a real object-permanence
signal, but it remains tied or near chance elsewhere; V-JEPA 2 ViT-g 1B does
not rescue the result, and the V-JEPA 2.1 follow-up does not rescue the
predictor-error readout either. The claim is still bounded: this is a
synthetic-only, project-built explicit-state detector versus frozen external
latent predictors, with known SSRD failures on gravity/support and
momentum/causality, plus a structurally mismatched V-JEPA 2.1 readout caveat.
Within that scope, the clean result is no longer just a laptop artifact:
explicit spatial-state rules win this controlled PhysicsCourt run, while the
tested V-JEPA latent prediction scores mostly tie, miss, or invert the twins.
