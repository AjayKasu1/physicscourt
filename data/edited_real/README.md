# Edited-Real Clips

This folder holds Phase 4 clips: real or edited-real possible/impossible video
pairs used to check whether the synthetic PhysicsCourt result survives outside
the procedural generator.

The first pair is an object-permanence example:

- `raw/object_permanence/pair000/possible.mov`: a pink ball rolls behind a box
  and reappears.
- `raw/object_permanence/pair000/impossible.mov`: the edited twin where the
  ball never reappears.

The second pair is a continuity/teleportation example:

- `raw/continuity_teleportation/pair000/possible.mp4`: a pink ball rolls
  smoothly left across the surface.
- `raw/continuity_teleportation/pair000/impossible.mp4`: the edited twin where
  the ball skips the middle path and reappears left.

The raw manifest is `data/manifests/edited_real_manifest.yaml`. The estimated
raw violation onset for the object-permanence pair is frame `80`, the first
frame where the possible clip shows the ball reappearing while the impossible
clip remains empty. The estimated raw violation onset for the
continuity/teleportation pair is frame `65`, the first frame where the edited
twin breaks the smooth trajectory.

Detector scoring uses the processed manifest
`data/manifests/edited_real_processed_manifest.yaml`. The processed clips are
512x288, 72 frames, 12 fps, and preserve the same event. In the processed clips,
the object-permanence violation onset is frame `32`, and the
continuity/teleportation violation onset is frame `16`.
