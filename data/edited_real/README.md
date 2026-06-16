# Edited-Real Clips

This folder holds Phase 4 clips: real or edited-real possible/impossible video
pairs used to check whether the synthetic PhysicsCourt result survives outside
the procedural generator.

The first pair is an object-permanence example:

- `raw/object_permanence/pair000/possible.mov`: a pink ball rolls behind a box
  and reappears.
- `raw/object_permanence/pair000/impossible.mov`: the edited twin where the
  ball never reappears.

The manifest is `data/manifests/edited_real_manifest.yaml`. The estimated
violation onset for pair 000 is frame `80`, the first frame where the possible
clip shows the ball reappearing while the impossible clip remains empty.
