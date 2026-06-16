# GCP L4 Reproducibility Runbook

This runbook is the clean cloud path for PhysicsCourt after the 8 GB M2 run. It
uses one `g2-standard-8` VM with one NVIDIA L4 GPU and runs fp32 throughout.

Official references:

- G2 machine series: <https://docs.cloud.google.com/compute/docs/gpus>
- PyTorch Deep Learning VM: <https://docs.cloud.google.com/deep-learning-vm/docs/pytorch_start_instance>
- Free Trial GPU restrictions: <https://docs.cloud.google.com/free/docs/free-cloud-features>

Important constraints:

- Do **not** regenerate the synthetic clips on GCP. Copy the Mac-generated MP4s
  and manifest, then rewrite only the absolute paths in the copied manifest.
- Do **not** pass `--fp16`. The cloud run is fp32 so V-JEPA 2 tie counts cannot
  be blamed on quantization, and so CoTracker avoids mixed Half/Float issues.
- Keep outputs in `_l4_fp32` paths until the run is accepted.
- Run the V-JEPA 2 full-pass reproduction check before the fairness gate or the
  full SSRD rerun. The key question is whether L4/fp32 reproduces the
  near-chance AUCs and large exact-tie count from the Mac CPU/fp32 reference.

## 1. Create The VM

If your $300 credit is attached to a non-billable Free Trial account, upgrade
the billing account first. Google blocks GPUs and quota increases on non-billable
trial accounts; remaining credit can continue after upgrade inside the original
trial window.

```bash
PROJECT_ID="your-project-id"
ZONE="us-central1-a"
INSTANCE="physicscourt-l4"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"

gcloud config set project "$PROJECT_ID"
gcloud services enable compute.googleapis.com

gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type="g2-standard-8" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="deeplearning-platform-release" \
  --boot-disk-size="200GB" \
  --boot-disk-type="pd-ssd" \
  --maintenance-policy="TERMINATE" \
  --metadata="install-nvidia-driver=True"
```

If a zone is stocked out, keep the same command and try another G2-supported
zone. The completed run used `us-central1-a`.

Verify CUDA after the driver install/reboot completes:

```bash
gcloud compute ssh "$INSTANCE" --zone="$ZONE"

nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 2. Copy This Exact Checkout And Dataset

Copy the working tree and the already-generated dataset from the Mac. This
preserves the exact MP4 inputs used by the Mac reference run.

```bash
cd /path/to/EmbodiedAI

tar \
  --exclude='physicscourt/.venv' \
  --exclude='physicscourt/.pytest_cache' \
  --exclude='physicscourt/results/features' \
  --exclude='physicscourt/results/figures' \
  --exclude='physicscourt/results/logs' \
  --exclude='physicscourt/results/.cache' \
  --exclude='physicscourt/results/.matplotlib' \
  -czf /tmp/physicscourt_l4_input.tgz \
  physicscourt

gcloud compute scp /tmp/physicscourt_l4_input.tgz "$INSTANCE":~ --zone="$ZONE"
```

If the Mac does not have `gcloud` installed, create the same archive on the Mac,
then use the SSH-in-browser **Upload file** button and upload
`physicscourt_l4_input.tgz` to the VM home directory.

On the VM:

```bash
tar xzf ~/physicscourt_l4_input.tgz
cd ~/physicscourt

sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements.txt

python scripts/rewrite_manifest_paths.py \
  --input data/manifests/synthetic_manifest.yaml \
  --output data/manifests/synthetic_manifest_l4.yaml \
  --root "$PWD"
```

Do not run `make generate-synthetic` on GCP.

If `tar` prints `LIBARCHIVE.xattr.com.apple.provenance` warnings, ignore them.
Those are Mac metadata headers and do not affect the files.

## 3. Download Weights And Smoke Test

Prefetch weights without a time cap, then run offline bounded smoke tests.

```bash
python scripts/download_weights.py \
  --config config/models.yaml \
  --report results/weights_report_l4_fp32.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/smoke_models.py \
  --config config/models.yaml \
  --report results/environment_report_l4_fp32.json \
  --device cuda \
  --offline
```

The smoke report should say `cuda` and `float32`.

## 4. V-JEPA 2 Reproduction Check First

Run the full V-JEPA 2 cache pass in fp32 before running SSRD or the fairness
gate. This is the scientific fork:

- If L4/fp32 reproduces the near-chance AUCs and high exact-tie count, the weak
  V-JEPA 2 result is no longer plausibly a laptop/MPS artifact.
- If L4/fp32 diverges and V-JEPA 2 starts separating pairs, the Mac result was
  a harness/device artifact and the interpretation must change.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_detector_a.py \
  --detector vjepa2 \
  --manifest data/manifests/synthetic_manifest_l4.yaml \
  --features-dir results/features_l4_fp32 \
  --timing-report results/vjepa2_l4_fp32_timing.json \
  --device cuda \
  --continue-on-error

python scripts/evaluate_detector_a.py \
  --detectors vjepa2 \
  --manifest data/manifests/synthetic_manifest_l4.yaml \
  --features-dir results/features_l4_fp32 \
  --report results/detector_a_vjepa2_l4_fp32_report.json \
  --calibration-report results/calibration_vjepa2_l4_fp32_report.json \
  --figures-dir results/figures/detector_a_vjepa2_l4_fp32
```

Print the V-JEPA 2 pair-tie counts:

```bash
python - <<'PY'
import json
from collections import defaultdict

report = json.load(open("results/detector_a_vjepa2_l4_fp32_report.json"))
rows = [
    row for row in report["detectors"]["vjepa2"]["rows"]
    if row["split"] == "test"
]
pairs = defaultdict(dict)
for row in rows:
    pairs[row["pair_id"]][bool(row["possible"])] = float(row["clip_level_score"])

counts = {"impossible_higher": 0, "possible_higher": 0, "tied": 0}
for pair in pairs.values():
    if True not in pair or False not in pair:
        continue
    margin = pair[False] - pair[True]
    if margin > 0:
        counts["impossible_higher"] += 1
    elif margin < 0:
        counts["possible_higher"] += 1
    else:
        counts["tied"] += 1

metrics = report["detectors"]["vjepa2"]["metrics"]
print("pair_sign_counts", counts)
print("auc_by_category", {k: round(v["roc_auc"], 3) for k, v in metrics.items()})
PY
```

Compare against the Mac reference: 32 impossible-higher, 15 possible-higher, 73
ties over 120 test pairs, with V-JEPA 2 AUCs near chance except
object_permanence.

## 5. Continue Full L4/Fp32 Pipeline

Only after reviewing the V-JEPA 2 reproduction check, continue:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_detector_a.py \
  --detector dino_latent \
  --manifest data/manifests/synthetic_manifest_l4.yaml \
  --features-dir results/features_l4_fp32 \
  --timing-report results/dino_l4_fp32_timing.json \
  --device cuda \
  --continue-on-error

python scripts/evaluate_detector_a.py \
  --manifest data/manifests/synthetic_manifest_l4.yaml \
  --features-dir results/features_l4_fp32 \
  --report results/detector_a_l4_fp32_report.json \
  --calibration-report results/calibration_l4_fp32_report.json \
  --figures-dir results/figures/detector_a_l4_fp32

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_detector_b.py \
  --stage all \
  --manifest data/manifests/synthetic_manifest_l4.yaml \
  --features-dir results/features_l4_fp32 \
  --timing-report results/ssrd_l4_fp32_timing.json \
  --device cuda \
  --continue-on-error \
  --clip-timeout-seconds 300

python scripts/evaluate_detector_b.py \
  --manifest data/manifests/synthetic_manifest_l4.yaml \
  --features-dir results/features_l4_fp32 \
  --report results/detector_b_l4_fp32_report.json \
  --calibration-report results/detector_b_calibration_l4_fp32_report.json \
  --figures-dir results/figures/detector_b_l4_fp32

python scripts/compare_detectors.py \
  --detector-a-report results/detector_a_l4_fp32_report.json \
  --detector-b-report results/detector_b_l4_fp32_report.json \
  --output results/head_to_head_l4_fp32_report.json

python scripts/statistical_audit.py \
  --detector-a-report results/detector_a_l4_fp32_report.json \
  --detector-b-report results/detector_b_l4_fp32_report.json \
  --output results/statistical_audit_l4_fp32_report.json
```

## 6. Fairness Gate On L4/Fp32

Run this after the V-JEPA 2 full-pass reproduction check, not before:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_vjepa2_fairness_job.py \
  --manifest data/manifests/synthetic_manifest_l4.yaml \
  --device cuda \
  --cache-dir results/features_l4_fp32/vjepa2_fairness_live \
  --report results/vjepa2_fairness_live_l4_fp32_report.json
```

## 7. Copy Results Back And Stop The VM

From the Mac:

```bash
gcloud compute scp --recurse \
  "$INSTANCE":~/physicscourt/results \
  /path/to/EmbodiedAI/physicscourt/results_l4_fp32 \
  --zone="$ZONE"
```

Stop the VM as soon as the results are copied:

```bash
gcloud compute instances stop "$INSTANCE" --zone="$ZONE"
```

Delete it instead if no snapshot or disk state is needed:

```bash
gcloud compute instances delete "$INSTANCE" --zone="$ZONE"
```
