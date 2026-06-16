PYTHON ?= /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
export PYTHONPATH := $(CURDIR)/src

.PHONY: phase0 smoke smoke-vjepa-offline smoke-cotracker-offline download-weights download-cotracker phase1-smoke generate-synthetic phase2-vjepa-first5 phase2-vjepa phase2-dino phase2-cache-detector-a phase2-evaluate vjepa2-fairness vjepa2-fairness-cached vjepa2-fairness-live vjepa2-fairness-live-mps start-vjepa2-fairness-live phase3-b-first5 phase3-b-evaluate compare-detectors detector-b-ablation statistical-audit motion-correlation visual-audit final-results clean-cache test

phase0: smoke

smoke:
	HF_HUB_OFFLINE=1 $(PYTHON) scripts/smoke_models.py --config config/models.yaml --report results/environment_report.json --device auto --fp16 --offline

smoke-vjepa-offline:
	HF_HUB_OFFLINE=1 $(PYTHON) scripts/smoke_models.py --config config/models.yaml --report results/vjepa_offline_report.json --device auto --fp16 --only vjepa2 --offline

smoke-cotracker-offline:
	HF_HUB_OFFLINE=1 $(PYTHON) scripts/smoke_models.py --config config/models.yaml --report results/cotracker_report.json --device auto --fp16 --only cotracker --offline

download-weights:
	$(PYTHON) scripts/download_weights.py --config config/models.yaml --report results/weights_report.json

download-cotracker:
	$(PYTHON) scripts/download_weights.py --config config/models.yaml --report results/cotracker_weights_report.json --only cotracker

phase1-smoke:
	$(PYTHON) scripts/generate_synthetic.py --smoke --overwrite

generate-synthetic:
	$(PYTHON) scripts/generate_synthetic.py --overwrite

phase2-vjepa-first5:
	HF_HUB_OFFLINE=1 $(PYTHON) scripts/run_detector_a.py --detector vjepa2 --limit 5 --timing-report results/vjepa_first5_timing.json --device auto --fp16

phase2-vjepa:
	HF_HUB_OFFLINE=1 $(PYTHON) scripts/run_detector_a.py --detector vjepa2 --timing-report results/vjepa_timing.json --device auto --fp16 --continue-on-error

phase2-dino:
	HF_HUB_OFFLINE=1 $(PYTHON) scripts/run_detector_a.py --detector dino_latent --timing-report results/dino_timing.json --device auto --fp16 --continue-on-error

phase2-cache-detector-a:
	$(PYTHON) scripts/cache_detector_a.py

phase2-evaluate:
	$(PYTHON) scripts/evaluate_detector_a.py

vjepa2-fairness:
	HF_HUB_OFFLINE=1 $(PYTHON) scripts/vjepa2_fairness_sweep.py --device auto --fp16 --categories object_permanence --pairs-per-category 1 --strides 4 --max-live-windows-per-clip 1

vjepa2-fairness-cached:
	HF_HUB_OFFLINE=1 $(PYTHON) scripts/vjepa2_fairness_sweep.py --skip-live --report results/vjepa2_fairness_cached_report.json

vjepa2-fairness-live:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) scripts/run_vjepa2_fairness_job.py --device auto --fp16

vjepa2-fairness-live-mps:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) scripts/run_vjepa2_fairness_job.py --device mps --fp16 --cache-dir results/features/vjepa2_fairness_live_mps --report results/vjepa2_fairness_live_mps_report.json

start-vjepa2-fairness-live:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) scripts/start_vjepa2_fairness_live_job.py

phase3-b-first5:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) scripts/run_detector_b.py --stage all --limit 5 --device auto --fp16 --continue-on-error

phase3-b-evaluate:
	$(PYTHON) scripts/evaluate_detector_b.py --report results/detector_b_report.json --calibration-report results/detector_b_calibration_report.json --figures-dir results/figures/detector_b_eval

compare-detectors:
	$(PYTHON) scripts/compare_detectors.py

detector-b-ablation:
	$(PYTHON) scripts/ablate_detector_b.py

statistical-audit:
	$(PYTHON) scripts/statistical_audit.py

motion-correlation:
	$(PYTHON) scripts/motion_correlation_audit.py

visual-audit:
	$(PYTHON) scripts/visual_audit.py

final-results: phase2-evaluate phase3-b-evaluate compare-detectors detector-b-ablation statistical-audit motion-correlation visual-audit

clean-cache:
	$(PYTHON) scripts/clean_cache.py --limit-gb 25

test:
	$(PYTHON) -m pytest tests
