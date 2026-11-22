# SimCSE reproduction workspace

This workspace reproduces unsupervised SimCSE (EMNLP 2021). Primary paper: https://aclanthology.org/2021.emnlp-main.552/ . Official code: https://github.com/princeton-nlp/SimCSE .

## Frozen design

See `results/protocol_20261004.json` and [RESULTS_INDEX.md](../RESULTS_INDEX.md). Training uses validation only for checkpoint selection. No test-set tuning. All configured seeds and the weight-decay ablation must finish before final scoring. Early smoke/development scores are not reproduction scores.

The completed grid logs are retained in `results/logs_20261004/`; run provenance and progress are under `results/training_runs/`. Checkpoint files are excluded from Git and have been removed from the local run directories; recorded hashes remain in run metadata. Scripts preserve previous runs and refuse to overwrite final artifacts.

## Completion

```bash
.venv/bin/python paper_repro/run_training_grid.py
.venv/bin/python paper_repro/evaluate_reproduction.py
.venv/bin/python paper_repro/build_repro_report.py
```

The frozen reproduction and ablation are completed. See [REPRO_REPORT.md](REPRO_REPORT.md). Existing final artifacts are immutable; scripts refuse to overwrite them. A new experiment requires distinct output names, preserving the original protocol and validation-only selection. No supervisor or training job is currently active.

## Reuse requirements

Use `requirements.lock` and the recorded GPU/driver/runtime. Model, dataset, and official source snapshot paths are recorded in audit artifacts; scripts currently target this local environment. Audit hashes must match before reuse. `prepare_official_source.py` verifies downloaded code against pinned Git blob identifiers; `prepare_data.py` audits normalized exact benchmark overlaps; `audit_runtime.py` checks the port's objective/head on a fixture. These checks do not establish full legacy training parity or absence of semantic leakage.

## Limitations

The author-hosted dataset mirror lacks a license card, so its licensing status could not be established. The dataset is not included; do not redistribute it based on this workspace. Modern runtime, Torch AdamW, exact-overlap removal, evaluation batch size, and validation-interval differences prevent a claim of an identical original setup. No publication or submission status is claimed.
