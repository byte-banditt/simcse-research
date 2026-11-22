# SimCSE Reproduction and Weight Decay Ablation

Reproduction study of unsupervised BERT-base SimCSE and a controlled AdamW weight decay ablation. The original paper reports **76.25%** average STS Spearman; our three-seed reference mean is **75.34%**. Full methods, per-seed scores, and provenance: [reproduction report](paper_repro/REPRO_REPORT.md) and [results index](RESULTS_INDEX.md).

## Results

Scores are average Spearman correlation (%) over STS12, STS13, STS14, STS15, STS16, STS-Benchmark, and SICK-Relatedness.

| Run | Weight decay | Mean | 95% CI across seeds |
| --- | ---: | ---: | ---: |
| Paper, Table 5 | 0.0 | 76.25 | Not reported |
| Reference reproduction | 0.0 | 75.34 | [73.56, 77.12] |
| Weight decay ablation | 0.01 | 76.17 | [75.54, 76.80] |

The ablation minus reference difference is **+0.83 points** (paired-seed 95% CI: **[-0.39, 2.05]**). This interval includes zero; these results do not establish an improvement. The paper reports a single-model score, while our intervals describe variation across only three training seeds.

## Reproduction

Training and evaluation scripts use the frozen protocol at [`paper_repro/results/protocol_20261004.json`](paper_repro/results/protocol_20261004.json). They expect the audited model, datasets, GPU environment, and paths recorded by the local run. Those inputs are not included in this repository, and the scripts are not a turnkey run on a clean machine. Check paths, hashes, dependencies, and hardware before attempting reproduction.

On the original environment, the recorded commands were:

```bash
.venv/bin/python paper_repro/run_training_grid.py
.venv/bin/python paper_repro/evaluate_reproduction.py
.venv/bin/python paper_repro/build_repro_report.py
```

The training grid must finish all frozen seeds before test evaluation. Checkpoint selection uses STS-B development data only. Scripts preserve existing results and refuse to overwrite final artifacts; use a separate checkout for new experiments. The lockfile is `paper_repro/requirements.lock`.

## Scope and limitations

- Three seeds provide limited precision; the ablation interval spans zero.
- The modern runtime is not bit-identical to the original implementation. Objective/head parity was checked on a fixture, not full training-trajectory parity.
- One exact normalized benchmark overlap was removed from the training corpus. Semantic and near-duplicate leakage audits were unavailable.
- The training-data mirror has no license card. Its license remains **unverified**; data are not included and should not be redistributed based on this study.
- Single-GPU results; no multi-GPU or throughput claim. Dataset provenance and licensing limitations are described in the reproduction report.

The pinned upstream implementation is in [`vendor/official/`](vendor/official/). See the [paper](https://aclanthology.org/2021.emnlp-main.552/) and its [appendix](https://arxiv.org/html/2104.08821v2).
