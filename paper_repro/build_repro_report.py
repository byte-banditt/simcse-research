#!/usr/bin/env python3
"""Build a candid report only from completed, frozen reproduction artifacts."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project_paths import resolve_legacy_path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    test_path = ROOT / 'results/reproduction_test_20261004.json'
    grid_path = ROOT / 'results/training_grid_20261004.json'
    test, grid = [json.loads(p.read_text()) for p in (test_path, grid_path)]
    assert test['grid_sha256'] == sha(grid_path)
    assert test['test_set_tuning'] is False
    report = ROOT / 'REPRO_REPORT.md'
    figure = ROOT / 'results/training_curves_20261004.png'
    artifact = ROOT / 'results/report_audit_20261004.json'
    assert not any(p.exists() for p in (report, figure, artifact)), 'Refusing to overwrite report artifacts'
    target = test['literature_target']['unsupervised_BERT_base_average_STS_Spearman_percent']
    lines = ['# SimCSE reproduction and optimizer ablation', '',
             '## Setup', '',
             'Gao, Yao, and Chen, **SimCSE: Simple Contrastive Learning of Sentence Embeddings**, EMNLP 2021. '
             '[Official proceedings](https://aclanthology.org/2021.emnlp-main.552/), '
             '[paper and appendices](https://arxiv.org/html/2104.08821v2), '
             '[official MIT-licensed code](https://github.com/princeton-nlp/SimCSE).', '',
             'The target is the unsupervised BERT-base average STS Spearman result in Table 5. '
             'The author-reported single-model number is compared with our training-seed mean; these are different estimands.', '',
             'All numerical evidence is indexed in [RESULTS_INDEX.md](../RESULTS_INDEX.md). '
             'The frozen protocol, source/data audits, training grid, and final-test JSON are the source of every value below.', '',
             'Hyperparameters and checkpoints were chosen using validation only. No test-set tuning. '
             'All frozen seeds were retained; tests ran only after the full training grid finished.', '',
             '## Results', '', '| Configuration | Average STS Spearman (%) | Across-seed 95% CI |',
             '| --- | ---: | --- |', f'| Paper, Table 5 | {target:.2f} | Not supplied for this headline |']
    for name, value in test['summaries'].items():
        lo, hi = value['t95_ci']
        lines.append(f'| {name} | {value["mean"]:.4f} | [{lo:.4f}, {hi:.4f}] |')
    difference = test['paired_ablation_minus_reference']
    lo, hi = difference['t95_ci']
    lines += ['', f'Reference minus paper: **{test["reference_minus_paper_percentage_points"]:.4f} percentage points**.', '',
              f'Weight-decay ablation minus reference: **{difference["mean"]:.4f} percentage points**, '
              f'paired-seed t interval [{lo:.4f}, {hi:.4f}]. Negative results are retained. '
              'An interval spanning zero does not establish a directional improvement.', '',
              'Intervals use Student-t over independent training-seed scores, with matched-seed differences for the ablation. '
              'They describe training variability, not uncertainty over newly sampled test examples. Few seeds limit precision.', '',
              '### Per-seed and per-task results', '', '| Config | Seed | Task | Spearman (%) |', '| --- | ---: | --- | ---: |']
    for name, rows in test['raw'].items():
        for row in rows:
            for task, score in row['task_spearman_percent'].items():
                lines.append(f'| {name} | {row["seed"]} | {task} | {score:.4f} |')
    lines += ['', '## Training and ablation', '',
              'The controlled change is optimizer weight decay. Rationale and exact configurations are in '
              '[the frozen protocol](results/protocol_20261004.json). No reported weight-decay sweep was found in the '
              'inspected paper, appendices, or example; this does not rule out unpublished trials.', '',
              '| Config | Seed | Best validation Spearman | Steps | AMP skipped steps | Elapsed seconds | Peak allocated bytes |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    inputs = {str(test_path.relative_to(REPO)): sha(test_path), str(grid_path.relative_to(REPO)): sha(grid_path)}
    for name, rows in grid['runs'].items():
        for row in rows:
            seed = row['config']['seed']
            lines.append(f'| {name} | {seed} | {row["best_dev_spearman"]:.6f} | {row["completed_steps"]} | '
                         f'{row["amp_skipped_steps"]} | {row["elapsed_seconds"]:.3f} | {row["max_cuda_memory_bytes"]} |')
            curves = row['validation_curves']
            axes[0].plot([r['step'] for r in curves], [r['dev_spearman'] for r in curves], label=f'{name}/{seed}')
            progress = (resolve_legacy_path(row['selected_checkpoint']).parent / 'progress.jsonl').resolve()
            inputs[str(progress.relative_to(REPO))] = sha(progress)
            losses = [json.loads(line) for line in progress.read_text().splitlines()]
            losses = [r for r in losses if 'loss' in r and 'step' in r]
            axes[1].plot([r['step'] for r in losses], [max(r['loss'], 1e-12) for r in losses], alpha=.6)
    axes[0].set(xlabel='Optimizer-loop step', ylabel='Validation Spearman', title='Validation only')
    axes[0].legend(fontsize=6)
    axes[1].set(xlabel='Optimizer-loop step', ylabel='Training contrastive loss', yscale='log', title='Logged loss (floor for display)')
    fig.tight_layout(); fig.savefig(figure, dpi=160); plt.close(fig)
    lines += ['', '![Training curves](results/training_curves_20261004.png)', '', '## Deviations and limitations', '',
              '- Modern Python/Torch/Transformers port replaces legacy dependencies; Torch AdamW is not bit-identical to legacy Hugging Face AdamW. '
              'Head/loss/gradient parity is audited on a fixture, not a complete optimizer trajectory.',
              '- Author-hosted training corpus loses an exact normalized benchmark overlap under our audit. '
              'Document identities and semantic/near-duplicate leakage checks are unavailable.',
              '- Dataset mirror has no license card, so its license could not be established. '
              'Upstream Wikipedia terms do not establish every bundled file’s terms. Data are excluded from this repository.',
              '- Validation interval follows paper appendix rather than the released shell example. '
              'Final evaluation uses FP32 and a smaller inference batch; hardware/runtime/version deviations are recorded.',
              '- Single GPU only; no DDP or concurrent-system throughput claim. Exact device and versions are recorded per run.',
              '- Initial source/network/data parsing failures and smoke artifacts are preserved. '
              'Smoke runs are excluded from headline metrics. AMP skipped steps are disclosed above.',
              '- Test results are final for this frozen experiment. Do not change hyperparameters to close the paper gap.', '',
              '## Reproduction commands', '', 'Use the audited local source, data, and pretrained-model paths; install pinned '
              '`requirements.lock` in a suitable GPU environment. Confirm their recorded hashes before reuse.', '',
              '```bash', '.venv/bin/python paper_repro/run_training_grid.py',
              '.venv/bin/python paper_repro/evaluate_reproduction.py',
              '.venv/bin/python paper_repro/build_repro_report.py', '```', '',
              'Commands refuse to overwrite final artifacts. Use a fresh checkout/output location for a new experiment; '
              'complete matching checkpoints may be reused only after hash/protocol checks. No submission/publication claim is made.', '']
    report.write_text('\n'.join(lines))
    audit = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'script_sha256': sha(Path(__file__)), 'input_sha256': inputs,
             'output_sha256': {str(p.relative_to(REPO)): sha(p) for p in (report, figure)},
             'test_set_tuning': False, 'gpus_used': 0, 'seed': None,
             'scope': 'Reporting completed measurements, not another experiment.'}
    artifact.write_text(json.dumps(audit, indent=2) + '\n')
    index = REPO / 'RESULTS_INDEX.md'
    text = index.read_text()
    marker = '| SimCSE completed reproduction and weight-decay ablation |'
    assert marker not in text
    entry = (marker + ' [`reproduction_test_20261004.json`](paper_repro/results/reproduction_test_20261004.json), '
             '[`training_grid_20261004.json`](paper_repro/results/training_grid_20261004.json), '
             '[`report_audit_20261004.json`](paper_repro/results/report_audit_20261004.json) | '
             '[`evaluate_reproduction.py`](paper_repro/evaluate_reproduction.py), '
             '[`build_repro_report.py`](paper_repro/build_repro_report.py) | `'+test['git_commit']+'` | '
             'All frozen training seeds; official SentEval tasks; mean and seed t95 intervals, paired ablation change and paper gap. '
             'Validation-only selection; single GPU; dataset/runtime deviations disclosed. |')
    index.write_text(text.replace('| --- | --- | --- | --- | --- |', '| --- | --- | --- | --- | --- |\n'+entry, 1))
    print('WROTE', report, artifact, flush=True)

if __name__ == '__main__':
    main()
