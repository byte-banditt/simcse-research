#!/usr/bin/env python3
"""Evaluate frozen seed checkpoints with official SentEval; no test-driven selection."""
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from transformers import AutoTokenizer

from simcse_runtime import UnsupervisedSimCSE

from project_paths import resolve_legacy_path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'vendor/official'
OUT = ROOT / 'results/reproduction_test_20261004.json'
GRID = ROOT / 'results/training_grid_20261004.json'
sys.path.insert(0, str(SOURCE / 'SentEval'))
import senteval

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def ci(values):
    a = np.asarray(values, dtype=float)
    assert len(a) == 3 and np.isfinite(a).all()
    mean = float(a.mean())
    width = float(t.ppf(0.975, 2) * a.std(ddof=1) / math.sqrt(3))
    return {'mean': mean, 't95_ci': [mean - width, mean + width], 'seeds': 3}

def main():
    assert not OUT.exists(), 'Refusing to overwrite test results'
    protocol_path = ROOT / 'results/protocol_20261004.json'
    protocol = json.loads(protocol_path.read_text())
    grid = json.loads(GRID.read_text())
    assert grid['protocol_sha256'] == sha(protocol_path)
    assert all(len(rows) == 3 for rows in grid['runs'].values()), 'Complete all frozen seed training before tests'
    assert all(not row['config']['smoke_only'] for rows in grid['runs'].values() for row in rows)
    model_path = Path('/home/ayussh/models/simcse_bert_base')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tasks = protocol['evaluation']['tasks']
    raw = {}
    for name, entries in grid['runs'].items():
        raw[name] = []
        for train in sorted(entries, key=lambda x: x['config']['seed']):
            path = resolve_legacy_path(train['selected_checkpoint'])
            assert sha(path) == train['selected_checkpoint_sha256']
            model = UnsupervisedSimCSE(model_path).to('cuda:0').eval()
            state = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model'], strict=True)
            def batcher(params, batch):
                if batch and batch[0] and isinstance(batch[0][0], bytes):
                    batch = [[word.decode('utf-8') for word in sentence] for sentence in batch]
                texts = [' '.join(sentence) for sentence in batch]
                tokens = tokenizer(texts, padding=True, return_tensors='pt').to('cuda:0')
                with torch.inference_mode():
                    return model.embeddings(tokens).cpu().numpy()
            params = {'task_path': '/home/ayussh/datasets/simcse/evaluation', 'usepytorch': True,
                      'batch_size': 32, 'nhid': 0, 'kfold': 10, 'seed': train['config']['seed'],
                      'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}}
            # Mirror archive has STS/ and SICK/ directly. Engine expects /downstream/.
            params['task_path'] = '/home/ayussh/datasets/simcse/senteval_root'
            se = senteval.engine.SE(params, batcher)
            values = se.eval(tasks)
            scores = {task: float(values[task]['all']['spearman']['all'] * 100) if task.startswith('STS') and task not in ('STSBenchmark',)
                      else float(values[task]['test']['spearman'][0] * 100) for task in tasks}
            # Official printer rounds each task before averaging; keep both conventions.
            average = float(np.mean(list(scores.values())))
            printed_average = float(np.mean([float(f'{score:.2f}') for score in scores.values()]))
            raw[name].append({'seed': train['config']['seed'], 'training_git_commit': train['git_commit'],
                              'checkpoint_sha256': train['selected_checkpoint_sha256'], 'task_spearman_percent': scores,
                              'average_spearman_percent': average, 'official_printer_average_percent': printed_average})
            del model
            torch.cuda.empty_cache()
            print('EVALUATED', name, train['config']['seed'], average, flush=True)
    summary = {name: ci([row['average_spearman_percent'] for row in entries]) for name, entries in raw.items()}
    reference = [row['average_spearman_percent'] for row in raw['reference']]
    ablation = [row['average_spearman_percent'] for row in raw['weight_decay_ablation']]
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': sha(Path(__file__)), 'runtime_sha256': sha(ROOT / 'simcse_runtime.py'),
              'grid_sha256': sha(GRID), 'protocol_sha256': sha(protocol_path),
              'python': platform.python_version(),
              'versions': {p: version(p) for p in ('torch', 'transformers', 'numpy', 'scipy', 'scikit-learn')},
              'hardware': {'device': torch.cuda.get_device_name(0), 'gpus_used': 1, 'cuda': torch.version.cuda},
              'official_source_audit': 'official_source_audit_retry1_20261004.json',
              'training_seeds': protocol['seeds'], 'literature_target': protocol['literature_target'],
              'raw': raw, 'summaries': summary, 'paired_ablation_minus_reference': ci(np.asarray(ablation) - np.asarray(reference)),
              'reference_minus_paper_percentage_points': summary['reference']['mean'] - protocol['literature_target']['unsupervised_BERT_base_average_STS_Spearman_percent'],
              'test_set_tuning': False,
              'limitations': ['Three training seeds, t interval approximate; original runtime and exact corpus differ.',
                              'Paper headline is an author-reported single-model result, not a matched seed mean.',
                              'Keep current tests fixed; do not tune to close a reported gap.']}
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('WROTE', OUT)

if __name__ == '__main__':
    main()
