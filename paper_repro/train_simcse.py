#!/usr/bin/env python3
"""Train one seed with validation-only checkpoint selection; never run final tests."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from simcse_runtime import UnsupervisedSimCSE, evaluate_dev, load_stsb_dev

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/ayussh/datasets/simcse')
MODEL = Path('/home/ayussh/models/simcse_bert_base')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--max-steps', type=int, default=0, help='Nonzero is a smoke deviation, never headline reproduction')
    args = parser.parse_args()
    run = ROOT / 'results/training_runs' / args.run_name
    assert not run.exists(), 'Refusing to overwrite a training run'
    assert args.weight_decay >= 0 and args.max_steps >= 0
    audit_path = ROOT / 'results/data_audit_20261004.json'
    audit = json.loads(audit_path.read_text())
    training_file = Path(audit['training_file'])
    assert sha(training_file) == audit['training_file_sha256'], 'Training bytes changed'
    source_audit = ROOT / 'results/official_source_audit_retry1_20261004.json'
    assert json.loads(source_audit.read_text())['complete']
    assert (MODEL / 'model.safetensors').is_file(), 'Pinned pretrained weights not downloaded'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = UnsupervisedSimCSE(MODEL).to('cuda:0')
    # Match HF AdamW's exclusion of bias and LayerNorm from decay.
    groups = [{'params': [p for n, p in model.named_parameters() if not (n.endswith('bias') or 'LayerNorm' in n)],
               'weight_decay': args.weight_decay},
              {'params': [p for n, p in model.named_parameters() if n.endswith('bias') or 'LayerNorm' in n], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=3e-5, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda')
    # Text-file rows are separated by physical newlines; Unicode separators may
    # occur inside sentences and must not create extra training examples.
    with training_file.open(encoding='utf-8') as stream:
        sentences = [line.rstrip('\n') for line in stream]
    assert len(sentences) == audit['training_sentences']
    order = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(sentences, batch_size=64, shuffle=True, generator=order, num_workers=0,
                        drop_last=False, collate_fn=lambda texts: tokenizer(texts, padding=True, truncation=True,
                                                                         max_length=32, return_tensors='pt'))
    epoch_steps = math.ceil(len(sentences) / 64)
    total_steps = min(epoch_steps, args.max_steps) if args.max_steps else epoch_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, (total_steps - step) / total_steps))
    dev_path, dev = load_stsb_dev(DATA / 'evaluation')
    run.mkdir(parents=True)
    provenance = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'script_sha256': sha(Path(__file__)), 'runtime_sha256': sha(ROOT / 'simcse_runtime.py'),
                  'data_audit_sha256': sha(audit_path), 'official_source_audit_sha256': sha(source_audit),
                  'protocol_sha256': sha(ROOT / 'results/protocol_20261004.json'),
                  'pretrained_model_sha256': sha(MODEL / 'model.safetensors'),
                  'pretrained_model_revision': '86b5e0934494bd15c9632b12f734a8a67f723594',
                  'dev_file_sha256': sha(dev_path), 'python': platform.python_version(),
                  'versions': {p: version(p) for p in ('torch', 'transformers', 'numpy', 'scipy', 'huggingface-hub', 'tokenizers', 'safetensors')},
                  'hardware': {'device': torch.cuda.get_device_name(0), 'gpus_used': 1, 'cuda': torch.version.cuda},
                  'config': {'seed': args.seed, 'batch_size': 64, 'max_train_length': 32, 'epochs': 1,
                             'learning_rate': 3e-5, 'temperature': 0.05, 'optimizer': 'torch AdamW',
                             'weight_decay': args.weight_decay, 'gradient_clip_norm': 1.0,
                             'warmup_steps': 0, 'schedule': 'linear decay', 'amp': True,
                             'validation_every_steps': 250, 'training_sentences': len(sentences),
                             'planned_steps': total_steps, 'smoke_only': bool(args.max_steps),
                             'selection': 'Maximum STS-B development Spearman; first checkpoint wins ties',
                             'test_set_tuning': False},
                  'deviations': ['Current Torch/Transformers runtime, modern port rather than original legacy Trainer.',
                                 'Float32 cosine/contrastive logits under mixed precision.',
                                 'Exact normalized validation/test sentence overlaps removed, as audited.',
                                 'Validation checks every 250 steps follow paper appendix, not released example every 125.',
                                 'Development evaluation loads only sts-dev.csv; original SentEval constructor also loads other splits.']}
    with (run / 'config.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    model.train()
    torch.cuda.reset_peak_memory_stats()
    best = -math.inf
    best_path = None
    curves = []
    started = time.perf_counter()
    skipped = 0
    with (run / 'progress.jsonl').open('x') as progress:
        progress.write(json.dumps({'kind': 'provenance', **provenance}) + '\n')
        progress.flush()
        for step, batch in enumerate(loader, 1):
            batch = batch.to('cuda:0')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model.loss(batch)
            assert torch.isfinite(loss), 'Nonfinite training loss'
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            else:
                skipped += 1
            if step % 25 == 0 or step == 1:
                row = {'step': step, 'loss': float(loss.detach()), 'lr': optimizer.param_groups[0]['lr'],
                       'elapsed_seconds': time.perf_counter() - started, 'amp_skipped_steps': skipped}
                progress.write(json.dumps(row) + '\n')
                progress.flush()
            if step % 250 == 0 or step == total_steps:
                validation = evaluate_dev(model, tokenizer, dev, batch_size=32)
                row = {'step': step, 'loss': float(loss.detach()), 'dev_spearman': validation['spearman'],
                       'dev_pair_count': validation['pair_count'], 'elapsed_seconds': time.perf_counter() - started}
                curves.append(row)
                progress.write(json.dumps(row) + '\n')
                progress.flush()
                if validation['spearman'] > best:
                    best = validation['spearman']
                    best_path = run / f'checkpoint_step_{step:06d}.pt'
                    assert not best_path.exists()
                    torch.save({'model': model.state_dict(), 'step': step, 'dev_spearman': best,
                                'seed': args.seed, 'git_commit': provenance['git_commit'],
                                'provenance': provenance}, best_path)
                print(args.run_name, 'STEP', step, 'DEV', validation['spearman'], flush=True)
            if step >= total_steps:
                break
    result = {**provenance, 'completed_steps': step, 'elapsed_seconds': time.perf_counter() - started,
              'amp_skipped_steps': skipped, 'max_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
              'validation_curves': curves, 'best_dev_spearman': best, 'selected_checkpoint': str(best_path),
              'selected_checkpoint_sha256': sha(best_path), 'final_test_evaluated': False}
    with (run / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('DONE', run, flush=True)

if __name__ == '__main__':
    main()
