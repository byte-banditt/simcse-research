#!/usr/bin/env python3
"""Measure batch feasibility only; random weights/tokens are not reproduction evidence."""
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import BertConfig, BertModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/memory_smoke_20261004.json'

def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite memory smoke result')
    seed, batch, length = 101, 64, 32
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = BertConfig()
    model = BertModel(config).to('cuda:0').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda')
    tokens = torch.randint(1, config.vocab_size, (batch, length), device='cuda:0').repeat_interleave(2, dim=0)
    mask = torch.ones_like(tokens)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            pooled = model(input_ids=tokens, attention_mask=mask).pooler_output.reshape(batch, 2, -1)
        first = torch.nn.functional.normalize(pooled[:, 0].float(), dim=-1)
        second = torch.nn.functional.normalize(pooled[:, 1].float(), dim=-1)
        loss = torch.nn.functional.cross_entropy(first @ second.T / 0.05, torch.arange(batch, device='cuda:0'))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        rows.append({'step': step, 'seconds': time.perf_counter() - start, 'loss': float(loss.detach())})
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Synthetic random-weight/random-token memory feasibility only. No trained result or real-data throughput.',
              'seed': seed, 'python': platform.python_version(), 'torch': torch.__version__,
              'transformers': transformers.__version__, 'hardware': torch.cuda.get_device_name(0),
              'gpus_used': 1, 'cuda': torch.version.cuda, 'batch_pairs': batch, 'sequence_length': length,
              'bert_config': config.to_dict(), 'parameters': sum(p.numel() for p in model.parameters()),
              'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
              'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(), 'amp': True,
              'steps': rows, 'training_data_used': False, 'test_data_used': False}
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(OUT, result['max_memory_allocated_bytes'], flush=True)

if __name__ == '__main__':
    main()
