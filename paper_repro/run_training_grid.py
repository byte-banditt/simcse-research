#!/usr/bin/env python3
"""Run the frozen SimCSE training design; leave final-test evaluation gated."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from project_paths import resolve_legacy_path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/training_grid_20261004.json'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    assert not OUT.exists(), 'Refusing to overwrite training summary'
    protocol_file = ROOT / 'results/protocol_20261004.json'
    protocol = json.loads(protocol_file.read_text())
    logs = ROOT / 'results/logs_20261004'
    logs.mkdir(exist_ok=True)
    runs = {}
    for name, config in protocol['configurations'].items():
        runs[name] = []
        for seed in protocol['seeds']:
            original = f'{name}_seed{seed}'
            existing = sorted((ROOT / 'results/training_runs').glob(f'{original}*/result.json'))
            reused = None
            for candidate in existing:
                saved = json.loads(candidate.read_text())
                conf = saved['config']
                if (saved.get('protocol_sha256') == sha(protocol_file)
                        and conf['seed'] == seed and conf['weight_decay'] == config['weight_decay']
                        and not conf['smoke_only'] and saved['completed_steps'] == conf['planned_steps']):
                    assert sha(resolve_legacy_path(saved['selected_checkpoint'])) == saved['selected_checkpoint_sha256']
                    reused = saved
                    break
            if reused:
                runs[name].append(reused)
                print('REUSE', original, flush=True)
                continue
            run_name = original
            retry = 0
            while (ROOT / 'results/training_runs' / run_name).exists() or (logs / f'{run_name}.log').exists():
                retry += 1
                run_name = f'{original}_retry{retry}'
            command = [sys.executable, str(ROOT / 'train_simcse.py'), '--seed', str(seed),
                       '--run-name', run_name, '--weight-decay', str(config['weight_decay'])]
            print('START', run_name, flush=True)
            with (logs / f'{run_name}.log').open('x') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise SystemExit(f'Training failed: {run_name}; code={result.returncode}; log preserved')
            saved = json.loads((ROOT / 'results/training_runs' / run_name / 'result.json').read_text())
            runs[name].append(saved)
            print('DONE', run_name, flush=True)
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'script_sha256': sha(Path(__file__)), 'protocol_sha256': sha(protocol_file),
               'training_seeds': protocol['seeds'], 'runs': runs,
               'final_test_evaluated': False, 'selection_used_validation_only': True}
    with OUT.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print('WROTE', OUT, flush=True)

if __name__ == '__main__':
    main()
