#!/usr/bin/env python3
"""Wait for the existing frozen grid, then evaluate/report without tuning."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-tag', default='20261004_migrated')
    parser.add_argument('--training-pid', type=int, required=True)
    args = parser.parse_args()
    out = ROOT / f'results/completion_pipeline_{args.run_tag}.json'
    log = ROOT / f'results/completion_pipeline_{args.run_tag}.log'
    assert not out.exists() and not log.exists(), 'Refusing to overwrite completion artifacts'
    started = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    stat_path = Path(f'/proc/{args.training_pid}/stat')
    # Linux start ticks identify this process even if its PID is later reused.
    def identity():
        try:
            text = stat_path.read_text().rsplit(')', 1)[1].split()
            return text[19] if text[0] != 'Z' else None
        except FileNotFoundError:
            return None
    initial = identity()
    stages = []
    error = None
    with log.open('x') as stream:
        try:
            if initial is not None:
                cmdline = Path(f'/proc/{args.training_pid}/cmdline').read_bytes()
                assert b'run_training_grid.py' in cmdline, 'PID does not identify the intended grid'
                print('WAIT existing frozen training grid', args.training_pid, flush=True)
                while identity() == initial:
                    time.sleep(30)
            grid = ROOT / 'results/training_grid_20261004.json'
            assert grid.exists(), 'Training process exited without a completed grid; inspect retained training logs'
            # The evaluator checks protocol hashes and all frozen training seeds before tests.
            for script in ('evaluate_reproduction.py', 'build_repro_report.py'):
                print('START', script, flush=True)
                result = subprocess.run([sys.executable, str(ROOT / script)], cwd=REPO,
                                        stdout=stream, stderr=subprocess.STDOUT)
                stream.flush()
                stages.append({'script': script, 'returncode': result.returncode})
                assert result.returncode == 0, f'{script} failed; inspect {log}'
                print('DONE', script, flush=True)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            print(error, file=stream, flush=True)
    result = {'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': commit, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python': sys.version, 'training_pid': args.training_pid, 'training_process_start_ticks': initial,
              'seed': None, 'gpus_used_by_supervisor': 0, 'stages': stages,
              'status': 'failed' if error else 'completed', 'error': error,
              'test_set_tuning': False, 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
    with out.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    if error:
        raise SystemExit(error)
    # Only task-owned artifacts are staged; unrelated working-tree changes stay untouched.
    subprocess.run(['git', 'add', 'paper_repro/results', 'paper_repro/REPRO_REPORT.md', 'RESULTS_INDEX.md'],
                   cwd=REPO, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'feat(research): record SimCSE reproduction gate', 'paper_repro/results', 'paper_repro/REPRO_REPORT.md', 'RESULTS_INDEX.md'], cwd=REPO, check=True)
    print('COMPLETE frozen SimCSE gate; limitations are documented in the report', flush=True)

if __name__ == '__main__':
    main()
