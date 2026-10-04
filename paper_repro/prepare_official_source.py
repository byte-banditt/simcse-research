#!/usr/bin/env python3
"""Fetch pinned reproduction code without optional demo/slides assets."""
import hashlib
import argparse
import json
import platform
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REV = '49fa580a853752ede55b8c76d9debf748e214d3f'
CHECKOUT = ROOT.parent / 'vendor/source_checkout_legacy'
DEST = ROOT.parent / 'vendor/official'
OUT = ROOT / 'results/official_source_audit_20261004.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--reuse-verified-files', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), 'Refusing to overwrite source audit'
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', REV], cwd=CHECKOUT, text=True).splitlines()
    files = [f for f in files if f in ('README.md', 'LICENSE', 'requirements.txt', 'train.py', 'evaluation.py', 'run_unsup_example.sh')
             or f.startswith(('simcse/', 'SentEval/senteval/', 'data/download_')) or f == 'SentEval/LICENSE']
    rows = []
    for name in files:
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/princeton-nlp/SimCSE/{REV}/{name}'
        expected = subprocess.check_output(['git', 'ls-tree', REV, name], cwd=CHECKOUT, text=True).split()[2]
        if args.reuse_verified_files and path.is_file():
            body = path.read_bytes()
            blob = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
            assert blob == expected, 'Existing source Git blob mismatch'
            rows.append({'file': name, 'url': url, 'sha256': hashlib.sha256(body).hexdigest(),
                         'bytes': len(body), 'git_blob': blob, 'reused_after_git_blob_verification': True})
            continue
        print('FETCH', name, flush=True)
        row = {'file': name, 'url': url, 'attempts': []}
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    body = response.read()
                blob = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
                assert blob == expected, 'Git blob hash mismatch'
                if path.exists():
                    assert path.read_bytes() == body, 'Existing source mismatch'
                else:
                    with path.open('xb') as stream:
                        stream.write(body)
                row.update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body), git_blob=blob)
                row['attempts'].append({'success': True})
                break
            except Exception as error:
                row['attempts'].append({'success': False, 'error': str(error)})
        rows.append(row)
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python': platform.python_version(), 'seed': None, 'gpus_used': 0,
              'official_repo': 'https://github.com/princeton-nlp/SimCSE', 'official_commit': REV,
              'source_path': str(DEST), 'files': rows, 'complete': all('sha256' in row for row in rows),
              'prior_attempts': ['Full clone: connection reset, early EOF.',
                                 'Filtered clone: commit/tree obtained; checkout timed out.',
                                 'Pinned complete tar archive: timed out; partial archive unused.']}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('COMPLETE', result['complete'], args.output, flush=True)

if __name__ == '__main__':
    main()
