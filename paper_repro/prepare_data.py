#!/usr/bin/env python3
"""Audit official mirror bytes and exclude exact validation/test sentence overlap."""
import csv
import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/ayussh/datasets/simcse')
OUT = ROOT / 'results/data_audit_20261004.json'
EXTRACTED = DATA / 'evaluation'
TRAIN = DATA / 'wiki_train_exact_disjoint_20261004.txt'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def normalized(text):
    return ' '.join(text.split()).casefold()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-extracted', action='store_true')
    args = parser.parse_args()
    assert not OUT.exists() and not TRAIN.exists(), 'Refusing to overwrite data audit or split data'
    archive = DATA / 'senteval.tar'
    wiki = DATA / 'wiki1m_for_simcse.txt'
    assert archive.is_file() and wiki.is_file(), 'Official data downloads not finished'
    if EXTRACTED.exists():
        assert args.reuse_extracted, 'Existing extraction; use --reuse-extracted for byte verification'
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    destination = (EXTRACTED / member.name).resolve()
                    assert destination.is_relative_to(EXTRACTED.resolve())
                    expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                    assert sha(destination) == expected, f'Extracted data mismatch: {member.name}'
    else:
        EXTRACTED.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(EXTRACTED, filter='data')
    protected = set()
    benchmark_files = []
    for file in sorted(EXTRACTED.rglob('*')):
        if not file.is_file():
            continue
        if file.name.startswith('STS.input.') and file.suffix == '.txt':
            for line in file.read_text(encoding='utf-8').splitlines():
                first, second = line.split('\t')
                protected.update((normalized(first), normalized(second)))
        elif file.name in ('sts-dev.csv', 'sts-test.csv'):
            with file.open(encoding='utf-8') as stream:
                for row in csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE):
                    protected.update((normalized(row[5]), normalized(row[6])))
        elif file.name in ('SICK_trial.txt', 'SICK_test_annotated.txt'):
            with file.open(encoding='utf-8') as stream:
                rows = csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE)
                next(rows)
                for row in rows:
                    protected.update((normalized(row[1]), normalized(row[2])))
        benchmark_files.append({'path': str(file.relative_to(EXTRACTED)), 'bytes': file.stat().st_size, 'sha256': sha(file)})
    total = kept = blank = 0
    removed = []
    with wiki.open(encoding='utf-8') as source, TRAIN.open('x', encoding='utf-8') as target:
        for number, line in enumerate(source, 1):
            total += 1
            norm = normalized(line)
            if not norm:
                blank += 1
                continue
            if norm in protected:
                removed.append({'line': number, 'sentence_sha256': hashlib.sha256(norm.encode()).hexdigest()})
                continue
            assert norm not in protected, 'Exact benchmark sentence leakage'
            target.write(line.rstrip('\n') + '\n')
            kept += 1
    assert kept and protected
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': sha(Path(__file__)), 'python': platform.python_version(), 'seed': None, 'gpus_used': 0,
              'dataset_repo': 'princeton-nlp/datasets-for-simcse', 'revision': 'e145e8bb659b2aa2669f32ef79cb4cdef6c58fef',
              'primary_download_script': 'https://github.com/princeton-nlp/SimCSE/blob/49fa580a853752ede55b8c76d9debf748e214d3f/data/download_wiki.sh',
              'license_metadata': 'UNVERIFIED: author-hosted mirror has no dataset card/license. Wikipedia upstream terms do not establish the provenance/license of every mirrored sentence or bundled benchmark. No data redistribution.',
              'upstream_terms_url': 'https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use',
              'wiki_sha256': sha(wiki), 'wiki_bytes': wiki.stat().st_size,
              'senteval_archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
              'training_file': str(TRAIN), 'training_file_sha256': sha(TRAIN), 'training_bytes': TRAIN.stat().st_size,
              'source_lines': total, 'training_sentences': kept, 'blank_lines_removed': blank,
              'exact_overlap_removed': removed, 'protected_unique_sentences': len(protected),
              'exact_leakage_assertion': 'Passed: training sentences absent from normalized validation/test sentences.',
              'limitations': ['No semantic/near-duplicate leakage check.', 'Original Wikipedia document IDs are unavailable; document-cluster separation cannot be asserted.',
                              'Sentence filtering is a deviation from the original training corpus, performed only for leakage prevention, not test-score tuning.'],
              'benchmark_files': benchmark_files}
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(OUT, kept, 'removed', len(removed), flush=True)

if __name__ == '__main__':
    main()
