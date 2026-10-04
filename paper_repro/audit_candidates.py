#!/usr/bin/env python3
"""Fetch primary sources and record literature targets, never experimental results."""
import hashlib
import json
import platform
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/candidate_sources_20261004.json'
SOURCES = {
    'simcse_venue': 'https://aclanthology.org/2021.emnlp-main.552/',
    'simcse_paper': 'https://arxiv.org/html/2104.08821v2',
    'simcse_license': 'https://raw.githubusercontent.com/princeton-nlp/SimCSE/main/LICENSE',
    'nafnet_venue': 'https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3043_ECCV_2022_paper.php',
    'nafnet_paper': 'https://arxiv.org/html/2204.04676v3',
    'nafnet_license': 'https://raw.githubusercontent.com/megvii-research/NAFNet/main/LICENSE',
    'depth_venue': 'https://proceedings.neurips.cc/paper_files/paper/2024/hash/26cfdcd8fe6fd75cc53e92963a656c58-Abstract-Conference.html',
    'depth_paper': 'https://arxiv.org/html/2406.09414v2',
    'depth_license_policy': 'https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/README.md',
    'depth_benchmark': 'https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/DA-2K.md',
}

def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite source audit')
    sources = {}
    for key, url in SOURCES.items():
        print('FETCH', key, flush=True)
        try:
            with urllib.request.urlopen(url, timeout=12) as response:
                body = response.read()
                sources[key] = {'url': url, 'resolved_url': response.url,
                                'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}
        except Exception as error:
            sources[key] = {'url': url, 'fetch_failed': str(error),
                            'verification': 'Agent checked primary page using web tool; local source hash unavailable. Human must independently verify.'}
            continue
        text = body.decode('utf-8', errors='replace')
        marker = {'simcse_paper': '76.25', 'nafnet_paper': '40.30', 'depth_paper': '95.3',
                  'simcse_license': 'MIT License', 'nafnet_license': 'Apache License'}.get(key)
        if marker:
            assert marker in text, (key, marker)
            sources[key]['verified_marker'] = marker
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python': platform.python_version(), 'seed': None, 'gpu_count_used': 0,
              'purpose': 'Literature source verification only; no reproduction measurements.',
              'sources': sources,
              'targets': {'SimCSE': {'table': 5, 'model': 'Unsupervised BERT base', 'average_STS_Spearman_percent': 76.25},
                          'NAFNet': {'table': 6, 'dataset': 'SIDD', 'PSNR_dB': 40.30},
                          'Depth Anything V2': {'table': 3, 'model': 'ViT-S', 'DA_2K_accuracy_percent': 95.3}}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(OUT)

if __name__ == '__main__':
    main()
