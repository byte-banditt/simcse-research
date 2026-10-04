#!/usr/bin/env python3
"""Check official-head/objective parity and create verified SentEval directory layout."""
import ast
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from simcse_runtime import contrastive_loss

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'vendor/official'
OUT = ROOT / 'results/runtime_audit_20261004.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not OUT.exists(), 'Refusing to overwrite runtime audit'
    torch.manual_seed(101)
    tree = ast.parse((SOURCE / 'simcse/models.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ('MLPLayer', 'Similarity')]
    assert len(selected) == 2
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<verified-official-classes>', 'exec'), namespace)
    mlp = namespace['MLPLayer'](SimpleNamespace(hidden_size=16))
    first = torch.randn(4, 2, 16, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    projected_first = mlp(first)
    projected_second = torch.tanh(mlp.dense(second))
    reference_logits = namespace['Similarity'](0.05)(projected_first[:, 0].unsqueeze(1), projected_first[:, 1].unsqueeze(0))
    reference = torch.nn.functional.cross_entropy(reference_logits, torch.arange(4))
    port = contrastive_loss(projected_second)
    reference.backward()
    port.backward()
    projection_error = float((projected_first - projected_second).abs().max().detach())
    loss_error = float((reference - port).abs().detach())
    gradient_error = float((first.grad - second.grad).abs().max())
    assert projection_error == 0 and loss_error == 0 and gradient_error == 0
    layout = Path('/home/ayussh/datasets/simcse/senteval_root')
    layout.mkdir(exist_ok=True)
    link = layout / 'downstream'
    target = Path('/home/ayussh/datasets/simcse/evaluation')
    if link.exists():
        assert link.is_symlink() and link.resolve() == target
    else:
        link.symlink_to(target, target_is_directory=True)
    versions = {name: version(name) for name in ('torch', 'transformers', 'numpy', 'scipy', 'scikit-learn', 'huggingface-hub', 'tokenizers', 'safetensors')}
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': sha(Path(__file__)), 'runtime_sha256': sha(ROOT / 'simcse_runtime.py'),
              'official_model_code_sha256': sha(SOURCE / 'simcse/models.py'),
              'python': platform.python_version(), 'versions': versions, 'seed': 101, 'gpus_used': 0,
              'projection_max_error': projection_error, 'loss_error': loss_error, 'input_gradient_max_error': gradient_error,
              'fixture': 'Synthetic CPU tensor; only head/objective algebra checked, not pretrained loading, AMP trajectory, or full legacy Trainer.',
              'senteval_downstream_link': str(link), 'target': str(target)}
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    lock = ROOT / 'requirements.lock'
    assert not lock.exists()
    lock.write_text('# Measured local runtime; legacy-version deviation disclosed.\n' +
                    '\n'.join(f'{name}=={torch.__version__ if name == "torch" else val}' for name, val in versions.items()) + '\n')
    print(OUT, 'parity passed')

if __name__ == '__main__':
    main()
