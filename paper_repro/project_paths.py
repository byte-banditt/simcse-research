"""Resolve checkpoint paths recorded before project organization."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def resolve_legacy_path(value):
    path = Path(value)
    for prefix in (Path('/home/ayussh/projects/cve_rag/research/paper_repro'), Path('/home/ayussh/projects/research/paper_repro')):
        try:
            return ROOT / path.relative_to(prefix)
        except ValueError:
            pass
    return path if path.is_absolute() else ROOT / path
