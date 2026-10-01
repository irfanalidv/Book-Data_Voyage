"""Resolve the Book-Data_Voyage repository root when running from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def find_repo_root() -> Optional[Path]:
    """Return repo root if this file lives under ``.../book/ch22/talentlens_core``."""
    here = Path(__file__).resolve()
    for base in here.parents:
        marker = base / "book" / "ch19" / "ch19_fastapi_deployment.py"
        if marker.is_file():
            return base
    return None


def ensure_repo_on_sys_path() -> Optional[Path]:
    """Insert repository root on ``sys.path`` so ``book.*`` imports work."""
    root = find_repo_root()
    if root is None:
        return None
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
