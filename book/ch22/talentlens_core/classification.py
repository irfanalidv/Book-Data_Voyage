"""Role classification - delegates to Chapter 9 after repo path setup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from talentlens_core.paths import ensure_repo_on_sys_path, find_repo_root


def load_role_classifier(path: Optional[Path] = None) -> Any:
    """Load Chapter 9's ``role_classifier.joblib`` from env, explicit path, or the checkout."""
    import joblib

    env = os.environ.get("TALENTLENS_ROLE_MODEL")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"TALENTLENS_ROLE_MODEL not a file: {p}")
        return joblib.load(p)
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return joblib.load(path)
    root = find_repo_root()
    if root is None:
        raise FileNotFoundError(
            "Could not locate repository root. Set TALENTLENS_ROLE_MODEL or install from checkout."
        )
    default = root / "book" / "ch09" / "models" / "role_classifier.joblib"
    if not default.is_file():
        raise FileNotFoundError(
            f"No classifier at {default}. Run: python book/ch09/ch09_supervised_learning.py"
        )
    return joblib.load(default)


def predict_job_role(job: dict[str, Any], pipeline: Optional[Any] = None) -> dict[str, Any]:
    """Predict role label + confidence (Chapter 9 ``predict_role``)."""
    if ensure_repo_on_sys_path() is None:
        raise RuntimeError("predict_job_role requires Book-Data_Voyage checkout on disk.")
    from book.ch09.ch09_supervised_learning import predict_role as _predict

    pl = pipeline if pipeline is not None else load_role_classifier()
    return _predict(job, pl)
