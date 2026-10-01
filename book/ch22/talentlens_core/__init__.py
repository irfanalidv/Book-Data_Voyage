"""TalentLens core - thin integration layer over book chapter modules."""

from __future__ import annotations

__version__ = "0.1.0"

from talentlens_core.app import create_app
from talentlens_core.classification import load_role_classifier, predict_job_role
from talentlens_core.cv import create_cv_parser
from talentlens_core.paths import ensure_repo_on_sys_path, find_repo_root
from talentlens_core.search import connect_vector_store

__all__ = [
    "__version__",
    "create_app",
    "create_cv_parser",
    "connect_vector_store",
    "ensure_repo_on_sys_path",
    "find_repo_root",
    "load_role_classifier",
    "predict_job_role",
]
