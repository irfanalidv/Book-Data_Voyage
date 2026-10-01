"""TalentLens - development package for the Data Voyage running project."""

from talentlens.config import Settings, get_settings
from talentlens.paths import BOOK_DIR, DATA_DIR, REPO_ROOT, find_repo_root

__version__ = "0.1.0"

__all__ = [
    "BOOK_DIR",
    "DATA_DIR",
    "REPO_ROOT",
    "Settings",
    "__version__",
    "find_repo_root",
    "get_settings",
]
