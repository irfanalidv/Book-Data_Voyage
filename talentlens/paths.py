"""Canonical path resolution for the Data Voyage / TalentLens monorepo."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* (or cwd) until ``pyproject.toml`` is found."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    msg = "Could not find pyproject.toml — run commands from the repository root."
    raise FileNotFoundError(msg)


REPO_ROOT: Path = find_repo_root()
DATA_DIR: Path = REPO_ROOT / "data"
BOOK_DIR: Path = REPO_ROOT / "book"


def jobs_clean_path() -> Path:
    """Return the preferred jobs_clean dataset path.

    Prefers ``data/clean/jobs_clean.large.csv`` (the real-data file
    written by ``make collect-dataset``) when it exists, otherwise falls
    back to the small bundled ``data/clean/jobs_clean.csv``. Chapters
    should call this rather than hard-coding either path so the choice
    lives in one place.

    Returns:
        Absolute :class:`Path` to the dataset chapters should read. May
        not exist if neither file is present - callers should still
        check ``.exists()``.
    """
    large = DATA_DIR / "clean" / "jobs_clean.large.csv"
    small = DATA_DIR / "clean" / "jobs_clean.csv"
    return large if large.exists() else small


def display_path(path: Path | str) -> str:
    """Return *path* relative to the repository root when it lies inside it.

    Reports use this so they show ``data/clean/jobs_clean.csv`` rather than
    an absolute path that includes the reader's home directory. Paths
    outside the repository are returned unchanged.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def role_classifier_path() -> Path:
    """Return the preferred role classifier model path.

    Prefers ``models/role_classifier_v2.joblib`` (Chapter 10's
    feature-engineered model) when it exists, otherwise falls back
    to ``book/ch09/models/role_classifier.joblib`` (Chapter 9's baseline,
    where Chapter 9's script saves it).
    Chapters 19 and 22 should call this rather than hard-coding
    either path.

    Returns:
        Absolute :class:`Path` to the preferred model file. May
        not exist if neither has been trained yet - callers should
        check ``.exists()``.
    """
    v2 = REPO_ROOT / "models" / "role_classifier_v2.joblib"
    v1 = REPO_ROOT / "book" / "ch09" / "models" / "role_classifier.joblib"
    return v2 if v2.exists() else v1
