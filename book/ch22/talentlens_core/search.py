"""Vector store - Chapter 17 ``VectorStore``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from talentlens_core.paths import ensure_repo_on_sys_path


def connect_vector_store(db_path: Optional[Path] = None, *, dim: int = 384) -> Any:
    """Open SQLite-backed ``VectorStore`` (call ``build_index`` before ``search``)."""
    if ensure_repo_on_sys_path() is None:
        raise RuntimeError("connect_vector_store requires Book-Data_Voyage checkout on disk.")
    from book.ch16.ch16_rag_vector_search import Config, VectorStore

    cfg = Config()
    path = db_path or cfg.db_path
    store = VectorStore(path, dim=dim)
    store.connect()
    return store
