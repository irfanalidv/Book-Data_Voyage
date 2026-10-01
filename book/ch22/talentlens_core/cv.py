"""CV parsing - Chapter 18 ``CVParser`` with stub LLM by default."""

from __future__ import annotations

from typing import Any

from talentlens_core.paths import ensure_repo_on_sys_path


def create_cv_parser(*, provider: str = "stub", **config_overrides: Any) -> Any:
    """Build ``CVParser`` + ``LLMClient`` (see Chapter 18)."""
    if ensure_repo_on_sys_path() is None:
        raise RuntimeError("create_cv_parser requires Book-Data_Voyage checkout on disk.")
    from book.ch17.ch17_llm_generation import Config, CVParser, LLMClient

    cfg = Config(provider=provider, **config_overrides)
    client = LLMClient(cfg)
    return CVParser(client, cfg)
