"""FastAPI application factory - Chapter 20 ``build_app``."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from talentlens_core.paths import ensure_repo_on_sys_path


def create_app(
    *,
    jobs_df: Optional[pd.DataFrame] = None,
    rate_limit_max: int = 120,
    rate_window_seconds: float = 60.0,
) -> Any:
    """Return the TalentLens FastAPI app (same as ``book.ch19``)."""
    if ensure_repo_on_sys_path() is None:
        raise RuntimeError("create_app requires Book-Data_Voyage checkout on disk.")
    from book.ch19.ch19_fastapi_deployment import build_app

    return build_app(
        jobs_df=jobs_df,
        rate_limit_max=rate_limit_max,
        rate_window_seconds=rate_window_seconds,
    )
