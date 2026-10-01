"""Chapter 4 tests - sources table, schema, Adzuna skip (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))

from ch04_data_sources import (  # noqa: E402
    normalize_remoteok_posting,
    source_comparison_table,
    try_adzuna_sample,
)

from talentlens.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_source_comparison_table_has_three_rows() -> None:
    rows = source_comparison_table()
    assert len(rows) == 3
    names = {r.name for r in rows}
    assert "Adzuna API" in names
    assert "RemoteOK API" in names
    assert "GitHub Jobs archive (Kaggle)" in names


def test_schema_normalisation_handles_remoteok_sample() -> None:
    raw = {
        "id": 42,
        "position": "Data Scientist",
        "company": "Acme",
        "tags": ["python"],
        "salary_min": 50000,
        "date": "2026-01-01",
        "url": "https://example.com/job",
    }
    out = normalize_remoteok_posting(raw)
    assert out["source"] == "remoteok"
    assert out["title"] == "Data Scientist"
    assert out["company"] == "Acme"
    assert out["is_remote"] is True
    assert out["salary_min"] == 50000.0


def test_adzuna_fetch_is_gracefully_skipped_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_API_KEY", raising=False)
    get_settings.cache_clear()
    msg = try_adzuna_sample()
    assert "skipped" in msg.lower()
    assert "ADZUNA_APP_ID" in msg
