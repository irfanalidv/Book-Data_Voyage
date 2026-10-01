"""Chapter 6 tests - cleaning pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))

from ch06_data_cleaning_preprocessing import (  # noqa: E402
    Config,
    _generate_demo_raw,
    clean_titles,
    final_dedup,
    impute_salary,
    run_cleaning_pipeline,
)

from talentlens.paths import DATA_DIR  # noqa: E402


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    raw = tmp_path / "raw" / "jobs_raw.csv"
    raw_demo = tmp_path / "raw" / "jobs_raw.demo.csv"
    clean = tmp_path / "clean" / "jobs_clean.csv"
    clean_demo = tmp_path / "clean" / "jobs_clean.demo.csv"
    figures = tmp_path / "figures"
    reports = tmp_path / "reports"
    raw.parent.mkdir(parents=True)
    demo = _generate_demo_raw()
    demo.to_csv(raw_demo, index=False)
    return Config(
        raw_path=raw,
        raw_demo_path=raw_demo,
        clean_path=clean,
        clean_demo_path=clean_demo,
        figures_dir=figures,
        reports_dir=reports,
    )


def test_pipeline_handles_demo_data_end_to_end(cfg: Config) -> None:
    before, after, output_path = run_cleaning_pipeline(cfg, overwrite=True)
    assert len(after) > 0
    assert output_path == cfg.clean_path
    assert cfg.clean_path.is_file()
    assert "salary_band" in after.columns
    assert len(before) >= len(after)


def test_pipeline_preserves_existing_clean_without_overwrite(cfg: Config) -> None:
    _, first, _ = run_cleaning_pipeline(cfg, overwrite=True)
    bundled_bytes = cfg.clean_path.read_bytes()
    cfg.raw_path.write_text("job_id,title\n", encoding="utf-8")
    before, after, output_path = run_cleaning_pipeline(cfg, overwrite=False)
    assert cfg.clean_path.read_bytes() == bundled_bytes
    assert output_path == cfg.clean_demo_path
    assert output_path.is_file()
    assert len(after) != len(first) or not before.equals(after)


def test_clean_titles_handles_known_variants() -> None:
    df = pd.DataFrame(
        {
            "title": [
                "ML Engineer (Bangalore)",
                "ML Engineer",
                "Senior NLP Engineer!!!",
            ],
            "company": ["A", "A", "B"],
            "description": ["x" * 60, "y" * 60, "z" * 60],
        }
    )
    out = clean_titles(df)
    assert out["title"].nunique() <= 2


def test_impute_salary_marks_imputed_rows(cfg: Config) -> None:
    df = pd.DataFrame(
        {
            "title": ["DS", "DE"],
            "company": ["A", "B"],
            "description": ["a" * 60, "b" * 60],
            "salary_min": [None, 1_000_000],
            "salary_max": [None, 1_200_000],
            "source": ["demo", "demo"],
        }
    )
    out = impute_salary(df, cfg)
    assert "salary_imputed" in out.columns
    assert out["salary_imputed"].iloc[0]
    assert not out["salary_imputed"].iloc[1]


def test_final_dedup_removes_duplicates_by_canonical_key() -> None:
    df = pd.DataFrame(
        {
            "title": ["ML Engineer", "ML Engineer"],
            "company": ["Razorpay", "Razorpay"],
            "city": ["Bangalore", "Bangalore"],
            "description": ["a" * 60, "b" * 60],
        }
    )
    out = final_dedup(df)
    assert len(out) == 1


def test_config_defaults_use_data_dir() -> None:
    default = Config()
    assert default.raw_path == DATA_DIR / "raw" / "jobs_raw.csv"
    assert default.raw_demo_path == DATA_DIR / "raw" / "jobs_raw.demo.csv"
    assert default.clean_path == DATA_DIR / "clean" / "jobs_clean.csv"
    assert default.clean_demo_path == DATA_DIR / "clean" / "jobs_clean.demo.csv"
