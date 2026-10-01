"""The bundled dataset must be reproducible from the reader path (Chapter 5 demo -> Chapter 6)."""

from __future__ import annotations

import pandas as pd

from book.ch05.ch05_data_collection import Config as Ch05Config
from book.ch05.ch05_data_collection import DataCollectionPipeline
from book.ch06.ch06_data_cleaning_preprocessing import Config as Ch06Config
from book.ch06.ch06_data_cleaning_preprocessing import run_cleaning_pipeline
from talentlens.paths import DATA_DIR

CANONICAL_ROLES = {"AI Engineer", "ML Engineer", "Data Scientist", "Data Engineer", "Data Analyst"}


def test_demo_pipeline_reproduces_bundled_csv(tmp_path):
    c5 = Ch05Config()
    c5.raw_demo_data_path = tmp_path / "jobs_raw.demo.csv"
    c5.figures_dir = tmp_path / "fig5"
    c5.reports_dir = tmp_path / "rep5"
    DataCollectionPipeline(c5).run(live=False)

    c6 = Ch06Config()
    c6.raw_demo_path = c5.raw_demo_data_path
    c6.clean_demo_path = tmp_path / "jobs_clean.demo.csv"
    _, df, _ = run_cleaning_pipeline(c6, overwrite=False)

    bundled = pd.read_csv(DATA_DIR / "clean" / "jobs_clean.csv")
    rebuilt = pd.read_csv(c6.clean_demo_path)
    pd.testing.assert_frame_equal(rebuilt, bundled)


def test_bundled_dataset_is_usable():
    df = pd.read_csv(DATA_DIR / "clean" / "jobs_clean.csv")
    assert len(df) > 500
    assert CANONICAL_ROLES <= set(df["role_category"])
    assert df["salary_band"].notna().all()
    salary = df.loc[df["salary_disclosed"], "salary_annual_inr"]
    assert salary.median() > 500_000, "salaries must be annual INR, not lakhs"
    assert salary.skew() > 0.5, "pay should be right-skewed, as Chapter 3 teaches"
    assert 0.1 < df["salary_imputed"].mean() < 0.35
