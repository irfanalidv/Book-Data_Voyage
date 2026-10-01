"""
Tests for Chapter 7: Exploratory Data Analysis (TalentLens EDA).

Run from repository root:

    pytest tests/test_ch07.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch07"))

from ch07_exploratory_data_analysis import (  # noqa: E402
    Config,
    _generate_demo_data,
    analyse_role_comparison,
    analyse_salary_distribution,
    analyse_skill_frequency,
)


@pytest.fixture
def demo_df():
    return _generate_demo_data()


@pytest.fixture
def cfg():
    return Config()


class TestDemoData:
    def test_returns_dataframe(self, demo_df):
        assert isinstance(demo_df, pd.DataFrame)

    def test_has_required_columns(self, demo_df, cfg):
        required = [cfg.salary_col, cfg.role_col, cfg.skills_col, cfg.remote_col]
        for col in required:
            assert col in demo_df.columns, f"Missing column: {col}"

    def test_reasonable_row_count(self, demo_df):
        assert len(demo_df) >= 1_000, "Demo data should have at least 1,000 rows"

    def test_salary_has_some_nulls(self, demo_df, cfg):
        null_pct = demo_df[cfg.salary_col].isna().mean()
        assert 0.01 < null_pct < 0.10, f"Expected 1–10% null salaries, got {null_pct:.2%}"

    def test_salary_values_reasonable(self, demo_df, cfg):
        salary = demo_df[cfg.salary_col].dropna()
        assert salary.min() > 100_000, "Salaries below ₹1L seem wrong"
        assert salary.max() < 50_000_000, "Salaries above ₹5Cr seem wrong"


class TestSalaryDistribution:
    def test_returns_dict_with_required_keys(self, demo_df, cfg):
        result = analyse_salary_distribution(demo_df, cfg)
        required_keys = ["count", "missing_pct", "mean", "median", "std", "p25", "p75", "skew"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_mean_median_relationship(self, demo_df, cfg):
        result = analyse_salary_distribution(demo_df, cfg)
        assert (
            result["mean"] >= result["median"] * 0.9
        ), "Expected mean >= median for typical salary distribution"

    def test_handles_all_nulls_gracefully(self, cfg):
        df = pd.DataFrame({cfg.salary_col: [np.nan] * 100})
        try:
            result = analyse_salary_distribution(df, cfg)
            assert result["count"] == 0
        except Exception as e:
            pytest.fail(f"Should handle all-null salary column, got: {e}")

    def test_missing_pct_within_bounds(self, demo_df, cfg):
        result = analyse_salary_distribution(demo_df, cfg)
        assert 0 <= result["missing_pct"] <= 100


class TestSkillFrequency:
    def test_returns_series(self, demo_df, cfg):
        result = analyse_skill_frequency(demo_df, cfg)
        assert isinstance(result, pd.Series)

    def test_sorted_descending(self, demo_df, cfg):
        result = analyse_skill_frequency(demo_df, cfg)
        assert all(
            result.iloc[i] >= result.iloc[i + 1] for i in range(len(result) - 1)
        ), "Skills should be sorted by frequency descending"

    def test_frequencies_are_proportions(self, demo_df, cfg):
        result = analyse_skill_frequency(demo_df, cfg)
        assert all(0 <= v <= 1 for v in result.values), "Frequencies should be 0–1 proportions"

    def test_top_n_skills_respected(self, demo_df, cfg):
        result = analyse_skill_frequency(demo_df, cfg)
        assert len(result) <= cfg.top_n_skills

    def test_python_in_top_skills(self, demo_df, cfg):
        result = analyse_skill_frequency(demo_df, cfg)
        assert "Python" in result.index, "Python should be in top skills"


class TestRoleComparison:
    def test_returns_dataframe(self, demo_df, cfg):
        result = analyse_role_comparison(demo_df, cfg)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, demo_df, cfg):
        result = analyse_role_comparison(demo_df, cfg)
        assert "median" in result.columns
        assert "count" in result.columns

    def test_sorted_by_median_descending(self, demo_df, cfg):
        result = analyse_role_comparison(demo_df, cfg)
        medians = result["median"].values
        assert all(
            medians[i] >= medians[i + 1] for i in range(len(medians) - 1)
        ), "Roles should be sorted by median salary descending"

    def test_all_counts_positive(self, demo_df, cfg):
        result = analyse_role_comparison(demo_df, cfg)
        assert all(result["count"] > 0), "All role counts should be positive"
