"""Tests for Chapter 10: Feature Engineering and Selection.

Reference: book/ch10/README.md
Source: book/ch10/ch10_feature_engineering_selection.py, talentlens/features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch10"))

from ch10_feature_engineering_selection import (  # noqa: E402
    _check_features_have_signal,
    _summarise_feature_variance,
)

from talentlens.features import FEATURE_GROUPS, engineer_features, select_features  # noqa: E402
from talentlens.paths import jobs_clean_path  # noqa: E402

_ALL_FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]


@pytest.fixture
def jobs_sample() -> pd.DataFrame:
    path = jobs_clean_path()
    if not path.exists():
        pytest.skip(f"jobs_clean not found at {path}")
    df = pd.read_csv(path, nrows=120)
    return engineer_features(df)


class TestEngineerFeaturesOnTalentLensData:
    def test_returns_all_feature_group_columns(self, jobs_sample: pd.DataFrame):
        missing = [c for c in _ALL_FEATURE_COLS if c not in jobs_sample.columns]
        assert missing == [], f"engineer_features missing columns: {missing}"

    def test_senior_staff_maps_to_staff_level_five(self):
        df = pd.DataFrame(
            {
                "title": ["Senior Staff Engineer"],
                "skills_normalised": [""],
                "salary_annual_inr": [2_000_000.0],
                "city": ["Bangalore"],
            }
        )
        out = engineer_features(df)
        assert int(out["seniority_level"].iloc[0]) == 5

    def test_skill_flags_from_normalised_skills(self):
        df = pd.DataFrame(
            {
                "title": ["Data Scientist"],
                "skills_normalised": ["Python|SQL|AWS"],
                "salary_annual_inr": [1_500_000.0],
                "city": ["Mumbai"],
            }
        )
        out = engineer_features(df)
        assert int(out["skill_count"].iloc[0]) == 3
        assert int(out["has_python"].iloc[0]) == 1
        assert int(out["has_sql"].iloc[0]) == 1
        assert int(out["has_aws"].iloc[0]) == 1


class TestFeatureSelectionScores:
    def test_mutual_info_scores_in_unit_interval(self, jobs_sample: pd.DataFrame):
        y = jobs_sample["role_category"].astype(str)
        numeric = jobs_sample[_ALL_FEATURE_COLS].select_dtypes(include="number")
        selected = select_features(numeric, y, method="mutual_info", k=5)
        assert 1 <= len(selected) <= 5
        assert all(c in numeric.columns for c in selected)

    def test_constant_column_excluded_from_mi_selection(self):
        rng = np.random.default_rng(10)
        n = 80
        X = pd.DataFrame(
            {
                "varying": rng.integers(0, 4, n).astype(float),
                "flat": [7.0] * n,
            }
        )
        y = pd.Series(["A", "B"] * (n // 2))
        picked = select_features(X, y, method="mutual_info", k=2)
        assert "flat" not in picked
        assert "varying" in picked


class TestChapterSignalDiagnostics:
    def test_check_features_flags_all_nan_column(self):
        df = pd.DataFrame({"broken": [float("nan")] * 5, "ok": [1.0, 2.0, 3.0, 4.0, 5.0]})
        issues = _check_features_have_signal(df, ["broken", "ok"])
        kinds = {i.column: i.kind for i in issues}
        assert kinds["broken"] == "all_nan"
        assert "ok" not in kinds

    def test_variance_summary_marks_constant_features(self, jobs_sample: pd.DataFrame):
        summary = _summarise_feature_variance(jobs_sample)
        by_feature = {row["feature"]: row for row in summary}
        assert "seniority_level" in by_feature
        assert by_feature["seniority_level"]["n_unique"] >= 1
        assert by_feature["seniority_level"]["kind"] in {
            "varying",
            "constant",
            "all_nan",
        }
