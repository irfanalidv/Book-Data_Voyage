"""Tests for talentlens.features.

Each feature group has its own test class. Tests focus on the
*contract* of each feature (idempotency, dtype, value range,
handling of edge cases), not on the specific numeric values
produced - those are validated by the chapter executable's
hypothesis-result table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from talentlens.features import (
    FEATURE_GROUPS,
    HIGH_SIGNAL_SKILLS,
    SENIORITY_KEYWORDS,
    _add_salary_features,
    _add_skill_features,
    _add_title_features,
    engineer_features,
    select_features,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def sample_titles_df() -> pd.DataFrame:
    """Five rows covering the seniority spectrum + edge cases."""
    return pd.DataFrame(
        {
            "title": [
                "Senior Data Scientist",
                "Junior ML Engineer (Remote)",
                "Principal Engineer, AI Platform",
                "Data Analyst",
                "",
            ],
        }
    )


# ---------------------------------------------------------------------
# Title features
# ---------------------------------------------------------------------


class TestAddTitleFeatures:
    """Contract tests for _add_title_features."""

    def test_returns_dataframe_with_new_columns(self, sample_titles_df):
        out = _add_title_features(sample_titles_df)
        for col in FEATURE_GROUPS["title"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_does_not_drop_existing_columns(self, sample_titles_df):
        out = _add_title_features(sample_titles_df)
        assert "title" in out.columns
        assert len(out) == len(sample_titles_df)

    def test_is_idempotent(self, sample_titles_df):
        once = _add_title_features(sample_titles_df)
        twice = _add_title_features(once)
        for col in FEATURE_GROUPS["title"]:
            assert (once[col] == twice[col]).all()

    def test_does_not_mutate_input(self, sample_titles_df):
        before = sample_titles_df.copy()
        _ = _add_title_features(sample_titles_df)
        pd.testing.assert_frame_equal(sample_titles_df, before)

    def test_seniority_level_dtype_is_int8(self, sample_titles_df):
        out = _add_title_features(sample_titles_df)
        assert out["seniority_level"].dtype == "int8"

    def test_seniority_level_range(self, sample_titles_df):
        out = _add_title_features(sample_titles_df)
        assert out["seniority_level"].min() >= 0
        assert out["seniority_level"].max() <= 6

    def test_seniority_specific_values(self, sample_titles_df):
        out = _add_title_features(sample_titles_df)
        expected = [4, 1, 6, 3, 3]
        assert list(out["seniority_level"]) == expected

    def test_remote_in_title_dtype_is_int8(self, sample_titles_df):
        out = _add_title_features(sample_titles_df)
        assert out["is_remote_in_title"].dtype == "int8"

    def test_remote_in_title_specific_values(self, sample_titles_df):
        out = _add_title_features(sample_titles_df)
        expected = [0, 1, 0, 0, 0]
        assert list(out["is_remote_in_title"]) == expected

    def test_handles_nan_title(self):
        df = pd.DataFrame({"title": [None, float("nan"), "Senior"]})
        out = _add_title_features(df)
        assert list(out["seniority_level"]) == [3, 3, 4]

    def test_senior_staff_resolves_to_staff_not_senior(self):
        df = pd.DataFrame({"title": ["Senior Staff Engineer"]})
        out = _add_title_features(df)
        assert out["seniority_level"].iloc[0] == 5


class TestSeniorityKeywords:
    """Confidence checks on the keyword table itself."""

    def test_all_levels_in_range(self):
        for _, level in SENIORITY_KEYWORDS:
            assert 0 <= level <= 6

    def test_keywords_are_lowercase(self):
        for keyword, _ in SENIORITY_KEYWORDS:
            assert keyword == keyword.lower()

    def test_specific_keywords_present(self):
        keywords = {kw for kw, _ in SENIORITY_KEYWORDS}
        for required in ("junior", "senior", "principal", "intern"):
            assert required in keywords


@pytest.fixture
def sample_skills_df() -> pd.DataFrame:
    """Five rows covering common skill-extraction edge cases."""
    return pd.DataFrame(
        {
            "skills_normalised": [
                "Python|SQL|AWS|PyTorch",
                "python|sql|tableau",
                "Java|Scala|Hadoop",
                "",
                None,
            ],
        }
    )


# ---------------------------------------------------------------------
# Skill features
# ---------------------------------------------------------------------


class TestAddSkillFeatures:
    """Contract tests for _add_skill_features."""

    def test_returns_dataframe_with_new_columns(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        for col in FEATURE_GROUPS["skills"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_does_not_drop_existing_columns(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        assert "skills_normalised" in out.columns
        assert len(out) == len(sample_skills_df)

    def test_is_idempotent(self, sample_skills_df):
        once = _add_skill_features(sample_skills_df)
        twice = _add_skill_features(once)
        for col in FEATURE_GROUPS["skills"]:
            assert (once[col] == twice[col]).all()

    def test_does_not_mutate_input(self, sample_skills_df):
        before = sample_skills_df.copy()
        _ = _add_skill_features(sample_skills_df)
        pd.testing.assert_frame_equal(sample_skills_df, before)

    def test_skill_count_dtype_is_int16(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        assert out["skill_count"].dtype == "int16"

    def test_indicator_dtypes_are_int8(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        for skill in HIGH_SIGNAL_SKILLS:
            assert out[f"has_{skill}"].dtype == "int8"

    def test_skill_count_specific_values(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        assert list(out["skill_count"]) == [4, 3, 3, 0, 0]

    def test_case_insensitive_matching(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        assert out["has_python"].iloc[0] == 1
        assert out["has_python"].iloc[1] == 1

    def test_no_match_produces_zero_indicators(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        for skill in HIGH_SIGNAL_SKILLS:
            assert out[f"has_{skill}"].iloc[2] == 0

    def test_empty_string_skills_produces_zero_count(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        assert out["skill_count"].iloc[3] == 0

    def test_handles_none_skills(self, sample_skills_df):
        out = _add_skill_features(sample_skills_df)
        assert out["skill_count"].iloc[4] == 0
        for skill in HIGH_SIGNAL_SKILLS:
            assert out[f"has_{skill}"].iloc[4] == 0

    def test_extra_whitespace_in_skills_handled(self):
        df = pd.DataFrame({"skills_normalised": ["  Python | SQL  | AWS  "]})
        out = _add_skill_features(df)
        assert out["has_python"].iloc[0] == 1
        assert out["has_sql"].iloc[0] == 1
        assert out["has_aws"].iloc[0] == 1
        assert out["skill_count"].iloc[0] == 3

    def test_no_false_positive_on_substring(self):
        df = pd.DataFrame({"skills_normalised": ["TensorFlow Lite"]})
        out = _add_skill_features(df)
        assert out["has_tensorflow"].iloc[0] == 0


class TestHighSignalSkills:
    """Confidence checks on the canonical skills list."""

    def test_count_matches_feature_groups(self):
        assert len(HIGH_SIGNAL_SKILLS) == len(
            [c for c in FEATURE_GROUPS["skills"] if c.startswith("has_")]
        )

    def test_all_lowercase(self):
        for skill in HIGH_SIGNAL_SKILLS:
            assert skill == skill.lower()

    def test_no_duplicates(self):
        assert len(HIGH_SIGNAL_SKILLS) == len(set(HIGH_SIGNAL_SKILLS))

    def test_column_names_match_skills(self):
        for skill in HIGH_SIGNAL_SKILLS:
            assert f"has_{skill}" in FEATURE_GROUPS["skills"]


@pytest.fixture
def sample_salary_df() -> pd.DataFrame:
    """Five rows exercising salary edge cases."""
    return pd.DataFrame(
        {
            "salary_min": [1_800_000, 600_000, 4_000_000, None, 1_200_000],
            "salary_annual_inr": [2_100_000, 750_000, 5_000_000, None, 1_500_000],
            "skill_count": [3, 1, 5, 2, 0],
            "city": ["Bangalore", "Bangalore", "Mumbai", "Mumbai", "Delhi"],
        }
    )


# ---------------------------------------------------------------------
# Salary features
# ---------------------------------------------------------------------


class TestAddSalaryFeatures:
    """Contract tests for _add_salary_features."""

    def test_returns_dataframe_with_new_columns(self, sample_salary_df):
        out = _add_salary_features(sample_salary_df)
        for col in FEATURE_GROUPS["salary"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_does_not_drop_existing_columns(self, sample_salary_df):
        out = _add_salary_features(sample_salary_df)
        for col in ["salary_min", "salary_annual_inr", "skill_count", "city"]:
            assert col in out.columns
        assert len(out) == len(sample_salary_df)

    def test_is_idempotent(self, sample_salary_df):
        once = _add_salary_features(sample_salary_df)
        twice = _add_salary_features(once)
        for col in FEATURE_GROUPS["salary"]:
            pd.testing.assert_series_equal(once[col], twice[col])

    def test_does_not_mutate_input(self, sample_salary_df):
        before = sample_salary_df.copy()
        _ = _add_salary_features(sample_salary_df)
        pd.testing.assert_frame_equal(sample_salary_df, before)

    def test_log_salary_annual_inr_handles_nan(self, sample_salary_df):
        out = _add_salary_features(sample_salary_df)
        assert pd.isna(out["log_salary_annual_inr"].iloc[3])

    def test_log_salary_annual_inr_specific_values(self, sample_salary_df):
        import numpy as np

        out = _add_salary_features(sample_salary_df)
        assert abs(out["log_salary_annual_inr"].iloc[0] - np.log1p(2_100_000)) < 1e-6

    def test_salary_per_skill_handles_zero_skills(self, sample_salary_df):
        out = _add_salary_features(sample_salary_df)
        assert out["salary_per_skill"].iloc[4] == 1_500_000

    def test_salary_per_skill_specific_value(self, sample_salary_df):
        out = _add_salary_features(sample_salary_df)
        assert abs(out["salary_per_skill"].iloc[0] - 700_000) < 1

    def test_within_city_zscore_uses_correct_grouping(self, sample_salary_df):
        out = _add_salary_features(sample_salary_df)
        assert out["salary_in_band_for_city"].iloc[0] > 0
        assert out["salary_in_band_for_city"].iloc[1] < 0

    def test_within_city_zscore_falls_back_to_zero_for_single_city(self, sample_salary_df):
        out = _add_salary_features(sample_salary_df)
        assert out["salary_in_band_for_city"].iloc[4] == 0

    def test_log_salary_annual_inr_dtype_is_float(self, sample_salary_df):
        out = _add_salary_features(sample_salary_df)
        assert out["log_salary_annual_inr"].dtype == float


# ---------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------


class TestEngineerFeatures:
    """Integration tests for the public engineer_features API."""

    @pytest.fixture
    def sample_full_df(self) -> pd.DataFrame:
        """A minimal DataFrame matching the ch06 schema enough to
        run engineer_features end-to-end."""
        return pd.DataFrame(
            {
                "title": ["Senior Data Scientist", "Junior ML Engineer"],
                "skills_normalised": ["Python|SQL|AWS", "Python|PyTorch"],
                "salary_min": [1_800_000, 600_000],
                "salary_annual_inr": [2_100_000, 750_000],
                "city": ["Bangalore", "Mumbai"],
            }
        )

    def test_adds_all_feature_group_columns(self, sample_full_df):
        out = engineer_features(sample_full_df)
        for group_cols in FEATURE_GROUPS.values():
            for col in group_cols:
                assert col in out.columns, f"Missing column: {col}"

    def test_is_idempotent(self, sample_full_df):
        once = engineer_features(sample_full_df)
        twice = engineer_features(once)
        for group_cols in FEATURE_GROUPS.values():
            for col in group_cols:
                if once[col].dtype.kind == "f":
                    pd.testing.assert_series_equal(once[col], twice[col])
                else:
                    assert (once[col] == twice[col]).all()

    def test_does_not_mutate_input(self, sample_full_df):
        before = sample_full_df.copy()
        _ = engineer_features(sample_full_df)
        pd.testing.assert_frame_equal(sample_full_df, before)

    def test_output_row_count_matches_input(self, sample_full_df):
        out = engineer_features(sample_full_df)
        assert len(out) == len(sample_full_df)


class TestSelectFeatures:
    """Contract tests for select_features across all three methods."""

    @pytest.fixture
    def synthetic_classification_data(self):
        """Synthetic data with one strongly-informative feature
        and several noise features. Every method should rank the
        informative feature highly."""
        import numpy as np

        rng = np.random.default_rng(0)
        n = 200
        real = rng.normal(0, 1, n)
        X = pd.DataFrame(
            {
                "real_signal": real,
                "noise_1": rng.normal(0, 1, n),
                "noise_2": rng.normal(0, 1, n),
                "noise_3": rng.normal(0, 1, n),
                "constant": np.zeros(n),
            }
        )
        y = pd.Series((real > 0).astype(int))
        return X, y

    @pytest.mark.parametrize("method", ["mutual_info", "rfe", "l1"])
    def test_returns_at_most_k_features(self, synthetic_classification_data, method):
        X, y = synthetic_classification_data
        selected = select_features(X, y, method=method, k=3)
        assert len(selected) <= 3

    @pytest.mark.parametrize("method", ["mutual_info", "rfe", "l1"])
    def test_selects_informative_feature(self, synthetic_classification_data, method):
        X, y = synthetic_classification_data
        selected = select_features(X, y, method=method, k=2)
        assert "real_signal" in selected, (
            f"{method} failed to select the only informative feature " f"out of 5; got {selected}"
        )

    def test_raises_on_unknown_method(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        with pytest.raises(ValueError, match="Unknown method"):
            select_features(X, y, method="nonexistent_method", k=2)

    @pytest.mark.parametrize("method", ["mutual_info", "rfe", "l1"])
    def test_does_not_select_constant_feature(self, synthetic_classification_data, method):
        X, y = synthetic_classification_data
        selected = select_features(X, y, method=method, k=4)
        assert (
            "constant" not in selected[:2]
        ), f"{method} selected the zero-variance feature in top-2: {selected}"
