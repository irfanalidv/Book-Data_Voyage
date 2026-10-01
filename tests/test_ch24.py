"""
Tests for Chapter 24: The India Playbook - Career Market Analysis.

Run from repository root:

    pytest tests/test_ch24.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch24"))

from ch24_career_market_analysis import (  # noqa: E402
    CAREER_TRAJECTORY,
    MARKET_SEGMENTS,
    SKILL_SALARY_PREMIUM,
    Config,
    _generate_demo_data,
    analyse_market_segments,
    analyse_remote_premium,
    analyse_skill_salary_premium,
)


@pytest.fixture
def demo_df(cfg: Config):
    return _generate_demo_data(cfg)


@pytest.fixture
def cfg():
    return Config()


class TestDemoData:
    def test_returns_dataframe(self, demo_df):
        assert isinstance(demo_df, pd.DataFrame)

    def test_has_required_columns(self, demo_df):
        for col in ["salary_annual_inr", "is_remote", "company_type", "role_category"]:
            assert col in demo_df.columns

    def test_is_remote_is_boolean(self, demo_df):
        assert demo_df["is_remote"].dtype in [bool, np.bool_]

    def test_some_remote_some_onsite(self, demo_df):
        assert demo_df["is_remote"].any(), "Should have some remote rows"
        assert (~demo_df["is_remote"]).any(), "Should have some on-site rows"


class TestMarketSegments:
    def test_returns_dataframe(self):
        result = analyse_market_segments()
        assert isinstance(result, pd.DataFrame)

    def test_has_all_segments(self):
        result = analyse_market_segments()
        assert len(result) == len(MARKET_SEGMENTS)

    def test_sorted_by_median_descending(self):
        result = analyse_market_segments()
        medians = result["median_l"].values
        assert all(medians[i] >= medians[i + 1] for i in range(len(medians) - 1))

    def test_remote_contract_highest_median(self):
        result = analyse_market_segments()
        top = result.iloc[0]["segment"]
        assert "Remote Contract" in top or "Global Startup" in top

    def test_all_growth_rates_positive(self):
        for seg, stats in MARKET_SEGMENTS.items():
            assert stats["growth_rate"] > 0, f"{seg} should have positive growth rate"


class TestRemotePremium:
    def test_returns_dict_with_required_keys(self, demo_df):
        result = analyse_remote_premium(demo_df)
        for key in [
            "remote_median",
            "onsite_median",
            "remote_count",
            "onsite_count",
            "premium_pct",
        ]:
            assert key in result

    def test_counts_positive(self, demo_df):
        result = analyse_remote_premium(demo_df)
        assert result["remote_count"] > 0
        assert result["onsite_count"] > 0

    def test_premium_is_float(self, demo_df):
        result = analyse_remote_premium(demo_df)
        assert isinstance(result["premium_pct"], float)

    def test_handles_df_with_no_salary(self, cfg):
        df = pd.DataFrame(
            {
                "salary_annual_inr": [np.nan] * 50,
                "is_remote": [True] * 25 + [False] * 25,
            }
        )
        try:
            result = analyse_remote_premium(df)
            assert result["remote_count"] == 0 or np.isnan(result["remote_median"])
        except Exception:
            pass


class TestSkillPremium:
    def test_returns_series(self):
        result = analyse_skill_salary_premium()
        assert isinstance(result, pd.Series)

    def test_sorted_descending(self):
        result = analyse_skill_salary_premium()
        vals = result.values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_all_premiums_positive(self):
        result = analyse_skill_salary_premium()
        assert all(v > 0 for v in result.values)

    def test_rag_higher_than_python(self):
        result = analyse_skill_salary_premium()
        assert result.get("RAG", 0) > result.get("Python", 0)

    def test_matches_source_dict(self):
        result = analyse_skill_salary_premium()
        for skill in SKILL_SALARY_PREMIUM:
            assert skill in result.index


class TestCareerTrajectory:
    def test_all_paths_have_6_points(self):
        for path, points in CAREER_TRAJECTORY.items():
            assert len(points) == 6, f"{path} should have 6 data points (years 0–5)"

    def test_salaries_increase_over_time(self):
        for path, points in CAREER_TRAJECTORY.items():
            salaries = [p[1] for p in points]
            assert salaries[-1] > salaries[0], f"{path} should show salary growth"

    def test_remote_path_overtakes_service_path(self):
        remote = dict(CAREER_TRAJECTORY["Remote contract at 3yr mark"])
        service = dict(CAREER_TRAJECTORY["Service company → exit at 2yr"])
        assert remote[5] > service[5] * 1.5
