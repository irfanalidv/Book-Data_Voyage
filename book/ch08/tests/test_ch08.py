"""Chapter 8 tests - inference helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))

from ch08_inference import (  # noqa: E402
    _generate_synthetic_jobs,
    remote_vs_onsite_by_seniority,
    remote_vs_onsite_test,
    salary_disclosure_chi_squared,
    simulate_null_p_values,
)


def test_remote_vs_onsite_test_returns_pvalue_and_effect_size() -> None:
    df = _generate_synthetic_jobs(400)
    result = remote_vs_onsite_test(df)
    assert "p_value" in result
    assert "effect_median_diff_lpa" in result


def test_seniority_controlled_test_runs_per_bin() -> None:
    df = _generate_synthetic_jobs(800)
    results = remote_vs_onsite_by_seniority(df)
    assert len(results) >= 2
    for band in results:
        assert "p_value" in results[band]


def test_chi_squared_on_2x2_contingency_handles_balanced_table() -> None:
    df = pd.DataFrame(
        {
            "is_remote": [True, True, False, False] * 25,
            "salary_disclosed": [True, False, True, False] * 25,
        }
    )
    result = salary_disclosure_chi_squared(df)
    assert result["p_value"] > 0.05


def test_p_value_distribution_under_null_is_approximately_uniform() -> None:
    pvals = simulate_null_p_values(200)
    frac = float((pvals < 0.05).mean())
    assert 0.02 <= frac <= 0.10
