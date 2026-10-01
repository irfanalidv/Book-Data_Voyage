"""Pin the jobs_clean.csv schema so downstream chapters can rely on it.

Every chapter from 7 onwards reads jobs_clean.csv. Without this test,
adding or renaming a column in Chapter 6 silently breaks any chapter
that doesn't have its own per-chapter test (currently 10, 11, 12, 13,
14, 15, 18). This test runs the cleaning pipeline against demo data and
asserts the contract is unchanged.
"""

from __future__ import annotations

import importlib

import pytest

ch06 = importlib.import_module("book.ch06.ch06_data_cleaning_preprocessing")

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "job_id",
        "title",
        "company",
        "description",
        "city",
        "country",
        "url",
        "source",
        "fingerprint",
        "salary_min",
        "salary_max",
        "salary_disclosed",
        "salary_imputed",
        "salary_annual_inr",
        "salary_band",
        "skills_normalised",
        "is_remote",
        "role_label",
        "role_category",
    }
)


def _build_cleaned_demo_df():
    cfg = ch06.Config()
    raw = ch06._generate_demo_raw()
    df = ch06.drop_invalid_rows(raw, cfg)
    df = ch06.clean_titles(df)
    df = ch06.derive_role_category(df)
    df = ch06.impute_salary(df, cfg)
    df = ch06.extract_skills(df, cfg)
    df = ch06.normalise_remote_flag(df)
    df = ch06.add_salary_band(df)
    df = ch06.derive_salary_annual_inr(df)
    df = ch06.final_dedup(df)
    return df


def test_jobs_clean_has_required_columns():
    df = _build_cleaned_demo_df()
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, (
        f"jobs_clean schema is missing required columns: {sorted(missing)}. "
        "Either add them in Chapter 6, or coordinate the removal with every "
        "downstream chapter that reads them."
    )


def test_jobs_clean_role_label_equals_role_category():
    df = _build_cleaned_demo_df()
    assert (df["role_label"] == df["role_category"]).all()


def test_jobs_clean_salary_annual_inr_is_between_min_and_max_when_both_present():
    df = _build_cleaned_demo_df()
    mask = df["salary_min"].notna() & df["salary_max"].notna()
    if mask.sum() == 0:
        pytest.skip("Demo data produced no rows with both bounds disclosed.")
    sub = df.loc[mask]
    assert (sub["salary_annual_inr"] >= sub["salary_min"]).all()
    assert (sub["salary_annual_inr"] <= sub["salary_max"]).all()


def test_jobs_clean_role_category_values_are_in_canonical_set():
    df = _build_cleaned_demo_df()
    allowed = {
        "AI Engineer",
        "ML Engineer",
        "Data Scientist",
        "Data Engineer",
        "Data Analyst",
        "Other",
    }
    bad = set(df["role_category"].unique()) - allowed
    assert not bad, f"Unexpected role_category values: {bad}"


ch09 = importlib.import_module("book.ch09.ch09_supervised_learning")


def test_ch09_feature_text_does_not_contain_title():
    """The leakage regression.

    Chapter 6's ``role_category`` is derived from ``title``. Chapter 9's
    feature text must therefore not include ``title``, or we reintroduce
    the leak that produced F1=1.000 in the first draft of ch09.
    """
    import pandas as pd

    row = pd.Series(
        {
            "title": "ZZZ_TITLE_SENTINEL_ZZZ",
            "description": "describe the role here",
            "skills_normalised": "Python|SQL",
        }
    )
    text = ch09.build_feature_text(row)
    assert "ZZZ_TITLE_SENTINEL_ZZZ" not in text, (
        "ch09.build_feature_text included the title — this is the bug "
        "documented in ch09's 'Common mistakes' section. If a future "
        "labelling change makes training on title safe, update both this "
        "test and that mistake-block at the same time, deliberately."
    )
