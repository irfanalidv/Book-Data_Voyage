"""Feature engineering for the TalentLens role classifier.

This module is consumed by:
  - Chapter 10 (the chapter that built it)
  - Chapter 11 (clustering)
  - Chapter 16 (RAG, optionally for ranking features)
  - Chapter 19 (FastAPI, at inference time)
  - Chapter 22 (talentlens-core PyPI package, via re-export)

The API surface is intentionally small and stable. Adding a new
feature group does not change the public functions - it adds an
entry to FEATURE_GROUPS and a private builder function.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# Canonical TalentLens role names. The scrub function below removes
# these from text features to prevent label leakage in any chapter
# where the labels are derived from title patterns (Chapter 6) and
# the text features include the description (Chapters 9 and 10).
# On real scraped data this is effectively a no-op; on the bundled
# demo data, every description embeds the role name verbatim and
# without the scrub the classifier learns the regex, not the task.
# See Chapter 9's "Common mistakes" section for the original story.
_CANONICAL_ROLE_NAMES: tuple[str, ...] = (
    "AI Engineer",
    "ML Engineer",
    "Machine Learning Engineer",
    "Data Scientist",
    "Data Engineer",
    "Data Analyst",
)


def scrub_role_phrases_from_text(text: str) -> str:
    """Remove canonical TalentLens role names from a text string.

    Used by Chapters 9 and 10 to prevent the label-in-features
    leakage that arises when role labels are derived from title
    patterns and text features include the description. On real
    data this is effectively a no-op (real postings rarely embed
    the literal role name in the description); on the bundled
    demo data it's load-bearing.

    Matching is case-insensitive. Longer phrases are scrubbed first
    so "Machine Learning Engineer" is removed before "ML Engineer"
    and "Engineer" patterns inside other phrases.

    Args:
        text: Arbitrary text string (a description, a title-plus-body
            concatenation, etc.).

    Returns:
        The same string with all canonical role names removed and
        collapsed whitespace.
    """
    import re

    if not isinstance(text, str):
        return ""
    result = text
    for name in sorted(_CANONICAL_ROLE_NAMES, key=len, reverse=True):
        result = re.sub(re.escape(name), " ", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()
    return result


# Seniority keywords mapped to an ordinal scale 0–6.
# Order matters: longer / more specific keywords are checked first
# so "senior staff" matches "staff" (5), not "senior" (3).
# The scale is ordinal, not categorical - chapter prose explains
# why a regression-style integer beats one-hot encoding here.
SENIORITY_KEYWORDS: list[tuple[str, int]] = [
    ("intern", 0),
    ("junior", 1),
    ("associate", 2),
    ("entry", 1),
    ("mid", 3),
    ("senior staff", 5),
    ("senior", 4),
    ("staff", 5),
    ("principal", 6),
    ("lead", 5),
    ("head of", 6),
    ("director", 6),
]


# The ten highest-signal canonical skills for binary indicator
# features. These are NOT the most frequent skills overall - they
# are the skills with the highest mutual information against
# role_category in the bundled dataset. Chapter prose explains
# the selection: a 200-skill one-hot encoding is mostly noise; ten
# well-chosen indicators carry most of the discriminative signal.
#
# If you change this list, also change FEATURE_GROUPS["skills"]
# below so the column names match.
HIGH_SIGNAL_SKILLS: list[str] = [
    "python",
    "sql",
    "aws",
    "pytorch",
    "llm",
    "spark",
    "kubernetes",
    "docker",
    "tableau",
    "tensorflow",
]


# Chapter 6 writes canonical names from talentlens.skills ("Cloud (AWS)",
# "LLMs"); scraped skill tags use short forms ("aws", "llm"). An indicator
# fires on either spelling, so a naming mismatch cannot silently turn a
# feature into a constant.
_SKILL_TOKEN_SYNONYMS: dict[str, frozenset[str]] = {
    "aws": frozenset({"aws", "cloud (aws)"}),
    "llm": frozenset({"llm", "llms"}),
}


# Below this MI score, sklearn's mutual_info_classif estimator
# produces noise indistinguishable from zero on truly constant
# features. Chapter 10's diagnostic reproduced this on the bundled
# dataset: four features known to be constant all received the
# same nonzero score of 0.043427. The threshold here is set
# conservatively at roughly 1/4 of that observed noise level.
_MI_NOISE_THRESHOLD: float = 0.01


# Names of features by group. Adding a new group (e.g. "embedding"
# for the merged Ch11) means adding the key here and the matching
# private builder. Public callers see only the merged DataFrame.
FEATURE_GROUPS: dict[str, list[str]] = {
    "title": [
        "seniority_level",
        "is_remote_in_title",
    ],
    "skills": [
        "skill_count",
        "has_python",
        "has_sql",
        "has_aws",
        "has_pytorch",
        "has_llm",
        "has_spark",
        "has_kubernetes",
        "has_docker",
        "has_tableau",
        "has_tensorflow",
    ],
    "salary": [
        "log_salary_annual_inr",
        "salary_per_skill",
        "salary_in_band_for_city",
    ],
}


SelectionMethod = Literal["mutual_info", "rfe", "l1"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered feature columns to a TalentLens DataFrame.

    Idempotent (running twice produces the same columns).
    Deterministic (no randomness without a seed argument).

    Builder order matters: salary features depend on skill_count
    from the skills builder. Adding new feature groups means
    adding both the builder and the corresponding entry in
    FEATURE_GROUPS.

    Args:
        df: DataFrame matching the schema in tests/test_schema_contract.py.

    Returns:
        Copy of df with all FEATURE_GROUPS columns added.
    """
    df = df.copy()
    df = _add_title_features(df)
    df = _add_skill_features(df)
    df = _add_salary_features(df)
    return df


def select_features(
    X: pd.DataFrame,  # noqa: N803 - ML convention for feature matrix
    y: pd.Series,
    method: SelectionMethod,
    k: int = 15,
) -> list[str]:
    """Return the top-k feature names selected by ``method``.

    Three methods, each representing a different family:
      - "mutual_info": filter method. Fast, model-agnostic. Ranks
        features by mutual information with the target. Captures
        non-linear univariate relationships but ignores feature
        interactions.
      - "rfe": wrapper method. Recursive Feature Elimination with
        cross-validation over a logistic-regression base estimator.
        Captures interactions but expensive and tied to the base
        model's view.
      - "l1": embedded method. L1-penalised logistic regression;
        features with non-zero coefficients are selected. Fast and
        principled, but the selection depends on the L1 strength
        (C parameter); we pick the C that yields roughly k features.

    All three operate on numeric feature columns only. The caller
    is expected to have already produced numeric features via
    engineer_features.

    Args:
        X: Feature DataFrame; non-numeric columns are dropped.
        y: Target Series.
        method: Selection method as named above.
        k: Number of features to return. If fewer features have
            non-zero importance, returns all of them.

    Returns:
        List of selected feature column names, length up to k.
    """
    from sklearn.feature_selection import RFECV, mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    X_num = X[numeric_cols].fillna(X[numeric_cols].median())  # noqa: N806 - matrix slice from X

    if method == "mutual_info":
        varying_cols = [
            c for c in numeric_cols if X_num[c].nunique(dropna=False) > 1
        ]
        mi = mutual_info_classif(X_num[varying_cols], y, random_state=42)
        ranked = sorted(
            zip(varying_cols, mi, strict=False), key=lambda kv: kv[1], reverse=True
        )
        return [name for name, score in ranked if score > _MI_NOISE_THRESHOLD][:k]

    if method == "rfe":
        rfecv = RFECV(
            estimator=LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight="balanced",
                random_state=42,
            ),
            step=1,
            cv=3,
            scoring="f1_macro",
            min_features_to_select=1,
        )
        X_scaled = StandardScaler().fit_transform(X_num)  # noqa: N806 - scaled feature matrix
        rfecv.fit(X_scaled, y)
        selected = [c for c, m in zip(numeric_cols, rfecv.support_, strict=False) if m]
        return selected[:k]

    if method == "l1":
        X_scaled = StandardScaler().fit_transform(X_num)  # noqa: N806 - scaled feature matrix
        nonzero_mask = None
        for C in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0):  # noqa: N806 - sklearn regularisation strength
            clf = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C,
                max_iter=2000,
                random_state=42,
            )
            clf.fit(X_scaled, y)
            nonzero_mask = (clf.coef_ != 0).any(axis=0)
            n_selected = int(nonzero_mask.sum())
            if n_selected >= k:
                selected = [c for c, m in zip(numeric_cols, nonzero_mask, strict=False) if m]
                return selected[:k]
        if nonzero_mask is not None:
            return [c for c, m in zip(numeric_cols, nonzero_mask, strict=False) if m]
        return []

    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------
# Private feature builders (one per FEATURE_GROUPS key)
# ---------------------------------------------------------------------


def _add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured features from the job title.

    Adds two columns:
      - ``seniority_level``: ordinal 0–6 derived from keyword match
        on the title. 3 (the "mid" default) is used when no keyword
        matches; this is the modal seniority level.
      - ``is_remote_in_title``: 1 if the title contains "remote" or
        "remote-first" as a word, else 0. This is separate from
        the structured ``is_remote`` column because some postings
        advertise remote in the title even when the structured
        field is missing (a real pattern in the bundled data).

    Args:
        df: DataFrame with a ``title`` column.

    Returns:
        Copy of df with the two new columns appended.
    """
    df = df.copy()
    titles = df["title"].fillna("").str.lower()

    def _seniority_of(title: str) -> int:
        for keyword, level in SENIORITY_KEYWORDS:
            if keyword in title:
                return level
        return 3  # modal default - "mid"

    df["seniority_level"] = titles.apply(_seniority_of).astype("int8")

    df["is_remote_in_title"] = (
        titles.str.contains(r"\bremote\b", regex=True, na=False).astype("int8")
    )
    return df


def _add_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured features from the skills_normalised column.

    Adds:
      - ``skill_count``: integer count of pipe-separated skill
        tokens. Proxy for posting comprehensiveness; longer skill
        lists weakly correlate with senior and engineering roles.
      - ``has_<skill>`` (10 columns): binary indicators for the
        canonical high-signal skills in ``HIGH_SIGNAL_SKILLS``.
        Matched case-insensitively against the pipe-split tokens.

    Missing or empty ``skills_normalised`` produces ``skill_count=0``
    and all indicators set to 0.

    Args:
        df: DataFrame with a ``skills_normalised`` column
            (pipe-separated string, as written by Chapter 6).

    Returns:
        Copy of df with the count and indicator columns appended.
    """
    df = df.copy()
    skills_str = df["skills_normalised"].fillna("").astype(str).str.lower()

    skill_lists = skills_str.str.split("|").apply(
        lambda lst: [s.strip() for s in lst if s.strip()]
    )
    df["skill_count"] = skill_lists.apply(len).astype("int16")

    skill_sets = skill_lists.apply(set)
    for skill in HIGH_SIGNAL_SKILLS:
        col = f"has_{skill}"
        accepted = _SKILL_TOKEN_SYNONYMS.get(skill, frozenset({skill}))
        df[col] = skill_sets.apply(
            lambda s, accepted=accepted: int(bool(s & accepted))
        ).astype("int8")

    return df


def _add_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer numeric features from salary columns.

    Adds:
      - ``log_salary_annual_inr``: ``log1p`` of ``salary_annual_inr``
        (the derived analysis column from Chapter 6;
        ``salary_min``/``salary_max`` are NaN-allowed audit columns
        with much sparser coverage on the bundled dataset). The raw
        salary distribution is heavily right-skewed; log1p compresses
        the tail so a linear model can use the feature without
        high-salary outliers dominating.
      - ``salary_per_skill``: ``salary_annual_inr / max(skill_count, 1)``.
        Hypothesis: specialised roles (few skills, high salary)
        pay more per skill than generalists. The ``max(..., 1)``
        guards against division by zero for postings with no
        skills listed.
      - ``salary_in_band_for_city``: z-score of ``salary_annual_inr``
        within ``city``. Hypothesis: normalising out cost-of-living
        improves cross-city comparison. NaN where ``salary_annual_inr``
        is NaN; falls back to 0 when a city has only one posting
        (no within-city variance to z-score against).

    Requires ``skill_count`` to exist on the DataFrame - typically
    means ``_add_skill_features`` was called first. The public
    ``engineer_features`` does this ordering automatically.

    Args:
        df: DataFrame with ``salary_annual_inr``, ``skill_count``,
            and ``city`` columns.

    Returns:
        Copy of df with the three salary features appended.
    """
    df = df.copy()

    df["log_salary_annual_inr"] = np.log1p(df["salary_annual_inr"])

    denom = df["skill_count"].clip(lower=1)
    df["salary_per_skill"] = df["salary_annual_inr"] / denom

    city_grouped = df.groupby("city")["salary_annual_inr"]
    df["salary_in_band_for_city"] = (
        (df["salary_annual_inr"] - city_grouped.transform("mean"))
        / city_grouped.transform("std")
    )
    df["salary_in_band_for_city"] = df["salary_in_band_for_city"].fillna(0)

    return df
