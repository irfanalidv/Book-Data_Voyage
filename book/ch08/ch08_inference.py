#!/usr/bin/env python3
"""Chapter 8: statistical inference on TalentLens-shaped salary data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from talentlens.paths import BOOK_DIR, REPO_ROOT, jobs_clean_path

logger = logging.getLogger(__name__)
FIGURES_DIR = BOOK_DIR / "ch08" / "reports" / "figures"
RNG = np.random.default_rng(8)
# Seniority levels read from the job title. We stratify on these, not on
# salary_band: salary_band is computed from salary, and comparing salaries
# within salary bands removes any gap by construction (conditioning on the
# outcome). The title is set before pay is known, so it is a fair control.
SENIORITY_BINS = ["junior", "mid", "senior", "lead"]


def title_seniority(title: str) -> str:
    """Map a job title to junior / mid / senior / lead from its seniority words."""
    t = str(title).lower()
    if any(k in t for k in ("lead", "staff", "principal", "head of", "director")):
        return "lead"
    if "senior" in t or "sr." in t:
        return "senior"
    if any(k in t for k in ("junior", "intern", "entry", "graduate", "trainee")):
        return "junior"
    return "mid"


def load_or_generate_jobs(n: int = 2500) -> pd.DataFrame:
    """Prefer cleaned CSV from Chapter 6; otherwise synthetic fixture."""
    clean_path = jobs_clean_path()
    if clean_path.exists():
        logger.info("Loading %s", clean_path)
        df = pd.read_csv(clean_path)
        if "salary_min" in df.columns and "salary_max" in df.columns:
            df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
            df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")
            df["salary_min"] = df["salary_min"].fillna(df["salary_max"])
        if "is_remote" in df.columns:
            df["is_remote"] = (
                df["is_remote"].astype(str).str.lower().isin(("true", "1", "yes", "remote"))
            )
        if "role_label" not in df.columns and "title" in df.columns:
            df["role_label"] = df["title"]
        if "salary_disclosed" not in df.columns:
            df["salary_disclosed"] = df["salary_min"].notna()
        else:
            # Chapter 6 imputed hidden salaries from group medians. Those are
            # guesses; the salary tests use disclosed pay only.
            df["salary_disclosed"] = df["salary_disclosed"].astype(str).str.lower() == "true"
            df.loc[~df["salary_disclosed"], "salary_min"] = np.nan
        df["seniority"] = df["title"].map(title_seniority)
        if "salary_band" not in df.columns:
            df = _add_salary_band(df)
        needs_fixture = (
            df["salary_min"].notna().sum() < 30
            or df["seniority"].nunique() < 3
            or df["role_label"].nunique() < 3
        )
        if needs_fixture:
            logger.warning("Clean file insufficient for inference demo — using synthetic fixture")
            return _generate_synthetic_jobs(n)
        return df.head(n) if len(df) > n else df

    logger.info("Generating synthetic TalentLens-shaped data (n=%d)", n)
    return _generate_synthetic_jobs(n)


def _add_salary_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [0, 800_000, 1_500_000, 3_000_000, float("inf")]
    labels = ["junior", "mid", "senior", "lead_plus"]
    out["salary_band"] = pd.cut(
        out["salary_min"].fillna(out["salary_min"].median()),
        bins=bins,
        labels=labels,
        right=True,
    ).astype(str)
    return out


def _generate_synthetic_jobs(n: int) -> pd.DataFrame:
    """Synthetic jobs where remote work tracks seniority but carries no premium."""
    roles = ["AI Engineer", "ML Engineer", "Data Scientist", "Data Engineer"]
    base_by_level = {"junior": 900_000, "mid": 1_800_000, "senior": 3_000_000, "lead": 4_200_000}
    p_remote = {"junior": 0.15, "mid": 0.3, "senior": 0.5, "lead": 0.6}
    prefix = {"junior": "Junior ", "mid": "", "senior": "Senior ", "lead": "Lead "}
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        level = str(RNG.choice(SENIORITY_BINS, p=[0.25, 0.35, 0.28, 0.12]))
        is_remote = bool(RNG.random() < p_remote[level])
        disclosed = bool(RNG.random() < (0.75 if is_remote else 0.6))
        sal = int(base_by_level[level] * RNG.lognormal(0, 0.25))
        role = str(RNG.choice(roles, p=[0.2, 0.35, 0.3, 0.15]))
        rows.append(
            {
                "title": prefix[level] + role,
                "role_label": role,
                "seniority": level,
                "salary_min": float(sal) if disclosed else np.nan,
                "salary_max": float(sal * 1.2) if disclosed else np.nan,
                "salary_disclosed": disclosed,
                "is_remote": is_remote,
                "company": f"Co_{RNG.integers(0, 80)}",
            }
        )
    return pd.DataFrame(rows)


def _median_salary_lpa(series: pd.Series) -> float:
    return float(series.dropna().median() / 100_000)


def remote_vs_onsite_test(df: pd.DataFrame) -> dict[str, float]:
    """Mann-Whitney U: remote vs on-site median salary (INR annual)."""
    remote = df.loc[df["is_remote"], "salary_min"].dropna()
    onsite = df.loc[~df["is_remote"], "salary_min"].dropna()
    stat, pval = stats.mannwhitneyu(remote, onsite, alternative="greater")
    effect = _median_salary_lpa(remote) - _median_salary_lpa(onsite)
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "effect_median_diff_lpa": effect,
        "n_remote": float(len(remote)),
        "n_onsite": float(len(onsite)),
    }


def remote_vs_onsite_by_seniority(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Stratified Mann-Whitney within each title-seniority level - surfaces confounding."""
    results: dict[str, dict[str, float]] = {}
    for band in SENIORITY_BINS:
        sub = df[df["seniority"] == band]
        if sub["is_remote"].nunique() < 2:
            continue
        remote = sub.loc[sub["is_remote"], "salary_min"].dropna()
        onsite = sub.loc[~sub["is_remote"], "salary_min"].dropna()
        if len(remote) < 8 or len(onsite) < 8:
            continue
        stat, pval = stats.mannwhitneyu(remote, onsite, alternative="two-sided")
        results[band] = {
            "statistic": float(stat),
            "p_value": float(pval),
            "effect_median_diff_lpa": _median_salary_lpa(remote) - _median_salary_lpa(onsite),
            "n_remote": float(len(remote)),
            "n_onsite": float(len(onsite)),
        }
    return results


def ai_vs_ml_engineer_test(df: pd.DataFrame) -> dict[str, float]:
    """Compare AI Engineer vs ML Engineer salaries - sample-size sensitive."""
    ai = df.loc[df["role_label"] == "AI Engineer", "salary_min"].dropna()
    ml = df.loc[df["role_label"] == "ML Engineer", "salary_min"].dropna()
    if len(ai) < 5 or len(ml) < 5:
        return {
            "p_value": float("nan"),
            "effect_median_diff_lpa": 0.0,
            "n_ai": float(len(ai)),
            "n_ml": float(len(ml)),
        }
    stat, pval = stats.mannwhitneyu(ai, ml, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "effect_median_diff_lpa": _median_salary_lpa(ai) - _median_salary_lpa(ml),
        "n_ai": float(len(ai)),
        "n_ml": float(len(ml)),
    }


def salary_disclosure_chi_squared(df: pd.DataFrame) -> dict[str, float]:
    """2×2: remote × salary_disclosed."""
    table = pd.crosstab(df["is_remote"], df["salary_disclosed"])
    chi2, pval, _, _ = stats.chi2_contingency(table.values)
    return {"chi2": float(chi2), "p_value": float(pval)}


def simulate_null_p_values(n_sims: int = 1000, n_per_group: int = 120) -> np.ndarray:
    """Under null (no pay gap), p-values should be ~Uniform(0,1)."""
    pvals: list[float] = []
    for _ in range(n_sims):
        a = RNG.lognormal(2.6, 0.4, n_per_group) * 1_000_000
        b = RNG.lognormal(2.6, 0.4, n_per_group) * 1_000_000
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        pvals.append(float(p))
    return np.array(pvals)


def plot_seniority_controlled_boxplots(df: pd.DataFrame, path: Path) -> None:
    """One panel per seniority level: remote vs on-site within each."""
    fig, axes = plt.subplots(1, len(SENIORITY_BINS), figsize=(7.0, 2.9), sharey=True)
    for ax, band in zip(axes, SENIORITY_BINS, strict=True):
        sub = df[(df["seniority"] == band) & df["salary_min"].notna()].copy()
        sub["salary_lpa"] = sub["salary_min"] / 100_000
        groups = [
            sub.loc[sub["is_remote"], "salary_lpa"],
            sub.loc[~sub["is_remote"], "salary_lpa"],
        ]
        ax.boxplot(groups, tick_labels=["Remote", "On-site"])
        ax.set_title(band.title())
        ax.set_ylabel("Salary (₹ LPA)" if band == SENIORITY_BINS[0] else "")
    fig.suptitle(
        "Remote vs on-site pay within each seniority level (disclosed salaries)", fontsize=12
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_p_value_null_distribution(pvals: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pvals, bins=20, color="#5C4D7D", edgecolor="white", density=True)
    ax.axhline(1.0, color="#E94F37", linestyle="--", label="Uniform under null")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("p-values under simulated null (no true effect)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def _fmt(key: str, val: float) -> str:
    """Readable numbers: counts as integers, p-values in scientific notation when tiny."""
    if key.startswith("n_"):
        return f"{int(val):,}"
    if key == "p_value":
        return f"{val:.2e}" if val < 1e-3 else f"{val:.4f}"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def print_interpretation(label: str, result: dict[str, float], null: str) -> None:
    print(f"\n=== {label} ===")
    print(f"Null: {null}")
    for key, val in result.items():
        print(f"  {key}: {_fmt(key, val)}")
    p = result.get("p_value", float("nan"))
    if np.isnan(p):
        print("  Interpretation: insufficient data for this comparison.")
    elif p < 0.05:
        print("  Interpretation: reject null at α=0.05 — pattern unlikely if null were true.")
    else:
        print("  Interpretation: do not reject null at α=0.05 — not enough evidence for a gap.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = load_or_generate_jobs()
    if "role_label" not in df.columns and "title" in df.columns:
        df["role_label"] = df["title"]

    t1 = remote_vs_onsite_test(df)
    print_interpretation(
        "Test 1 — Remote vs on-site (unadjusted)",
        t1,
        "Remote median salary ≤ on-site median salary",
    )

    strat = remote_vs_onsite_by_seniority(df)
    print("\n=== Test 1b — Within each title-seniority level (confounding check) ===")
    for band, res in strat.items():
        print(
            f"  [{band}] p={res['p_value']:.4f}, Δ median ₹{res['effect_median_diff_lpa']:.1f}L, n={int(res['n_remote'])} remote / {int(res['n_onsite'])} on-site"
        )

    t2 = ai_vs_ml_engineer_test(df)
    print_interpretation(
        "Test 2 — AI Engineer vs ML Engineer",
        t2,
        "Medians are equal between AI Engineer and ML Engineer postings",
    )

    t3 = salary_disclosure_chi_squared(df)
    print_interpretation(
        "Test 3 — Salary disclosure vs remote",
        t3,
        "Remote status and salary disclosure are independent",
    )

    pvals = simulate_null_p_values(1000)
    frac = float((pvals < 0.05).mean())
    print(f"\nNull simulation: {frac:.1%} of p-values < 0.05 (expect ~5%)")

    plot_seniority_controlled_boxplots(
        df, FIGURES_DIR / "ch08_remote_vs_onsite_with_seniority_controls.png"
    )
    plot_p_value_null_distribution(pvals, FIGURES_DIR / "ch08_p_value_distribution_under_null.png")

    print(f"\nRepo root: {REPO_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
