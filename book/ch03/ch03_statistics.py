#!/usr/bin/env python3
"""Chapter 3: descriptive statistics on synthetic TalentLens salary data."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from talentlens.paths import BOOK_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

FIGURES_DIR = BOOK_DIR / "ch03" / "reports" / "figures"
RNG = np.random.default_rng(42)
N_ROWS = 400


def generate_synthetic_salaries(n: int = N_ROWS) -> pd.Series:
    """Log-normal salaries (INR LPA) - median ~15, mean ~20, tail to 80+."""
    # mu/sigma chosen so exp(mu) ≈ 15 LPA median on annual INR lakhs scale
    raw = RNG.lognormal(mean=2.65, sigma=0.55, size=n)
    return pd.Series(np.clip(raw, 4.0, 95.0), name="salary_lpa")


def interpret_descriptive(series: pd.Series) -> None:
    """Print describe() output with TalentLens-oriented commentary."""
    desc = series.describe(percentiles=[0.25, 0.5, 0.75])
    print("\n--- .describe() on synthetic annual salary (INR LPA) ---")
    print(desc.to_string())
    mean_val = series.mean()
    median_val = series.median()
    skew_val = series.skew()
    ratio = mean_val / median_val if median_val else float("nan")
    print(f"\nMean ₹{mean_val:.1f}L vs median ₹{median_val:.1f}L (ratio {ratio:.2f}x)")
    print(f"Skew: {skew_val:.2f} — right-skewed if > 1.0")
    if mean_val > median_val * 1.15:
        print(
            "Interpretation: a few high-paying roles pull the mean up. "
            "For 'typical' pay, lead with the median."
        )


def interpret_log_transform(series: pd.Series) -> None:
    """Compare skew before and after log1p."""
    log_s = np.log1p(series)
    raw_skew = stats.skew(series, bias=False)
    log_skew = stats.skew(log_s, bias=False)
    print("\n--- Log transform (np.log1p) ---")
    print(f"Skew raw: {raw_skew:.2f}  |  Skew log1p: {log_skew:.2f}")
    print(
        "Interpretation: log1p compresses the right tail so correlations and "
        "plots in Chapter 7 behave more honestly."
    )


def save_histogram(series: pd.Series, path: Path) -> None:
    """Raw vs log-transformed histogram side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mean_v, median_v = series.mean(), series.median()
    axes[0].hist(series, bins=40, color="#2E86AB", edgecolor="white")
    axes[0].axvline(mean_v, color="#E94F37", linestyle="--", label=f"mean ₹{mean_v:.1f}L")
    axes[0].axvline(median_v, color="#F6AE2D", linestyle="-", label=f"median ₹{median_v:.1f}L")
    axes[0].set_title("Raw salary (right-skewed)")
    axes[0].set_xlabel("Annual salary (INR LPA)")
    axes[0].legend(fontsize=8)
    log_s = np.log1p(series)
    axes[1].hist(log_s, bins=40, color="#4A7C59", edgecolor="white")
    axes[1].set_title("log1p(salary)")
    axes[1].set_xlabel("log1p(INR LPA)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def save_boxplot(series: pd.Series, path: Path) -> None:
    """Box plot showing quartiles and outliers."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(series, vert=True, patch_artist=True)
    ax.set_ylabel("Annual salary (INR LPA)")
    ax.set_title("Salary distribution — quartiles & outliers")
    q1, med, q3 = series.quantile([0.25, 0.5, 0.75])
    ax.text(1.15, med, f"median ₹{med:.1f}L", va="center", fontsize=9)
    ax.text(1.15, q1, f"Q1 ₹{q1:.1f}L", va="center", fontsize=8)
    ax.text(1.15, q3, f"Q3 ₹{q3:.1f}L", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def save_skew_demonstration(path: Path) -> None:
    """Symmetric vs right-skew with mean/median marked."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.linspace(-3, 3, 200)
    sym = stats.norm.pdf(x, 0, 1)
    axes[0].plot(x, sym, color="#2E86AB")
    axes[0].axvline(0, color="#F6AE2D", label="mean = median")
    axes[0].set_title("Symmetric (rare for salary)")
    axes[0].legend(fontsize=8)

    skew_x = np.linspace(0, 8, 200)
    skew_y = stats.gamma.pdf(skew_x, a=2, scale=1.5)
    axes[1].plot(skew_x, skew_y, color="#E94F37")
    mean_s = 3.0
    median_s = 2.3
    axes[1].axvline(mean_s, color="#E94F37", linestyle="--", label="mean")
    axes[1].axvline(median_s, color="#F6AE2D", linestyle="-", label="median")
    axes[1].set_title("Right-skewed (typical salary)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def main() -> int:
    """Generate data, print interpretations, write figures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    salaries = generate_synthetic_salaries()
    logger.info("Generated %d synthetic salaries (repo root: %s)", len(salaries), REPO_ROOT)

    interpret_descriptive(salaries)
    interpret_log_transform(salaries)

    save_histogram(salaries, FIGURES_DIR / "ch03_salary_histogram.png")
    save_boxplot(salaries, FIGURES_DIR / "ch03_salary_boxplot.png")
    save_skew_demonstration(FIGURES_DIR / "ch03_skew_demonstration.png")

    try:
        fig_rel = FIGURES_DIR.relative_to(REPO_ROOT)
    except ValueError:
        fig_rel = FIGURES_DIR
    print(f"\nFigures saved under {fig_rel}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
