"""
Chapter 7: Exploratory Data Analysis
Data Voyage - Building TalentLens

TalentLens milestone: interrogate 50k cleaned job postings to answer real
business questions about skills, salaries, and role distributions. Produces
three publication-quality charts and a written EDA summary.

Run (from repo root or this directory; paths resolve to this chapter folder):

    python book/ch07/ch07_exploratory_data_analysis.py

Outputs (under book/ch07/ when this file lives in that folder):

    reports/figures/ch07_salary_distribution.png
    reports/figures/ch07_skill_frequency.png
    reports/figures/ch07_role_comparison.png
    reports/eda_summary.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)
sns.set_palette("husl")
FIGURE_DPI = 150
SAVE_DPI = 300

_CHAPTER_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _CHAPTER_ROOT.parent.parent


def _default_jobs_clean_path() -> Path:
    from talentlens.paths import jobs_clean_path

    primary = jobs_clean_path()
    if primary.exists():
        return primary
    return _CHAPTER_ROOT / "data" / "clean" / "jobs_clean.csv"


@dataclass
class Config:
    data_path: Path = field(default_factory=_default_jobs_clean_path)
    figures_dir: Path = field(default_factory=lambda: _CHAPTER_ROOT / "reports" / "figures")
    reports_dir: Path = field(default_factory=lambda: _CHAPTER_ROOT / "reports")
    salary_col: str = "salary_annual_inr"
    role_col: str = "role_category"
    skills_col: str = "skills_normalised"  # pipe-separated string: "Python|SQL|PyTorch"
    remote_col: str = "is_remote"
    salary_bands: list[tuple[float, float, str]] = field(
        default_factory=lambda: [
            (0, 800_000, "₹0–8L"),
            (800_000, 1_500_000, "₹8–15L"),
            (1_500_000, 3_000_000, "₹15–30L"),
            (3_000_000, float("inf"), "₹30L+"),
        ]
    )
    top_n_skills: int = 20
    top_n_roles: int = 7


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``salary_annual_inr`` and ``role_category`` if a pre-fix
    ``jobs_clean.csv`` is on disk.

    Chapter 6's cleaning pipeline writes these columns natively. This
    shim exists so readers who have an older ``jobs_clean.csv`` (from
    before ch06 was updated) still get a working EDA run without
    re-executing Chapter 6 first. New runs of ch06 make this a no-op.

    Args:
        df: DataFrame loaded from ``jobs_clean.csv``.

    Returns:
        DataFrame with both derived columns guaranteed to exist.
    """
    if "salary_annual_inr" not in df.columns:
        if {"salary_min", "salary_max"}.issubset(df.columns):
            midpoint = (df["salary_min"] + df["salary_max"]) / 2.0
            midpoint = midpoint.fillna(df["salary_min"]).fillna(df["salary_max"])
            df = df.assign(salary_annual_inr=midpoint)
            logger.info(
                "load_data: derived salary_annual_inr on the fly — "
                "regenerate jobs_clean.csv via Chapter 6 for the persisted column."
            )
        else:
            raise KeyError(
                "jobs_clean.csv has no salary_annual_inr and no salary_min/max "
                "to derive it from. Re-run Chapter 6."
            )
    if "role_category" not in df.columns:
        if "role_label" in df.columns:
            df = df.assign(role_category=df["role_label"])
        else:
            df = df.assign(role_category="Data Scientist")
            logger.warning(
                "load_data: jobs_clean.csv has no role_category / role_label — "
                "assigning a constant default. Re-run Chapter 6 for real labels."
            )
    return df


def load_data(cfg: Config) -> pd.DataFrame:
    """Load the cleaned TalentLens job postings dataset.

    Falls back to synthetic demo data if the real dataset isn't found,
    so you can run this chapter before completing Chapter 5/6.

    Args:
        cfg: Config object with data_path.

    Returns:
        DataFrame of job postings with at minimum:
            salary_annual_inr, role_category, skills_normalised, is_remote.
    """
    if cfg.data_path.exists():
        logger.info(f"Loading real data from {cfg.data_path}")
        df = pd.read_csv(cfg.data_path)
        df = _ensure_derived_columns(df)
        logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df

    logger.warning(
        f"Data file not found at {cfg.data_path}. "
        "Generating demo data — complete Chapters 5–6 for real data."
    )
    return _generate_demo_data()


def _generate_demo_data() -> pd.DataFrame:
    """Generate realistic synthetic TalentLens data for demo purposes.

    DEMO DATA - replace with real data from Chapter 5/6 pipeline.
    Schema matches the real dataset exactly so all downstream code works.

    Returns:
        DataFrame with ~5,000 synthetic job postings.
    """
    rng = np.random.default_rng(42)
    n = 5_000

    roles = ["AI Engineer", "ML Engineer", "Data Scientist", "Data Engineer", "Data Analyst"]
    role_weights = [0.10, 0.17, 0.25, 0.20, 0.28]
    role_salary_params = {
        "AI Engineer": {"median": 2_450_000, "sigma": 0.55},
        "ML Engineer": {"median": 2_180_000, "sigma": 0.50},
        "Data Scientist": {"median": 1_820_000, "sigma": 0.48},
        "Data Engineer": {"median": 1_760_000, "sigma": 0.45},
        "Data Analyst": {"median": 1_040_000, "sigma": 0.42},
    }
    common_skills = [
        "Python",
        "SQL",
        "Machine Learning",
        "PyTorch",
        "TensorFlow",
        "Cloud (AWS/GCP/Azure)",
        "FastAPI",
        "Docker",
        "MLflow",
        "RAG",
        "LLMs",
        "Spark",
        "Kubernetes",
        "dbt",
        "Airflow",
        "pandas",
        "scikit-learn",
        "Git",
        "Linux",
        "Statistics",
    ]
    skill_probs = [
        0.68,
        0.54,
        0.49,
        0.31,
        0.25,
        0.30,
        0.19,
        0.18,
        0.12,
        0.12,
        0.24,
        0.10,
        0.09,
        0.07,
        0.06,
        0.55,
        0.45,
        0.60,
        0.40,
        0.35,
    ]

    role_col = rng.choice(roles, size=n, p=role_weights)
    salaries = np.array(
        [
            int(
                np.exp(
                    np.log(role_salary_params[r]["median"])
                    + rng.normal(0, role_salary_params[r]["sigma"])
                )
            )
            for r in role_col
        ]
    )
    is_remote = rng.random(n) < 0.28

    def random_skills(n_rows: int) -> list[str]:
        result = []
        for _ in range(n_rows):
            selected = [s for s, p in zip(common_skills, skill_probs) if rng.random() < p]
            result.append("|".join(selected) if selected else "Python")
        return result

    null_salary_mask = rng.random(n) < 0.034  # 3.4% missing

    df = pd.DataFrame(
        {
            "role_category": role_col,
            "salary_annual_inr": np.where(null_salary_mask, np.nan, salaries),
            "skills_normalised": random_skills(n),
            "is_remote": is_remote,
            "company_type": rng.choice(
                ["product", "service", "startup", "consulting"],
                size=n,
                p=[0.35, 0.30, 0.25, 0.10],
            ),
            "city": rng.choice(
                ["Bangalore", "Mumbai", "Hyderabad", "Delhi NCR", "Pune", "Remote"],
                size=n,
                p=[0.38, 0.18, 0.16, 0.14, 0.08, 0.06],
            ),
        }
    )
    return df


# ---------------------------------------------------------------------------
# EDA functions
# ---------------------------------------------------------------------------


def describe_shape(mean: float, median: float, skew: float) -> str:
    """Plain-English reading of a salary distribution's shape.

    The mean/median gap and the skew statistic can disagree: a long thin tail
    raises skew while barely moving the mean. Say which one you are seeing.
    """
    ratio = mean / median if median else float("nan")
    if ratio > 1.15:
        return (
            "Right-skewed: a small number of very high-paying roles pull the mean "
            "well above the median. Report the median as 'typical' pay."
        )
    if skew > 1.0:
        return (
            "Long right tail (skew > 1) but the bulk is compact, so mean and median "
            "are close. Report the median and name the tail separately."
        )
    return "Roughly symmetric: mean and median tell the same story."


def analyse_salary_distribution(df: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Compute and interpret salary distribution statistics.

    Args:
        df: Job postings DataFrame.
        cfg: Config with salary_col and salary_bands.

    Returns:
        Dictionary of key salary statistics for downstream use.
    """
    # Chapter 6 imputed hidden salaries from group medians. Imputed values are
    # guesses, and they shrink the spread - so the distribution statistics use
    # disclosed salaries only, and we report how many rows were set aside.
    disclosed = (
        df["salary_disclosed"].astype(bool)
        if "salary_disclosed" in df.columns
        else df[cfg.salary_col].notna()
    )
    salary = df.loc[disclosed, cfg.salary_col].dropna()
    missing_pct = (1 - len(salary) / len(df)) * 100 if len(df) else 0.0

    stats_dict = {
        "count": len(salary),
        "missing_pct": missing_pct,
        "mean": salary.mean(),
        "median": salary.median(),
        "std": salary.std(),
        "p25": salary.quantile(0.25),
        "p75": salary.quantile(0.75),
        "skew": salary.skew(),
    }
    stats_dict["shape"] = describe_shape(
        stats_dict["mean"], stats_dict["median"], stats_dict["skew"]
    )

    _print_interpretation_block(
        title="Salary Distribution",
        lines=[
            f"Count:     {stats_dict['count']:,} postings with a disclosed salary "
            f"({missing_pct:.1f}% hid pay; Chapter 6 imputed those, excluded here)",
            f"Mean:      ₹{stats_dict['mean']/100_000:.1f}L",
            f"Median:    ₹{stats_dict['median']/100_000:.1f}L",
            f"Std:       ₹{stats_dict['std']/100_000:.1f}L",
            f"P25–P75:   ₹{stats_dict['p25']/100_000:.1f}L – ₹{stats_dict['p75']/100_000:.1f}L",
            f"Skew:      {stats_dict['skew']:.2f}",
            "",
            "INTERPRETATION:",
            f"  Mean/Median ratio: {stats_dict['mean']/stats_dict['median']:.2f}x",
            f"  → {stats_dict['shape']}",
        ],
    )

    return stats_dict


def analyse_skill_frequency(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Extract and rank skill frequency across all job postings.

    Args:
        df: Job postings DataFrame with pipe-separated skills column.
        cfg: Config with skills_col and top_n_skills.

    Returns:
        Series of skill frequencies as proportions, sorted descending.
    """
    all_skills: list[str] = []
    for cell in df[cfg.skills_col].dropna():
        all_skills.extend([s.strip() for s in str(cell).split("|") if s.strip()])

    skill_counts = pd.Series(all_skills).value_counts()
    skill_freq = skill_counts / len(df)  # proportion of all postings

    top = skill_freq.head(cfg.top_n_skills)

    _print_interpretation_block(
        title=f"Top {cfg.top_n_skills} Skills by Posting Frequency",
        lines=[
            f"  {skill:<30} {freq*100:5.1f}%  {'█' * int(freq * 50)}" for skill, freq in top.items()
        ]
        + [
            "",
            "INTERPRETATION:",
            f"  '{top.index[0]}' appears in {top.iloc[0]*100:.0f}% of postings — non-negotiable baseline.",
            f"  '{top.index[1]}' at {top.iloc[1]*100:.0f}% — also essential.",
            "  Skills below 10%: valuable specialisations, not baseline requirements.",
        ],
    )

    return top


def analyse_role_comparison(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compare salary distributions across role categories.

    Args:
        df: Job postings DataFrame.
        cfg: Config with role_col, salary_col, top_n_roles.

    Returns:
        DataFrame with median salary and sample size per role, sorted by median.
    """
    salary_df = df.dropna(subset=[cfg.salary_col])
    if "salary_disclosed" in salary_df.columns:
        salary_df = salary_df[salary_df["salary_disclosed"].astype(bool)]
    role_stats = (
        salary_df.groupby(cfg.role_col)[cfg.salary_col]
        .agg(median="median", count="count")
        .sort_values("median", ascending=False)
        .head(cfg.top_n_roles)
    )

    _print_interpretation_block(
        title="Median Salary by Role",
        lines=[
            f"  {role:<25} ₹{row['median']/100_000:5.1f}L  (n={int(row['count']):,})"
            for role, row in role_stats.iterrows()
        ]
        + [
            "",
            "INTERPRETATION:",
            f"  Top role: '{role_stats.index[0]}' at ₹{role_stats['median'].iloc[0]/100_000:.1f}L",
            f"  vs lowest shown: '{role_stats.index[-1]}' at ₹{role_stats['median'].iloc[-1]/100_000:.1f}L",
            f"  Premium: {role_stats['median'].iloc[0]/role_stats['median'].iloc[-1]:.1f}x",
        ],
    )

    return role_stats


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_salary_distribution(df: pd.DataFrame, cfg: Config) -> Path:
    """Plot salary histogram + box plots by role, with annotation.

    Args:
        df: Job postings DataFrame.
        cfg: Config with salary_col, role_col, figures_dir.

    Returns:
        Path to saved figure.
    """
    # Same rule as the summary statistics: disclosed salaries only, because
    # Chapter 6's imputed values would shrink the spread.
    if "salary_disclosed" in df.columns:
        df = df[df["salary_disclosed"].astype(bool)]
    salary = df[cfg.salary_col].dropna()
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.4))

    # --- Histogram ---
    ax = axes[0]
    ax.hist(
        salary / 100_000,  # convert to lakhs for readability
        bins=40,
        edgecolor="white",
        linewidth=0.5,
        color="#4C72B0",
        alpha=0.85,
    )
    median_l = salary.median() / 100_000
    mean_l = salary.mean() / 100_000
    ax.axvline(
        median_l, color="#DD8452", linewidth=2, linestyle="--", label=f"Median: ₹{median_l:.1f}L"
    )
    ax.axvline(mean_l, color="#55A868", linewidth=2, linestyle=":", label=f"Mean: ₹{mean_l:.1f}L")
    ax.set_xlabel("Annual salary (₹ lakhs)", fontsize=9)
    ax.set_ylabel("Job postings", fontsize=9)
    ax.set_title("Salary distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.98,
        0.62,
        "Median below mean: right-skewed,\nso use the median for typical pay",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#333333",
    )

    # --- Box plot by role ---
    ax2 = axes[1]
    top_roles = (
        df.dropna(subset=[cfg.salary_col])
        .groupby(cfg.role_col)[cfg.salary_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plot_df = df[df[cfg.role_col].isin(top_roles)].dropna(subset=[cfg.salary_col]).copy()
    plot_df["salary_l"] = plot_df[cfg.salary_col] / 100_000

    role_order = (
        plot_df.groupby(cfg.role_col)["salary_l"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxplot(
        data=plot_df,
        x="salary_l",
        y=cfg.role_col,
        order=role_order,
        palette="husl",
        ax=ax2,
        width=0.5,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    ax2.set_xlabel("Annual salary (₹ lakhs)", fontsize=9)
    ax2.set_ylabel("")
    ax2.set_title("Salary by role (line in each box = median)", fontsize=10, fontweight="bold")
    ax2.tick_params(labelsize=8.5)

    plt.tight_layout(h_pad=1.5)

    out_path = cfg.figures_dir / "ch07_salary_distribution.png"
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_skill_frequency(skill_freq: pd.Series, cfg: Config) -> Path:
    """Horizontal bar chart of top-N skill frequencies.

    Args:
        skill_freq: Series of skill proportions (0–1), sorted descending.
        cfg: Config with figures_dir.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.1))

    colors = [
        "#2196F3" if p >= 0.40 else "#64B5F6" if p >= 0.20 else "#BBDEFB" for p in skill_freq.values
    ]

    bars = ax.barh(range(len(skill_freq)), skill_freq.values * 100, color=colors, edgecolor="white")
    ax.set_yticks(range(len(skill_freq)))
    ax.set_yticklabels(skill_freq.index, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("% of Job Postings Mentioning This Skill", fontsize=12)
    ax.set_title(
        "Top Skills by Posting Frequency — TalentLens Dataset", fontsize=13, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    for i, (bar, val) in enumerate(zip(bars, skill_freq.values)):
        ax.text(val * 100 + 0.5, i, f"{val*100:.1f}%", va="center", fontsize=10, color="#333333")

    # Legend for colour coding
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2196F3", label="≥40% — non-negotiable baseline"),
        Patch(facecolor="#64B5F6", label="20–39% — strongly preferred"),
        Patch(facecolor="#BBDEFB", label="<20% — valuable specialisation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = cfg.figures_dir / "ch07_skill_frequency.png"
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_role_comparison(role_stats: pd.DataFrame, cfg: Config) -> Path:
    """Bar chart comparing median salary by role with sample size annotation.

    Args:
        role_stats: DataFrame with columns [median, count], indexed by role name.
        cfg: Config with figures_dir.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    roles = role_stats.index.tolist()
    medians_l = role_stats["median"].values / 100_000
    counts = role_stats["count"].values

    bars = ax.bar(
        roles, medians_l, color=sns.color_palette("husl", len(roles)), edgecolor="white", width=0.6
    )

    for bar, median, count in zip(bars, medians_l, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"₹{median:.1f}L\n(n={count:,})",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_ylabel("Median annual salary (₹ lakhs)", fontsize=9.5)
    ax.set_title("Median salary by role", fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels([r.replace(" ", "\n", 1) for r in roles], fontsize=8.5)
    ax.set_ylim(0, medians_l.max() * 1.3)
    fig.text(
        0.01,
        0.01,
        "Medians, not means: salaries are right-skewed, so the mean overstates typical pay.",
        fontsize=7.5,
        color="gray",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = cfg.figures_dir / "ch07_role_comparison.png"
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _remote_line(df: pd.DataFrame, cfg: Config) -> str:
    """One sentence comparing disclosed remote and on-site medians."""
    if cfg.remote_col not in df.columns:
        return "Remote column not available in this dataset."
    d = df
    if "salary_disclosed" in d.columns:
        d = d[d["salary_disclosed"].astype(bool)]
    med = d.groupby(d[cfg.remote_col].astype(bool))[cfg.salary_col].median() / 100_000
    n = d[cfg.remote_col].astype(bool).value_counts()
    if True not in med or False not in med:
        return "Only one of remote / on-site is present; no comparison possible."
    return (
        f"Remote median ₹{med[True]:.1f}L (n={n[True]}) vs on-site ₹{med[False]:.1f}L "
        f"(n={n[False]}) — a gap to test for confounding in Chapter 8, not a finding yet."
    )


def write_eda_summary(
    df: pd.DataFrame,
    salary_stats: dict[str, float],
    skill_freq: pd.Series,
    role_stats: pd.DataFrame,
    cfg: Config,
) -> Path:
    """Write a Markdown EDA summary report ready to paste into a slide deck.

    Args:
        df: Full job postings DataFrame.
        salary_stats: Output from analyse_salary_distribution().
        skill_freq: Output from analyse_skill_frequency().
        role_stats: Output from analyse_role_comparison().
        cfg: Config with reports_dir.

    Returns:
        Path to saved report.
    """
    top3_skills = ", ".join(skill_freq.index[:3].tolist())
    top_role = role_stats.index[0]
    top_role_median = role_stats["median"].iloc[0] / 100_000

    report = f"""# TalentLens EDA Summary
*Generated by Chapter 7 — Data Voyage*

## Dataset
- **Total postings analysed:** {len(df):,}
- **Salary data available:** {salary_stats['count']:,} ({100 - salary_stats['missing_pct']:.1f}%)

## Key Findings

### 1. Salary — use median, not mean
Skew = {salary_stats['skew']:.2f}; median ₹{salary_stats['median']/100_000:.1f}L, mean ₹{salary_stats['mean']/100_000:.1f}L,
middle half ₹{salary_stats['p25']/100_000:.1f}L – ₹{salary_stats['p75']/100_000:.1f}L (disclosed salaries only).
{salary_stats['shape']}

### 2. Baseline skills
The top three skills by posting frequency are **{top3_skills}**
({", ".join(f"{v:.0%}" for v in skill_freq.iloc[:3].tolist())} of postings).

### 3. Role salary premium
**{top_role}** has the highest median at ₹{top_role_median:.1f}L,
vs {role_stats.index[-1]} at ₹{role_stats['median'].iloc[-1]/100_000:.1f}L
({role_stats['median'].iloc[0]/role_stats['median'].iloc[-1]:.1f}x).

### 4. Remote vs on-site
{_remote_line(df, cfg)}

## What this sets up
- Chapter 8: test whether the remote gap survives controlling for seniority (from the job title).
- Chapter 9: top skills and description text become the role classifier's features.
- Log-transform salary before correlation analysis (skew = {salary_stats['skew']:.2f}).
"""

    out_path = cfg.reports_dir / "eda_summary.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _print_interpretation_block(title: str, lines: list[str]) -> None:
    """Print a formatted interpretation block to stdout."""
    separator = "=" * 60
    logger.info(f"\n{separator}")
    logger.info(f"  {title.upper()}")
    logger.info(separator)
    for line in lines:
        logger.info(f"  {line}")


def _ensure_dirs(cfg: Config) -> None:
    """Create output directories if they don't exist."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Chapter 7 EDA pipeline."""
    logger.info("=" * 60)
    logger.info("  CHAPTER 7: EXPLORATORY DATA ANALYSIS")
    logger.info("  TalentLens — Job Market Intelligence Platform")
    logger.info("=" * 60)

    cfg = Config()
    _ensure_dirs(cfg)

    # 1. Load data
    df = load_data(cfg)

    # 2. Salary analysis
    logger.info("\n[1/4] Salary distribution analysis...")
    salary_stats = analyse_salary_distribution(df, cfg)
    plot_salary_distribution(df, cfg)

    # 3. Skill frequency
    logger.info("\n[2/4] Skill frequency analysis...")
    skill_freq = analyse_skill_frequency(df, cfg)
    plot_skill_frequency(skill_freq, cfg)

    # 4. Role comparison
    logger.info("\n[3/4] Role salary comparison...")
    role_stats = analyse_role_comparison(df, cfg)
    plot_role_comparison(role_stats, cfg)

    # 5. Summary report
    logger.info("\n[4/4] Writing EDA summary report...")
    write_eda_summary(df, salary_stats, skill_freq, role_stats, cfg)

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 7 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Figures → {cfg.figures_dir}/")
    logger.info(f"  Report  → {cfg.reports_dir}/eda_summary.md")
    logger.info("\nNext: Chapter 8 — Machine Learning Fundamentals")
    logger.info("We'll use these EDA findings to frame our modelling problem.")


if __name__ == "__main__":
    main()
