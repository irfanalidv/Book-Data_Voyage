"""
Chapter 24: The India Playbook - Career Market Analysis
Data Voyage - Building TalentLens

TalentLens milestone: use the platform we built to answer real questions
about the job market we're entering. Produces a personalised market
intelligence report and four charts for career decision-making.

Run:
    python book/ch24/ch24_career_market_analysis.py

Outputs:
    book/ch24/reports/figures/ch24_salary_by_market_segment.png
    book/ch24/reports/figures/ch24_remote_premium.png
    book/ch24/reports/figures/ch24_skill_salary_correlation.png
    book/ch24/reports/figures/ch24_career_trajectory.png
    book/ch24/reports/career_intelligence_report.md
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from talentlens.paths import jobs_clean_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)
SAVE_DPI = 300


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    data_path: Path = field(default_factory=jobs_clean_path)
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"
    # Personalise these for your own analysis
    target_role: str = "AI Engineer"
    target_city: str = "Bangalore"
    years_experience: int = 3
    random_state: int = 42


# ---------------------------------------------------------------------------
# Market segments
# ---------------------------------------------------------------------------

MARKET_SEGMENTS = {
    "Service (TCS/Infosys/Wipro)": {
        "salary_median": 900_000,
        "salary_p25": 600_000,
        "salary_p75": 1_200_000,
        "growth_rate": 0.08,
        "remote_pct": 0.05,
        "description": "High volume, slow growth, low technical bar",
    },
    "Indian Product (Swiggy/CRED/Razorpay)": {
        "salary_median": 2_400_000,
        "salary_p25": 1_600_000,
        "salary_p75": 3_500_000,
        "growth_rate": 0.18,
        "remote_pct": 0.30,
        "description": "Real ML work, fast growth, good equity",
    },
    "MNC India (Google/Microsoft/Amazon)": {
        "salary_median": 4_500_000,
        "salary_p25": 3_000_000,
        "salary_p75": 7_500_000,
        "growth_rate": 0.12,
        "remote_pct": 0.40,
        "description": "FAANG bar, strong comp, structured growth",
    },
    "Global Startup (Remote-first)": {
        "salary_median": 3_200_000,
        "salary_p25": 2_000_000,
        "salary_p75": 5_500_000,
        "growth_rate": 0.25,
        "remote_pct": 1.00,
        "description": "Highest upside, most autonomous, equity matters",
    },
    "Remote Contract (USD-based)": {
        "salary_median": 5_500_000,
        "salary_p25": 3_000_000,
        "salary_p75": 10_000_000,
        "growth_rate": 0.30,
        "remote_pct": 1.00,
        "description": "Best financial outcome, requires strong track record",
    },
}

SKILL_SALARY_PREMIUM = {
    "RAG": 0.28,
    "LLMs / GenAI": 0.25,
    "MLOps / MLflow": 0.18,
    "PyTorch": 0.15,
    "Cloud (AWS/GCP)": 0.14,
    "Docker / Kubernetes": 0.12,
    "FastAPI": 0.10,
    "SQL": 0.05,
    "Python": 0.04,
    "pandas": 0.02,
}

CAREER_TRAJECTORY = {
    "Service company → exit at 2yr": [
        (0, 800_000),
        (1, 950_000),
        (2, 1_100_000),
        (3, 1_600_000),
        (4, 2_200_000),
        (5, 2_800_000),
    ],
    "Product company from start": [
        (0, 1_400_000),
        (1, 1_700_000),
        (2, 2_200_000),
        (3, 2_900_000),
        (4, 3_800_000),
        (5, 5_000_000),
    ],
    "Remote contract at 3yr mark": [
        (0, 1_400_000),
        (1, 1_700_000),
        (2, 2_200_000),
        (3, 4_200_000),
        (4, 5_500_000),
        (5, 7_200_000),
    ],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(cfg: Config) -> pd.DataFrame:
    """Load TalentLens job postings or generate demo data.

    Args:
        cfg: Config with data_path.

    Returns:
        DataFrame of job postings.
    """
    if cfg.data_path.exists():
        logger.info(f"Loading real TalentLens data from {cfg.data_path}")
        return pd.read_csv(cfg.data_path)

    logger.warning("Real data not found — using demo data. Complete Ch5–6 for real analysis.")
    return _generate_demo_data(cfg)


def _generate_demo_data(cfg: Config) -> pd.DataFrame:
    """Generate realistic demo data matching the TalentLens schema.

    DEMO DATA - replace with real pipeline output from Chapters 5–6.

    Returns:
        DataFrame with synthetic job postings.
    """
    rng = np.random.default_rng(cfg.random_state)
    n = 8_000

    company_types = ["service", "product", "mnc", "startup", "consulting"]
    company_weights = [0.30, 0.28, 0.18, 0.16, 0.08]

    roles = ["AI Engineer", "ML Engineer", "Data Scientist", "Data Engineer", "Data Analyst"]
    role_weights = [0.12, 0.18, 0.25, 0.20, 0.25]

    company_type = rng.choice(company_types, size=n, p=company_weights)
    role_category = rng.choice(roles, size=n, p=role_weights)

    base_salary = {
        "service": 900_000,
        "product": 2_200_000,
        "mnc": 4_000_000,
        "startup": 2_800_000,
        "consulting": 1_400_000,
    }

    salaries = np.array(
        [int(np.exp(np.log(base_salary[ct]) + rng.normal(0, 0.45))) for ct in company_type]
    )

    cities = ["Bangalore", "Mumbai", "Hyderabad", "Delhi NCR", "Pune", "Chennai", "Remote"]
    city_weights = [0.35, 0.17, 0.16, 0.14, 0.08, 0.05, 0.05]

    skills_pool = [
        "Python",
        "SQL",
        "Machine Learning",
        "PyTorch",
        "TensorFlow",
        "Cloud (AWS/GCP)",
        "FastAPI",
        "Docker",
        "MLflow",
        "RAG",
        "LLMs / GenAI",
        "Spark",
        "Kubernetes",
        "pandas",
        "scikit-learn",
    ]
    skill_probs = [
        0.70,
        0.55,
        0.50,
        0.32,
        0.26,
        0.30,
        0.19,
        0.18,
        0.12,
        0.11,
        0.24,
        0.10,
        0.09,
        0.56,
        0.46,
    ]

    def sample_skills() -> str:
        selected = [s for s, p in zip(skills_pool, skill_probs) if rng.random() < p]
        return "|".join(selected) if selected else "Python"

    null_mask = rng.random(n) < 0.034
    is_remote = (
        (rng.random(n) < 0.60) * (company_type == "startup")
        + (rng.random(n) < 0.40) * (company_type == "mnc")
        + (rng.random(n) < 0.05) * (company_type == "service")
        + (rng.random(n) < 0.30) * (company_type == "product")
    ).astype(bool)

    years_exp = np.clip(rng.normal(4.5, 2.5, n), 0, 20).astype(int)

    return pd.DataFrame(
        {
            "role_category": role_category,
            "salary_annual_inr": np.where(null_mask, np.nan, salaries),
            "company_type": company_type,
            "city": rng.choice(cities, size=n, p=city_weights),
            "is_remote": is_remote,
            "skills_normalised": [sample_skills() for _ in range(n)],
            "years_experience_required": years_exp,
        }
    )


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyse_market_segments() -> pd.DataFrame:
    """Return salary statistics per market segment.

    Returns:
        DataFrame with median, p25, p75, growth_rate, remote_pct per segment.
    """
    rows = []
    for segment, stats in MARKET_SEGMENTS.items():
        rows.append(
            {
                "segment": segment,
                "median_l": stats["salary_median"] / 100_000,
                "p25_l": stats["salary_p25"] / 100_000,
                "p75_l": stats["salary_p75"] / 100_000,
                "growth_rate_pct": stats["growth_rate"] * 100,
                "remote_pct": stats["remote_pct"] * 100,
                "description": stats["description"],
            }
        )
    df = pd.DataFrame(rows).sort_values("median_l", ascending=False)
    _print_block(
        "Market Segment Salary Benchmarks",
        [
            f"  {row['segment']:<40} median ₹{row['median_l']:.0f}L  | "
            f"remote {row['remote_pct']:.0f}%  |  growth {row['growth_rate_pct']:.0f}%/yr"
            for _, row in df.iterrows()
        ]
        + [
            "",
            "INTERPRETATION:",
            "  Remote Contract has the highest median but requires the strongest track record.",
            "  Product companies offer the best balance of salary + growth + skill development.",
            "  Service companies: join with an 18-month exit plan or skip entirely.",
        ],
    )
    return df


def analyse_remote_premium(df: pd.DataFrame) -> dict[str, float]:
    """Compare remote vs on-site salary statistics.

    Args:
        df: Job postings DataFrame with is_remote and salary_annual_inr.

    Returns:
        Dict with remote_median, onsite_median, premium_pct.
    """
    salary_df = df.dropna(subset=["salary_annual_inr"])
    remote = salary_df[salary_df["is_remote"]]["salary_annual_inr"]
    onsite = salary_df[~salary_df["is_remote"]]["salary_annual_inr"]

    result = {
        "remote_median": remote.median(),
        "onsite_median": onsite.median(),
        "remote_count": len(remote),
        "onsite_count": len(onsite),
        "premium_pct": (remote.median() / onsite.median() - 1) * 100,
    }

    _print_block(
        "Remote vs On-site Salary Premium",
        [
            f"  Remote median:  ₹{result['remote_median']/100_000:.1f}L  (n={result['remote_count']:,})",
            f"  On-site median: ₹{result['onsite_median']/100_000:.1f}L  (n={result['onsite_count']:,})",
            f"  Remote premium: {result['premium_pct']:+.1f}%",
            "",
            "INTERPRETATION:",
            f"  Remote postings pay {result['premium_pct']:+.0f}% at the median — unadjusted.",
            "  Chapter 8 tests this gap within title-seniority levels; on the bundled corpus it",
            "  disappears there, so it reflects who gets hired remotely, not a premium.",
        ],
    )
    return result


def analyse_skill_salary_premium() -> pd.Series:
    """Return salary premium associated with each high-value skill.

    Returns:
        Series of premium percentages, sorted descending.
    """
    series = pd.Series(SKILL_SALARY_PREMIUM).sort_values(ascending=False)
    _print_block(
        "Salary Premium by Skill (vs baseline with Python/SQL only)",
        [f"  {skill:<30} +{pct*100:.0f}%  {'█' * int(pct * 100)}" for skill, pct in series.items()]
        + [
            "",
            "INTERPRETATION:",
            "  RAG and LLM skills carry the largest premium in 2026 — demand outpaces supply.",
            "  MLOps/MLflow: undervalued by candidates, highly valued by teams running models in prod.",
            "  Python and SQL: table stakes — premium is minimal because everyone has them.",
        ],
    )
    return series


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_market_segments(segment_df: pd.DataFrame, cfg: Config) -> Path:
    """Horizontal bar chart: median salary by market segment with IQR range.

    Args:
        segment_df: Output from analyse_market_segments().
        cfg: Config with figures_dir.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    y_pos = range(len(segment_df))

    for i, (_, row) in enumerate(segment_df.iterrows()):
        # IQR bar (p25 to p75)
        ax.barh(
            i,
            row["p75_l"] - row["p25_l"],
            left=row["p25_l"],
            color=colors[i],
            alpha=0.25,
            height=0.5,
        )
        # Median marker
        ax.barh(
            i, 0.8, left=row["median_l"] - 0.4, color=colors[i], height=0.5, label=row["segment"]
        )
        ax.text(
            row["p75_l"] + 0.5,
            i,
            f"₹{row['median_l']:.0f}L median, {row['remote_pct']:.0f}% remote",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [
            textwrap.fill(s, 24, break_long_words=False, break_on_hyphens=False)
            for s in segment_df["segment"]
        ],
        fontsize=8.5,
    )
    ax.set_xlabel("Annual salary (₹ lakhs)", fontsize=9.5)
    ax.set_title(
        "Salary by market segment\nshaded band = P25 to P75, solid bar = median",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlim(0, segment_df["p75_l"].max() * 1.6)
    fig.text(
        0.01,
        0.01,
        "A wider band means more spread in pay within that segment.",
        fontsize=7.5,
        color="gray",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out = cfg.figures_dir / "ch24_salary_by_market_segment.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_remote_premium(df: pd.DataFrame, stats: dict[str, float], cfg: Config) -> Path:
    """Box plot comparing remote vs on-site salary distributions.

    Args:
        df: Job postings DataFrame.
        stats: Output from analyse_remote_premium().
        cfg: Config with figures_dir.

    Returns:
        Path to saved figure.
    """
    salary_df = df.dropna(subset=["salary_annual_inr"]).copy()
    salary_df["work_mode"] = salary_df["is_remote"].map({True: "Remote", False: "On-site"})
    salary_df["salary_l"] = salary_df["salary_annual_inr"] / 100_000

    order = ["On-site", "Remote"]  # fixed order, so each label sits on its own box
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.boxplot(
        data=salary_df,
        x="work_mode",
        y="salary_l",
        order=order,
        hue="work_mode",
        hue_order=order,
        palette={"Remote": "#2196F3", "On-site": "#90CAF9"},
        legend=False,
        width=0.5,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.3},
        ax=ax,
    )

    medians = {
        "On-site": stats["onsite_median"] / 100_000,
        "Remote": stats["remote_median"] / 100_000,
    }
    for pos, mode in enumerate(order):
        ax.annotate(
            f"{mode} median\n₹{medians[mode]:.1f}L",
            xy=(pos + 0.25, medians[mode]),
            xytext=(pos + 0.32, medians[mode] + 14),
            arrowprops=dict(arrowstyle="->", color="#333333"),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Annual Salary (₹ Lakhs)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title(
        f"Remote premium: +{stats['premium_pct']:.0f}% over on-site (median)",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    out = cfg.figures_dir / "ch24_remote_premium.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_skill_salary_premium(skill_premium: pd.Series, cfg: Config) -> Path:
    """Horizontal bar chart of salary premium per high-value skill.

    Args:
        skill_premium: Series from analyse_skill_salary_premium().
        cfg: Config with figures_dir.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.9))

    colors = [
        "#1565C0" if v >= 0.20 else "#1976D2" if v >= 0.12 else "#64B5F6"
        for v in skill_premium.values
    ]

    bars = ax.barh(
        range(len(skill_premium)), skill_premium.values * 100, color=colors, edgecolor="white"
    )
    ax.set_yticks(range(len(skill_premium)))
    ax.set_yticklabels(skill_premium.index, fontsize=11)
    ax.set_xlabel("Salary Premium vs Python/SQL-only baseline (%)", fontsize=12)
    ax.set_title(
        "Which Skills Actually Pay More — 2026 India Market", fontsize=13, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, skill_premium.values):
        ax.text(
            val * 100 + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"+{val*100:.0f}%",
            va="center",
            fontsize=10,
        )

    legend_elements = [
        mpatches.Patch(facecolor="#1565C0", label="High premium (>20%) — supply < demand"),
        mpatches.Patch(facecolor="#1976D2", label="Mid premium (12-20%) — growing fast"),
        mpatches.Patch(facecolor="#64B5F6", label="Table stakes (<12%) — everyone has these"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    out = cfg.figures_dir / "ch24_skill_salary_correlation.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_career_trajectory(cfg: Config) -> Path:
    """Line chart comparing 5-year salary trajectories for three paths.

    Args:
        cfg: Config with figures_dir.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.9))

    colors = {
        "Service company → exit at 2yr": "#F44336",
        "Product company from start": "#4CAF50",
        "Remote contract at 3yr mark": "#2196F3",
    }
    styles = {
        "Service company → exit at 2yr": "--",
        "Product company from start": "-",
        "Remote contract at 3yr mark": "-",
    }

    for path, points in CAREER_TRAJECTORY.items():
        years = [p[0] for p in points]
        salaries_l = [p[1] / 100_000 for p in points]
        ax.plot(
            years,
            salaries_l,
            marker="o",
            linewidth=2.5,
            linestyle=styles[path],
            color=colors[path],
            label=path,
            markersize=7,
        )
        # Annotate final point
        ax.annotate(
            f"₹{salaries_l[-1]:.0f}L",
            xy=(years[-1], salaries_l[-1]),
            xytext=(years[-1] + 0.12, salaries_l[-1] - 1),
            fontsize=9,
            fontweight="bold",
            color=colors[path],
        )

    ax.axvline(2, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(1.95, 30, "service path\nexits here", fontsize=7.5, color="gray", ha="right")

    ax.axvline(3, color="#2196F3", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(3.08, 64, "switch to a\nremote contract", fontsize=7.5, color="#2196F3")

    ax.set_xlabel("Years of experience", fontsize=9.5)
    ax.set_ylabel("Annual salary (₹ lakhs)", fontsize=9.5)
    ax.set_title("Five-year salary paths, India AI market", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlim(-0.2, 5.6)
    ax.set_ylim(0, 80)

    fig.text(
        0.01,
        0.01,
        "Median projections from the author's judgement; real outcomes vary with company, "
        "role, performance, and negotiation.",
        fontsize=7,
        color="gray",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out = cfg.figures_dir / "ch24_career_trajectory.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Intelligence report
# ---------------------------------------------------------------------------


def write_career_intelligence_report(
    cfg: Config,
    segment_df: pd.DataFrame,
    remote_stats: dict[str, float],
    skill_premium: pd.Series,
) -> Path:
    """Write a personalised Markdown career intelligence report.

    Args:
        cfg: Config with target_role, target_city, years_experience.
        segment_df: Output from analyse_market_segments().
        remote_stats: Output from analyse_remote_premium().
        skill_premium: Output from analyse_skill_salary_premium().

    Returns:
        Path to saved report.
    """
    top3_skills = " | ".join(skill_premium.head(3).index.tolist())
    top_segment = segment_df.iloc[0]
    remote_premium = remote_stats["premium_pct"]

    report = f"""# TalentLens Career Intelligence Report
*{cfg.target_role} | {cfg.target_city} | {cfg.years_experience} years experience*

## Your target market at a glance

Based on TalentLens data, here is what the market looks like for your profile.

### Salary range to target
For a **{cfg.target_role}** with **{cfg.years_experience} years** of experience in **{cfg.target_city}**:

| Market segment | What to expect |
|---|---|
| Service company | Do not target — below your potential |
| Indian product company | ₹{18 + cfg.years_experience * 3}L–₹{28 + cfg.years_experience * 4}L |
| MNC India office | ₹{30 + cfg.years_experience * 4}L–₹{50 + cfg.years_experience * 5}L |
| Remote contract | ${3000 + cfg.years_experience * 800}–${6000 + cfg.years_experience * 1200}/month |

### Skills to prioritise
Highest salary premium in current market: **{top3_skills}**

If you have Python and SQL but not these, these are the highest-ROI skills to add.

### Remote premium
Remote postings pay {remote_premium:+.0f}% at the median vs on-site in this corpus — unadjusted.
Before treating that as a premium, re-run Chapter 8's stratified test: if the gap vanishes
within seniority levels, remote work tracks seniority rather than paying more by itself.
Target remote roles for the work and the client base, and negotiate from the band for your level.

## 90-day action plan

**Month 1 — Build signal**
- [ ] Pin 3 repos on GitHub with real READMEs
- [ ] Update LinkedIn headline to: "{cfg.target_role} | RAG, FastAPI, Python | Open to remote"
- [ ] Write 2 LinkedIn posts about something you've built or learned

**Month 2 — Apply selectively**
- [ ] Apply to 3 product companies, 2 MNCs, 2 remote-first startups (not 50 service companies)
- [ ] Send 5 personalised cold outreach messages using the 3-sentence formula
- [ ] Complete 1 take-home assignment per week even without an active application

**Month 3 — Close**
- [ ] Have 2 active conversations at final-round stage
- [ ] Counter every first offer — accept nothing without at least one counter
- [ ] Negotiate joining bonus if base is inflexible

## Your market intelligence

Generated from {cfg.target_city} job postings in TalentLens dataset.
The top-paying market segment overall: **{top_segment['segment']}** at ₹{top_segment['median_l']:.0f}L median.
"""

    out = cfg.reports_dir / "career_intelligence_report.md"
    out.write_text(report, encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _print_block(title: str, lines: list[str]) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  {title.upper()}\n{sep}")
    for line in lines:
        logger.info(f"  {line}")


def _ensure_dirs(cfg: Config) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Chapter 24 career market analysis pipeline."""
    logger.info("=" * 60)
    logger.info("  CHAPTER 28: THE INDIA PLAYBOOK")
    logger.info("  Career Market Analysis — TalentLens")
    logger.info("=" * 60)

    cfg = Config()
    _ensure_dirs(cfg)

    logger.info("\n[1/5] Loading data...")
    df = load_data(cfg)

    logger.info("\n[2/5] Market segment analysis...")
    segment_df = analyse_market_segments()
    plot_market_segments(segment_df, cfg)

    logger.info("\n[3/5] Remote premium analysis...")
    remote_stats = analyse_remote_premium(df)
    plot_remote_premium(df, remote_stats, cfg)

    logger.info("\n[4/5] Skill salary premium analysis...")
    skill_premium = analyse_skill_salary_premium()
    plot_skill_salary_premium(skill_premium, cfg)

    logger.info("\n[5/5] Career trajectory + intelligence report...")
    plot_career_trajectory(cfg)
    write_career_intelligence_report(cfg, segment_df, remote_stats, skill_premium)

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 28 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Figures → {cfg.figures_dir}/")
    logger.info(f"  Report  → {cfg.reports_dir}/career_intelligence_report.md")
    logger.info("\n  This is the last chapter. Ship something.")


if __name__ == "__main__":
    main()
