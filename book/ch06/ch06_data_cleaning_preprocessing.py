# ruff: noqa: E701, E501, B905
"""
Chapter 6: Data Cleaning and Preprocessing
Data Voyage - Building TalentLens

TalentLens milestone: turn raw job postings from Chapter 5 into
data/clean/jobs_clean.csv - the file every downstream chapter uses.

Run:
    python book/ch06/ch06_data_cleaning_preprocessing.py
    python book/ch06/ch06_data_cleaning_preprocessing.py --overwrite   # rebuild jobs_clean.csv from jobs_raw.csv

Inputs:  data/raw/jobs_raw.demo.csv (default reader path after Ch5 demo)
         data/raw/jobs_raw.csv (--overwrite / live collection)
Outputs: data/clean/jobs_clean.demo.csv (default - bundled jobs_clean.csv untouched)
         data/clean/jobs_clean.csv (--overwrite only)
         book/ch06/reports/figures/*.png
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from talentlens.paths import DATA_DIR, display_path
from talentlens.skills import CANONICAL_SKILLS, SKILL_ALIASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
_THIS_DIR = Path(__file__).resolve().parent
SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)


# Shared with Chapter 5 demo collector and downstream ML chapters (D3 reproducibility).
BUNDLED_DATA_RANDOM_STATE: int = 42


@dataclass
class Config:
    raw_path: Path = field(default_factory=lambda: DATA_DIR / "raw" / "jobs_raw.csv")
    raw_demo_path: Path = field(default_factory=lambda: DATA_DIR / "raw" / "jobs_raw.demo.csv")
    clean_path: Path = field(default_factory=lambda: DATA_DIR / "clean" / "jobs_clean.csv")
    clean_demo_path: Path = field(
        default_factory=lambda: DATA_DIR / "clean" / "jobs_clean.demo.csv"
    )
    figures_dir: Path = field(default_factory=lambda: _THIS_DIR / "reports" / "figures")
    reports_dir: Path = field(default_factory=lambda: _THIS_DIR / "reports")
    salary_min_floor: float = 200_000
    salary_max_ceiling: float = 50_000_000
    min_description_len: int = 50
    random_state: int = BUNDLED_DATA_RANDOM_STATE


def drop_invalid_rows(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop rows missing required fields or with too-short descriptions."""
    before = len(df)
    df = df.dropna(subset=["title", "company", "description"])
    df = df[df["title"].str.strip().str.len() > 0]
    df = df[df["company"].str.strip().str.len() > 0]
    df = df[df["description"].str.len() >= cfg.min_description_len]
    logger.info(f"drop_invalid_rows: -{before - len(df):,} rows -> {len(df):,} remain")
    return df.copy()


def clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise job titles: strip noise, title-case, preserve acronyms."""

    def _clean(t: str) -> str:
        if not isinstance(t, str):
            return ""
        t = re.sub(r"\(.*?\)", "", t)
        t = re.sub(r"\[.*?\]", "", t)
        t = re.sub(r"[|/\\].*$", "", t)
        t = re.sub(r"\s+", " ", t).strip().title()
        for acro in ["AI", "ML", "NLP", "SQL", "CV", "API", "AWS", "GCP", "ETL"]:
            t = re.sub(rf"\b{acro.title()}\b", acro, t)
        return t

    df = df.copy()
    df["title"] = df["title"].apply(_clean)
    return df


def impute_salary(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clip outliers, group-median impute missing salary, add salary_disclosed flag."""
    df = df.copy()
    df["salary_disclosed"] = df["salary_min"].notna() | df["salary_max"].notna()
    df["salary_imputed"] = False
    for col in ["salary_min", "salary_max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < cfg.salary_min_floor, col] = np.nan
            df.loc[df[col] > cfg.salary_max_ceiling, col] = np.nan
    group_cols = ["source"] + (["role_label"] if "role_label" in df.columns else [])
    for col in ["salary_min", "salary_max"]:
        if col in df.columns:
            gmed = df.groupby(group_cols, observed=True)[col].transform("median")
            df[col] = df[col].fillna(gmed).fillna(df[col].median())
    if {"salary_min", "salary_max"}.issubset(df.columns):
        mask = df["salary_max"] < df["salary_min"]
        df.loc[mask, "salary_max"] = df.loc[mask, "salary_min"] * 1.3
    imputed_mask = ~df["salary_disclosed"]
    df.loc[imputed_mask, "salary_imputed"] = True
    n = int(imputed_mask.sum())
    logger.info(f"impute_salary: {n:,} rows imputed ({n / len(df) * 100:.1f}%)")
    return df


def extract_skills(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build normalised pipe-separated skills_normalised column."""
    df = df.copy()
    alias_map = {a.lower(): c for a, c in SKILL_ALIASES.items()}

    def _skills(row: pd.Series) -> str:
        found: set[str] = set()
        raw = str(row.get("skills_raw", "") or "")
        for s in re.split(r"[,|;\s]+", raw):
            s = s.strip().lower()
            if s in alias_map:
                found.add(alias_map[s])
            else:
                for c in CANONICAL_SKILLS:
                    if s == c.lower():
                        found.add(c)
        desc = str(row.get("description", "")).lower()
        for c in CANONICAL_SKILLS:
            if c.lower() in desc:
                found.add(c)
        return "|".join(sorted(found))

    df["skills_normalised"] = df.apply(_skills, axis=1)
    pct = (df["skills_normalised"].str.len() > 0).mean() * 100
    logger.info(f"extract_skills: {pct:.1f}% rows have skills")
    return df


def normalise_remote_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure is_remote is a clean bool column."""
    df = df.copy()
    if "is_remote" not in df.columns:
        df["is_remote"] = False

    def _b(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "remote")
        return bool(v) if isinstance(v, (int, float)) else False

    df["is_remote"] = df["is_remote"].apply(_b)
    if "city" in df.columns:
        df.loc[df["city"].str.lower().str.strip() == "remote", "is_remote"] = True
    logger.info(f"normalise_remote: {df['is_remote'].sum():,} remote roles")
    return df


def add_salary_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add salary_band: junior / mid / senior / lead_plus."""
    df = df.copy()
    bins = [0, 800_000, 1_500_000, 3_000_000, float("inf")]
    labels = ["junior", "mid", "senior", "lead_plus"]
    df["salary_band"] = pd.cut(
        df["salary_min"].fillna(df["salary_min"].median()),
        bins=bins,
        labels=labels,
        right=True,
    ).astype(str)
    return df


def derive_salary_annual_inr(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``salary_annual_inr`` as the midpoint of (salary_min, salary_max).

    Chapters 7, 17, and 19 want a single numeric salary value per row for
    analysis and ranking. The midpoint is the simplest defensible choice when
    only a range is available; where one bound is missing we use the other;
    where both are missing the value is NaN (and the row was already flagged
    by ``salary_imputed`` / ``salary_disclosed`` in :func:`impute_salary`).

    The unit is INR per year. We do not currency-convert here - source-level
    conversion happens in Chapter 5 before this chapter sees the data.

    Args:
        df: DataFrame after :func:`impute_salary`.

    Returns:
        DataFrame with a new ``salary_annual_inr`` column.
    """
    df = df.copy()
    smin = df["salary_min"]
    smax = df["salary_max"]
    midpoint = (smin + smax) / 2.0
    midpoint = midpoint.fillna(smin).fillna(smax)
    df["salary_annual_inr"] = midpoint
    n_present = df["salary_annual_inr"].notna().sum()
    logger.info(
        f"derive_salary_annual_inr: {n_present:,} / {len(df):,} rows have a value "
        f"({n_present / len(df) * 100:.1f}%)"
    )
    return df


_ROLE_TITLE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AI Engineer", ("ai engineer", "llm", "genai", "gen ai", "applied scientist")),
    ("ML Engineer", ("ml engineer", "machine learning engineer", "mlops")),
    ("Data Scientist", ("data scientist", "research scientist")),
    ("Data Engineer", ("data engineer", "analytics engineer", "etl")),
    ("Data Analyst", ("data analyst", "business analyst", "bi analyst")),
)


def derive_role_category(df: pd.DataFrame) -> pd.DataFrame:
    """Derive role_category from title via substring patterns.

    Default is "Other" - titles that don't match any canonical pattern stay
    as "Other" rather than being lumped into a canonical role. This matters
    on real scraped data, where Adzuna's broad "IT jobs" category includes
    everything from SAP consultants to medical-device engineers. Lumping them
    into "Data Scientist" would poison any downstream classifier.

    The five canonical TalentLens roles are AI Engineer, ML Engineer, Data
    Scientist, Data Engineer, Data Analyst. Anything else → "Other".

    Args:
        df: DataFrame with a ``title`` column.

    Returns:
        DataFrame with new ``role_label`` and ``role_category`` columns
        (same values; both names exist for backwards compatibility).
    """
    df = df.copy()
    titles = df["title"].fillna("").astype(str).str.lower()
    role = pd.Series(["Other"] * len(df), index=df.index, name="role_category")

    for category, patterns in _ROLE_TITLE_PATTERNS:
        for pattern in patterns:
            mask = titles.str.contains(pattern, case=False, na=False)
            role.loc[mask & (role == "Other")] = category

    df["role_category"] = role
    df["role_label"] = role
    counts = role.value_counts().to_dict()
    logger.info(f"derive_role_category: distribution = {counts}")
    return df


def final_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup on title + company + city fingerprint."""
    df = df.copy()
    if "fingerprint" not in df.columns:
        fp_key = (
            df["title"].str.lower().fillna("")
            + "|"
            + df["company"].str.lower().fillna("")
            + "|"
            + df.get("city", pd.Series([""] * len(df))).str.lower().fillna("")
        )
        df["fingerprint"] = fp_key.map(lambda s: hashlib.md5(s.encode()).hexdigest())
    before = len(df)
    fp = (
        df["title"].str.lower().fillna("")
        + "|"
        + df["company"].str.lower().fillna("")
        + "|"
        + df.get("city", pd.Series([""] * len(df))).str.lower().fillna("")
    )
    df = df[~fp.duplicated(keep="first")].copy()
    logger.info(f"final_dedup: -{before - len(df):,} -> {len(df):,}")
    return df


def _generate_demo_raw(cfg: Config | None = None) -> pd.DataFrame:
    seed = cfg.random_state if cfg is not None else BUNDLED_DATA_RANDOM_STATE
    rng = np.random.default_rng(seed)
    roles = [
        ("Senior NLP Engineer", "Python,PyTorch,NLP,RAG,FastAPI", 3_000_000),
        ("ML Engineer", "Python,scikit-learn,MLflow,XGBoost", 2_200_000),
        ("AI Engineer", "Python,LLMs,RAG,pgvector,FastAPI", 3_200_000),
        ("Data Scientist", "Python,SQL,Statistics,pandas", 1_800_000),
        ("Data Engineer", "Python,SQL,Spark,dbt,Airflow", 1_800_000),
        ("Data Analyst", "SQL,Tableau,Excel,Power BI", 900_000),
    ]
    companies = [
        "Nimbus Fintech",
        "Kestrel Commerce",
        "Monsoon Payments",
        "Banyan Health",
        "Indigo Logistics",
        "Remote Startup",
    ]
    cities = ["Bangalore", "Mumbai", "Hyderabad", "Delhi NCR", "Remote"]
    rows = []
    for i in range(600):
        title, skills, base = roles[i % len(roles)]
        scale = 0.7 + rng.random() * 0.6
        sal = int(base * scale) if rng.random() > 0.2 else None
        rows.append(
            {
                "job_id": f"demo_{i:04d}",
                "source": "demo",
                "title": title,
                "company": companies[i % len(companies)],
                "city": rng.choice(cities),
                "country": "IN",
                "description": (
                    f"We need a {title} with {skills.split(',')[0]} and "
                    f"{skills.split(',')[1]} skills. Production experience required. "
                    f"Competitive salary and equity."
                ),
                "skills_raw": skills if rng.random() > 0.3 else "",
                "salary_min": sal,
                "salary_max": int(sal * 1.35) if sal else None,
                "currency": "INR",
                "is_remote": rng.random() > 0.6,
                "posted_date": "2026-01-01",
                "url": f"https://example.com/{i}",
            }
        )
    rows.append(
        {
            "job_id": "bad_001",
            "source": "demo",
            "title": "",
            "company": "X",
            "description": "short",
            "skills_raw": "",
            "salary_min": None,
            "salary_max": None,
            "currency": "INR",
            "is_remote": False,
            "posted_date": "",
            "url": "",
            "city": "",
            "country": "",
        }
    )
    return pd.DataFrame(rows)


def plot_field_coverage(df_before: pd.DataFrame, df_after: pd.DataFrame, cfg: Config) -> None:
    key_cols = [
        "title",
        "company",
        "description",
        "salary_min",
        "salary_max",
        "salary_annual_inr",
        "role_category",
        "is_remote",
        "skills_normalised",
        "city",
    ]
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    for df, suffix, color in [(df_before, "before", "#F44336"), (df_after, "after", "#4CAF50")]:
        cov = {
            c: (df[c].notna() & (df[c].astype(str).str.strip() != "")).mean() * 100
            for c in key_cols
            if c in df.columns
        }
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(
            list(cov.keys()), list(cov.values()), color=color, alpha=0.8, edgecolor="white"
        )
        for bar, val in zip(bars, cov.values()):
            ax.text(
                val + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%",
                va="center",
                fontsize=9,
            )
        ax.set_xlim(0, 110)
        ax.axvline(90, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("% rows with non-null value", fontsize=11)
        ax.set_title(f"Field Coverage — {suffix.title()} Cleaning", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = cfg.figures_dir / f"ch06_missing_values_{suffix}.png"
        plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out}")


def plot_salary_distribution(df: pd.DataFrame, cfg: Config) -> None:
    salary = df["salary_min"].dropna() / 100_000
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(salary, bins=35, color="#2196F3", edgecolor="white", alpha=0.85)
    for lakh, lbl in [(8, "8L"), (15, "15L"), (30, "30L")]:
        ax.axvline(lakh, color="#F44336", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(lakh + 0.3, ax.get_ylim()[1] * 0.85, f"\u20b9{lbl}", fontsize=8.5, color="#F44336")
    ax.set_xlabel("Annual Salary (\u20b9 Lakhs)", fontsize=12)
    ax.set_ylabel("Job Postings", fontsize=12)
    ax.set_title("Salary Distribution \u2014 Cleaned Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = cfg.figures_dir / "ch06_salary_distribution_cleaned.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def write_cleaning_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cfg: Config,
    *,
    output_path: Path,
) -> Path:
    n_imp = (~df_after.get("salary_disclosed", pd.Series([True] * len(df_after)))).sum()
    remote = df_after.get("is_remote", pd.Series([False] * len(df_after))).sum()
    skills_pct = (
        (df_after["skills_normalised"].str.len() > 0).mean() * 100
        if "skills_normalised" in df_after.columns
        else 0
    )
    report = (
        f"# TalentLens Data Cleaning Report\n\n"
        f"| Stage | Rows |\n|-------|------|\n"
        f"| Raw | {len(df_before):,} |\n"
        f"| Clean | {len(df_after):,} |\n"
        f"| Removed | {len(df_before) - len(df_after):,} |\n\n"
        f"- Salary imputed: {n_imp:,} rows\n"
        f"- Remote roles: {remote:,} ({remote / len(df_after) * 100:.1f}%)\n"
        f"- Rows with skills: {skills_pct:.1f}%\n\n"
        f"Output: `{display_path(output_path)}`\n"
    )
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / "cleaning_report.md"
    out.write_text(report, encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


def run_cleaning_pipeline(
    cfg: Config, *, overwrite: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    if overwrite:
        if cfg.raw_path.exists():
            logger.info(f"Loading {cfg.raw_path}")
            df = pd.read_csv(cfg.raw_path)
        else:
            logger.warning("Raw data not found — using demo data")
            df = _generate_demo_raw(cfg)
        output_path = cfg.clean_path
    else:
        if cfg.raw_demo_path.exists():
            logger.info(f"Loading {cfg.raw_demo_path}")
            df = pd.read_csv(cfg.raw_demo_path)
        else:
            logger.warning("Demo raw not found — using synthetic demo data")
            df = _generate_demo_raw(cfg)
        output_path = cfg.clean_demo_path
        if cfg.clean_path.exists():
            logger.info(
                f"Bundled dataset preserved at {cfg.clean_path} "
                f"(book quoted numbers come from this file; demo output -> {output_path})."
            )

    df_before = df.copy()
    df = drop_invalid_rows(df, cfg)
    df = clean_titles(df)
    df = derive_role_category(df)
    df = impute_salary(df, cfg)
    df = extract_skills(df, cfg)
    df = normalise_remote_flag(df)
    df = add_salary_band(df)
    df = derive_salary_annual_inr(df)
    df = final_dedup(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path} ({len(df):,} rows)")
    return df_before, df, output_path


def main() -> None:
    import sys

    overwrite = "--overwrite" in sys.argv
    cfg = Config()
    logger.info("=" * 60)
    logger.info("  CHAPTER 6: DATA CLEANING AND PREPROCESSING")
    logger.info("  TalentLens — Raw -> Clean Job Postings")
    logger.info("=" * 60)
    df_before, df_after, output_path = run_cleaning_pipeline(cfg, overwrite=overwrite)
    plot_field_coverage(df_before, df_after, cfg)
    plot_salary_distribution(df_after, cfg)
    write_cleaning_report(df_before, df_after, cfg, output_path=output_path)
    logger.info("  CHAPTER 6 COMPLETE")
    logger.info(f"  Output: {output_path} ({len(df_after):,} rows)")
    if not overwrite and cfg.clean_path.exists():
        logger.info(f"  Bundled contract unchanged: {cfg.clean_path}")
    logger.info("  Next: python book/ch07/ch07_exploratory_data_analysis.py")


if __name__ == "__main__":
    main()
