"""
Chapter 15: Scaling Python - When You Outgrow Pandas
Data Voyage - Building TalentLens

TalentLens milestone: our job postings dataset grows month over month.
This chapter covers the scaling path: chunked pandas, polars for speed,
dask for parallelism - and an honest answer to "do I need Spark?"

Run: python book/ch15/ch15_scaling_python.py

Outputs:
    book/ch15/reports/figures/ch15_benchmark_comparison.png
    book/ch15/reports/figures/ch15_memory_usage.png
    book/ch15/reports/scaling_report.md
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from talentlens.paths import jobs_clean_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)


@dataclass
class Config:
    clean_path: Path = field(default_factory=jobs_clean_path)
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"
    benchmark_sizes: list = None

    def __post_init__(self):
        if self.benchmark_sizes is None:
            self.benchmark_sizes = [1_000, 5_000, 10_000, 50_000, 100_000]


def generate_synthetic_large(n: int) -> pd.DataFrame:
    """Generate a synthetic dataset of n rows matching TalentLens schema."""
    rng = np.random.default_rng(15)
    titles = ["ML Engineer", "AI Engineer", "Data Scientist", "Data Engineer", "Data Analyst"]
    phrases = [
        "build production machine learning pipelines",
        "design experiments and dashboards",
        "ship LLM features behind APIs",
        "maintain batch and streaming ETL",
    ]
    tools = [
        "Python and SQL",
        "PyTorch and FastAPI",
        "Spark and Airflow",
        "pandas and Tableau",
        "Docker and Kubernetes",
    ]
    return pd.DataFrame(
        {
            "title": [titles[i % len(titles)] for i in range(n)],
            "salary_min": rng.integers(800_000, 4_000_000, n).astype(float),
            "is_remote": rng.random(n) > 0.6,
            "skill_count": rng.integers(2, 12, n),
            # Unique per row, like real postings - identical strings would make the
            # category-dtype saving look far better than it is on real data.
            "description": [
                f"Posting {i}: {phrases[i % len(phrases)]} with {tools[(i * 7) % len(tools)]}."
                for i in range(n)
            ],
            "company": [f"Company_{i % 50}" for i in range(n)],
        }
    )


def benchmark_pandas_operations(df: pd.DataFrame) -> dict[str, float]:
    """Time common pandas operations on a DataFrame.

    Args:
        df: DataFrame to benchmark.

    Returns:
        Dict of operation name to elapsed seconds.
    """
    results = {}

    t = time.perf_counter()
    _ = df.groupby("title")["salary_min"].median()
    results["groupby_median"] = time.perf_counter() - t

    t = time.perf_counter()
    _ = df[df["is_remote"] & (df["salary_min"] > 2_000_000)]
    results["filter"] = time.perf_counter() - t

    t = time.perf_counter()
    _ = df.sort_values("salary_min", ascending=False)
    results["sort"] = time.perf_counter() - t

    t = time.perf_counter()
    _ = df["description"].str.contains("python", case=False)
    results["str_contains"] = time.perf_counter() - t

    # The same arithmetic two ways: a Python function per row vs one vectorised op.
    t = time.perf_counter()
    _ = df["salary_min"].apply(lambda x: x * 1.1 if x > 1_500_000 else x)
    results["apply_lambda"] = time.perf_counter() - t

    t = time.perf_counter()
    _ = df["salary_min"].where(df["salary_min"] <= 1_500_000, df["salary_min"] * 1.1)
    results["vectorised"] = time.perf_counter() - t

    return results


def benchmark_chunked_read(filepath: Path, chunksize: int = 1000) -> float:
    """Benchmark reading a CSV in chunks vs all at once.

    Chunked reading is the first scaling technique - process large files
    without loading everything into RAM.

    Args:
        filepath: Path to CSV file.
        chunksize: Rows per chunk.

    Returns:
        Elapsed seconds for full chunked read.
    """
    if not filepath.exists():
        return 0.0
    t = time.perf_counter()
    total = 0
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        total += len(chunk)
    return time.perf_counter() - t


def demonstrate_memory_dtypes(df: pd.DataFrame) -> tuple[float, float]:
    """Show memory savings from optimising dtypes.

    String columns stored as 'object' use far more RAM than 'category'.
    Salary stored as float64 uses 2x the RAM of float32.

    Args:
        df: Original DataFrame.

    Returns:
        Tuple of (original_mb, optimised_mb).
    """
    original_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    df_opt = df.copy()
    for col in df_opt.select_dtypes(include="object").columns:
        n_unique = df_opt[col].nunique()
        if n_unique / len(df_opt) < 0.5:  # less than 50% unique → use category
            df_opt[col] = df_opt[col].astype("category")
    for col in df_opt.select_dtypes(include="float64").columns:
        df_opt[col] = df_opt[col].astype("float32")

    optimised_mb = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)
    savings_pct = (1 - optimised_mb / original_mb) * 100
    logger.info(f"Memory: {original_mb:.1f}MB → {optimised_mb:.1f}MB ({savings_pct:.0f}% saving)")
    return original_mb, optimised_mb


def plot_benchmark_comparison(benchmark_data: dict[int, dict], cfg: Config) -> Path:
    """Line chart: operation time vs dataset size for key pandas operations."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(benchmark_data.keys())
    operations = list(next(iter(benchmark_data.values())).keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#009688"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for op, color in zip(operations, colors):
        times_ms = [benchmark_data[s][op] * 1000 for s in sizes]
        ax.plot(sizes, times_ms, "o-", label=op, color=color, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (rows, log scale)", fontsize=12)
    ax.set_ylabel("Time (ms, log scale)", fontsize=12)
    ax.set_title("Pandas Operation Scaling — Time vs Dataset Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.annotate(
        "Linear scaling: 10x rows = 10x time\n"
        "When this becomes painful, consider polars or dask",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        color="gray",
    )
    plt.tight_layout()
    out = cfg.figures_dir / "ch15_benchmark_comparison.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_memory_usage(
    sizes: list[int], orig_mbs: list[float], opt_mbs: list[float], cfg: Config
) -> Path:
    """Bar chart: original vs optimised memory usage by dataset size."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(sizes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - w / 2,
        orig_mbs,
        w,
        label="Default dtypes (float64, object)",
        color="#F44336",
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar(
        x + w / 2,
        opt_mbs,
        w,
        label="Optimised dtypes (float32, category)",
        color="#4CAF50",
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in sizes], fontsize=10)
    ax.set_xlabel("Dataset size (rows)", fontsize=12)
    ax.set_ylabel("Memory usage (MB)", fontsize=12)
    ax.set_title("Memory Savings from dtype Optimisation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = cfg.figures_dir / "ch15_memory_usage.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def write_scaling_report(benchmark_data: dict, cfg: Config) -> Path:
    """Write scaling recommendations based on benchmark results."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(benchmark_data.keys())
    max_size = max(sizes)
    slowest_op = max(benchmark_data[max_size], key=benchmark_data[max_size].get)
    slowest_ms = benchmark_data[max_size][slowest_op] * 1000

    report = f"""# TalentLens Scaling Report

## Benchmark results at {max_size:,} rows

Slowest operation: `{slowest_op}` at {slowest_ms:.1f}ms

## When to upgrade from pandas

| Row count | Recommendation |
|-----------|---------------|
| < 1M | Pandas is fine. Optimise dtypes for memory. |
| 1M – 10M | Try polars (`pip install polars`). 5-10x faster for most operations. |
| 10M – 100M | Dask for parallelism on one machine. |
| 100M+ | Spark or DuckDB depending on query patterns. |

## Quick wins before changing tools

1. **dtype optimisation**: convert object columns with <50% unique values to `category`.
   Saves 30-70% memory. Zero code changes elsewhere.

2. **Chunked reading**: `pd.read_csv(path, chunksize=10_000)` processes files larger
   than RAM without loading everything at once.

3. **Select only needed columns**: `pd.read_csv(path, usecols=['title', 'salary_min'])`
   is faster and uses less memory than reading everything.

4. **Vectorise instead of apply**: `df['salary_min'] * 1.1` is 100x faster than
   `df['salary_min'].apply(lambda x: x * 1.1)`. Avoid `.apply()` on large DataFrames.

## For TalentLens specifically

> **Note on scale figures below:** Row counts (50k, 500k, 5M) are benchmarks for when each tool starts to matter — not claims about the bundled sample in the repo. The bundled dataset is much smaller; pandas is more than fast enough on it.

At 50k postings (production-scale benchmark), pandas is fast enough.
At 500k postings (after 6 months of live collection), switch str operations to polars.
At 5M postings, use DuckDB for SQL-style aggregations — it's faster than Spark for
analytical queries and runs on a single machine.
"""
    out = cfg.reports_dir / "scaling_report.md"
    out.write_text(report, encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


def main() -> None:
    cfg = Config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  CHAPTER 15: SCALING PYTHON")
    logger.info("  TalentLens — When You Outgrow Pandas")
    logger.info("=" * 60)

    logger.info("\n[1/4] Benchmarking pandas at different scales...")
    benchmark_data: dict[int, dict] = {}
    orig_mbs, opt_mbs = [], []

    for n in cfg.benchmark_sizes:
        logger.info(f"  n={n:,}...")
        df = generate_synthetic_large(n)
        # Best of five runs: the first call pays one-off warm-up costs that
        # would otherwise make small sizes look slower than large ones.
        runs = [benchmark_pandas_operations(df) for _ in range(5)]
        benchmark_data[n] = {op: min(r[op] for r in runs) for op in runs[0]}
        o, p = demonstrate_memory_dtypes(df)
        orig_mbs.append(o)
        opt_mbs.append(p)

    logger.info("\n[2/4] Plotting benchmark results...")
    plot_benchmark_comparison(benchmark_data, cfg)
    plot_memory_usage(cfg.benchmark_sizes, orig_mbs, opt_mbs, cfg)

    logger.info("\n[3/4] Chunked read demo...")
    if cfg.clean_path.exists():
        elapsed = benchmark_chunked_read(cfg.clean_path)
        logger.info(f"  Chunked read of jobs_clean.csv: {elapsed*1000:.1f}ms")

    logger.info("\n[4/4] Writing scaling report...")
    write_scaling_report(benchmark_data, cfg)

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 15 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Figures: {cfg.figures_dir}/")
    logger.info(f"  Report:  {cfg.reports_dir}/scaling_report.md")
    logger.info("\nKey insight: optimise dtypes before changing tools.")
    logger.info("Next: Chapter 16 — RAG and Vector Databases")


if __name__ == "__main__":
    main()
