"""
Chapter 1: The Data Science Landscape
Data Voyage - Building TalentLens

TalentLens milestone: verify the reader's environment is ready, and
produce one simple chart to confirm the matplotlib/pandas toolchain works.

Run:
    python book/ch01/ch01_data_science_landscape.py

Outputs:
    book/ch01/reports/figures/ch01_role_distribution.png
    stdout: environment-readiness report
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAPTER_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = CHAPTER_ROOT / "reports" / "figures"

MIN_PYTHON = (3, 11)

# Dependency name → (is_required_now, chapter_first_needed)
# "required_now" means Chapter 1 itself needs it; everything else is checked
# but not blocking.
DEPENDENCIES: dict[str, tuple[bool, int]] = {
    "matplotlib": (True, 1),
    "pandas": (True, 1),
    "numpy": (True, 1),
    "scikit-learn": (False, 9),
    "sentence_transformers": (False, 13),
    "fastapi": (False, 19),
    "pydantic": (False, 19),
    "openai": (False, 17),
}

# The five roles from the chapter, with plausible counts for a tiny sample.
# Real data starts arriving in Chapter 5; this is enough to verify the
# plotting toolchain.
SAMPLE_ROLE_COUNTS = {
    "Data Analyst": 142,
    "Data Scientist": 188,
    "Data Engineer": 95,
    "ML Engineer": 215,
    "AI Engineer": 167,
}


@dataclass
class DependencyStatus:
    """Status of a single dependency."""

    name: str
    installed: bool
    version: str | None
    required_now: bool
    chapter_first_needed: int


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------


def check_python_version() -> tuple[bool, str]:
    """Verify the running Python version meets the book's minimum.

    Returns:
        (ok, version_string)
    """
    actual = sys.version_info
    version_string = f"{actual.major}.{actual.minor}.{actual.micro}"
    ok = actual >= MIN_PYTHON
    return ok, version_string


def check_dependency(name: str) -> DependencyStatus:
    """Try to import a dependency and report its version.

    Args:
        name: pip package name (may differ from import name for some packages).

    Returns:
        DependencyStatus with installation and version info.
    """
    required_now, chapter = DEPENDENCIES[name]
    # Some packages have pip name != import name; handle the common cases.
    import_name = {
        "scikit-learn": "sklearn",
    }.get(name, name)

    try:
        importlib.import_module(import_name)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "(unknown)"
        return DependencyStatus(name, True, version, required_now, chapter)
    except ImportError:
        return DependencyStatus(name, False, None, required_now, chapter)


def check_all_dependencies() -> list[DependencyStatus]:
    """Check every dependency listed in DEPENDENCIES."""
    return [check_dependency(name) for name in DEPENDENCIES]


# ---------------------------------------------------------------------------
# The demo chart
# ---------------------------------------------------------------------------


def build_sample_role_dataframe() -> pd.DataFrame:
    """Build the small sample dataset for the role-distribution chart.

    This is not real data - real job postings arrive in Chapter 5. The
    purpose here is to confirm pandas and matplotlib are working.

    Returns:
        DataFrame with columns ['role', 'count'].
    """
    df = pd.DataFrame(
        {
            "role": list(SAMPLE_ROLE_COUNTS.keys()),
            "count": list(SAMPLE_ROLE_COUNTS.values()),
        }
    )
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def plot_role_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Render a simple bar chart of role frequency.

    Deliberately minimal - Chapter 7 introduces the book's full
    visualisation standards. This chart exists only to verify that
    matplotlib renders and file output works.

    Args:
        df: DataFrame with 'role' and 'count' columns.
        output_path: Where to save the PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["role"], df["count"], color="#4a7cb8", edgecolor="#2d4f7d")

    ax.set_title(
        "The Five Roles in Modern Data and AI",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Postings in sample")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=15)

    # Annotate counts on top of each bar
    for bar, count in zip(bars, df["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_dependency_line(status: DependencyStatus) -> str:
    """Format a single dependency status line for the report."""
    name_col = f"{status.name}:".ljust(25)
    if status.installed:
        return f"  {name_col}{status.version}  ✓"
    elif status.required_now:
        return f"  {name_col}NOT INSTALLED  ✗  REQUIRED NOW — run `pip install {status.name}`"
    else:
        return f"  {name_col}not installed  (needed from Chapter {status.chapter_first_needed})"


def print_environment_report(
    python_ok: bool,
    python_version: str,
    deps: list[DependencyStatus],
) -> bool:
    """Print the environment-readiness report. Returns True if ready."""
    logger.info("=" * 60)
    logger.info("Data Voyage — Chapter 1 environment check")
    logger.info("=" * 60)
    logger.info("")

    py_marker = "✓" if python_ok else "✗"
    logger.info(f"  Python version:          {python_version}  {py_marker}")
    if not python_ok:
        logger.info(
            f"  REQUIRED: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ "
            "(install via pyenv / official installer)"
        )
    logger.info("")

    for dep in deps:
        logger.info(format_dependency_line(dep))

    logger.info("")
    required_missing = [d for d in deps if d.required_now and not d.installed]
    ready = python_ok and not required_missing

    if ready:
        logger.info("You are ready for Chapter 2.")
    else:
        logger.info("Environment is NOT ready. Fix the items above and re-run.")

    logger.info("")
    return ready


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the chapter's environment check and produce the sample chart.

    Returns:
        Exit code: 0 if environment is ready, 1 otherwise.
    """
    python_ok, python_version = check_python_version()
    deps = check_all_dependencies()
    ready = print_environment_report(python_ok, python_version, deps)

    # The chart only requires pandas + matplotlib + numpy. If those are
    # installed, we generate the chart regardless of whether other
    # dependencies are present.
    core_installed = all(d.installed for d in deps if d.name in {"matplotlib", "pandas", "numpy"})

    if core_installed:
        df = build_sample_role_dataframe()
        output_path = FIGURES_DIR / "ch01_role_distribution.png"
        plot_role_distribution(df, output_path)
        logger.info(f"Chart saved: {output_path.relative_to(CHAPTER_ROOT.parents[1])}")
        logger.info("")

    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
