"""Tests for Chapter 1: The Data Science Landscape."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the chapter module importable
CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))

from ch01_data_science_landscape import (  # noqa: E402
    DEPENDENCIES,
    SAMPLE_ROLE_COUNTS,
    build_sample_role_dataframe,
    check_dependency,
    check_python_version,
    plot_role_distribution,
)


def test_python_version_check_returns_string_and_bool() -> None:
    """The version check returns a (bool, str) tuple."""
    ok, version = check_python_version()
    assert isinstance(ok, bool)
    assert isinstance(version, str)
    assert "." in version  # e.g. "3.11.7"


def test_dependency_check_handles_installed_package() -> None:
    """pandas is installed (we just imported it), so it should report installed."""
    status = check_dependency("pandas")
    assert status.installed is True
    assert status.version is not None
    assert status.name == "pandas"


def test_dependency_check_handles_missing_package() -> None:
    """A package that doesn't exist should report not installed without raising."""
    # Temporarily add a fake package to DEPENDENCIES for the test
    DEPENDENCIES["definitely_not_a_real_package_xyz123"] = (False, 99)
    try:
        status = check_dependency("definitely_not_a_real_package_xyz123")
        assert status.installed is False
        assert status.version is None
    finally:
        del DEPENDENCIES["definitely_not_a_real_package_xyz123"]


def test_sample_role_dataframe_has_five_roles() -> None:
    """The chapter introduces five roles; the sample DataFrame must reflect that."""
    df = build_sample_role_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert set(df["role"]) == set(SAMPLE_ROLE_COUNTS.keys())
    # DataFrame must be sorted descending by count
    assert df["count"].is_monotonic_decreasing


def test_plot_role_distribution_writes_file(tmp_path: Path) -> None:
    """The chart-saving function actually writes a PNG to the requested path."""
    df = build_sample_role_dataframe()
    output_path = tmp_path / "test_chart.png"
    plot_role_distribution(df, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
