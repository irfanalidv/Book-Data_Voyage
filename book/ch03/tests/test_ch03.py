"""Chapter 3 tests - synthetic salary statistics and figure generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))

from ch03_statistics import (  # noqa: E402
    generate_synthetic_salaries,
    main,
)

from talentlens.paths import find_repo_root  # noqa: E402


def test_synthetic_salary_dataset_is_right_skewed() -> None:
    series = generate_synthetic_salaries(500)
    assert series.mean() > series.median() * 1.05


def test_log_transform_reduces_skew() -> None:
    series = generate_synthetic_salaries(500)
    raw_skew = abs(stats.skew(series, bias=False))
    log_skew = abs(stats.skew(np.log1p(series), bias=False))
    assert log_skew < raw_skew


def test_figures_are_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ch03_statistics as mod

    out = tmp_path / "figures"
    monkeypatch.setattr(mod, "FIGURES_DIR", out)
    assert main() == 0
    for name in (
        "ch03_salary_histogram.png",
        "ch03_salary_boxplot.png",
        "ch03_skew_demonstration.png",
    ):
        assert (out / name).is_file()
        assert (out / name).stat().st_size > 1000


def test_main_script_exits_zero() -> None:
    repo = find_repo_root()
    script = repo / "book" / "ch03" / "ch03_statistics.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
