"""Tests for Chapter 15: Scaling Python.

Reference: book/ch15/README.md
Source: book/ch15/ch15_scaling_python.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch15"))

from ch15_scaling_python import (  # noqa: E402
    benchmark_chunked_read,
    benchmark_pandas_operations,
    demonstrate_memory_dtypes,
    generate_synthetic_large,
)

from talentlens.paths import jobs_clean_path  # noqa: E402


@pytest.fixture
def scaling_df() -> pd.DataFrame:
    return generate_synthetic_large(5_000)


class TestPandasBenchmarks:
    def test_vectorised_beats_apply(self):
        df = generate_synthetic_large(50_000)
        runs = [benchmark_pandas_operations(df) for _ in range(3)]
        assert min(r["vectorised"] for r in runs) < min(r["apply_lambda"] for r in runs)

    def test_groupby_benchmark_keys(self, scaling_df):
        timings = benchmark_pandas_operations(scaling_df)
        assert set(timings.keys()) == {
            "groupby_median",
            "filter",
            "sort",
            "str_contains",
            "apply_lambda",
            "vectorised",
        }
        assert all(t >= 0 for t in timings.values())

    def test_memory_optimisation_reduces_footprint(self, scaling_df):
        original_mb, optimised_mb = demonstrate_memory_dtypes(scaling_df)
        assert original_mb > 0
        assert optimised_mb < original_mb


class TestChunkedRead:
    def test_chunked_read_matches_full_row_count(self, tmp_path):
        path = jobs_clean_path()
        if not path.exists():
            df = generate_synthetic_large(2_500)
            path = tmp_path / "jobs_sample.csv"
            df.to_csv(path, index=False)
        expected = len(pd.read_csv(path))
        chunked_total = 0
        for chunk in pd.read_csv(path, chunksize=500):
            chunked_total += len(chunk)
        assert chunked_total == expected
        elapsed = benchmark_chunked_read(path, chunksize=500)
        assert elapsed >= 0


@pytest.mark.skip(reason="polars/dask comparison is out of scope; see SCOPE.md and ch15 prose.")
def test_polars_dask_benchmarks_deferred():
    pass
