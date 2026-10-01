# TalentLens Scaling Report

## Benchmark results at 100,000 rows

Slowest operation: `str_contains` at 42.0ms

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
