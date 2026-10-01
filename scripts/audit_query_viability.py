"""Audit whether ch18's planned test queries are answerable on the bundled jobs_clean.csv.

For each query the chapter executable will run, check whether the
data has the rows needed to answer it. Outputs a viability table
that the chapter's report references. Run at iteration 0 and
again whenever the dataset changes.

The audit doesn't use the agent - it does direct dataframe queries
that mimic what the agent's tools should find.
"""

from __future__ import annotations

import pandas as pd

from talentlens.paths import display_path, jobs_clean_path

QUERIES = [
    {
        "id": "ml_engineer_simple",
        "query": "Find ML Engineer postings",
        "viability_check": lambda df: (df["role_category"] == "ML Engineer").sum(),
        "purpose": "Simple 1-2 tool-call success path",
    },
    {
        "id": "ml_engineer_classify_chain",
        "query": "Find ML Engineer postings, then classify each for actual role match",
        "viability_check": lambda df: (df["role_category"] == "ML Engineer").sum(),
        "purpose": "Multi-step chain: search then classify",
    },
    {
        "id": "no_good_answer",
        "query": "Find postings for Quantum Engineer roles",
        "viability_check": lambda df: df["title"]
        .str.contains("Quantum", case=False, na=False)
        .sum(),
        "purpose": "No-good-answer case — tests graceful empty vs hallucination",
    },
    {
        "id": "compare_top_three",
        "query": "Compare the top 3 AI Engineer postings on salary and required skills",
        "viability_check": lambda df: (df["role_category"] == "AI Engineer").sum(),
        "purpose": "Tests agent's behaviour when query terminology doesn't match data taxonomy",
    },
    {
        "id": "ambiguous",
        "query": "Show me good jobs",
        "viability_check": lambda df: len(df),
        "purpose": "Ambiguous query — tests clarification vs default behaviour",
    },
]


def main() -> int:
    df = pd.read_csv(jobs_clean_path())
    print(f"Dataset: {display_path(jobs_clean_path())} ({len(df)} rows)")
    print(f"role_category distribution: {df['role_category'].value_counts().to_dict()}")
    print()
    print(f"{'Query ID':<28} {'Rows':>6}  Notes")
    print("-" * 80)
    for q in QUERIES:
        n = q["viability_check"](df)
        note = q["purpose"]
        if n == 0:
            note += "  [empty on this data]"
        print(f"{q['id']:<28} {n:>6}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
