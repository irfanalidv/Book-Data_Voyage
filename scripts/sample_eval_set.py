"""Sample 30 stratified postings from jobs_clean.csv for hand-labelling.

Stratification dimensions (rough; adjusted to what the dataset
actually has):
  - description-only (skills_raw empty or near-empty)
  - skills_raw-rich (5+ items in skills_raw)
  - multi-paragraph descriptions (likely complex)
  - short descriptions

Output: book/ch13/data/eval_set.jsonl with empty 'skills_verified'
fields ready for the author to fill in by hand.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from talentlens.paths import jobs_clean_path

EVAL_SET_PATH = Path("book/ch13/data/eval_set.jsonl")


def _adzuna_url(job_id: str) -> str:
    """Link to the original listing, as Adzuna's API terms require for displayed ads."""
    return f"https://www.adzuna.in/details/{job_id.split('_', 1)[1]}"


DEFAULT_ROWS = 30
RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Number of rows to sample (default: {DEFAULT_ROWS})",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    target_rows = args.n

    data_path = jobs_clean_path()
    df = pd.read_csv(data_path)
    random.seed(args.seed)
    print(f"Input: {data_path} ({len(df):,} rows)")

    # Stratification proxies - adjust if the actual column names
    # differ from what was expected.
    df["desc_len"] = df["description"].fillna("").astype(str).str.len()
    df["skills_token_count"] = df["skills_normalised"].fillna("").astype(str).str.count(r"\|") + 1
    df.loc[df["skills_normalised"].fillna("") == "", "skills_token_count"] = 0

    strata = {
        "description_only": df[df["skills_token_count"] == 0],
        "skills_sparse": df[(df["skills_token_count"] >= 1) & (df["skills_token_count"] < 3)],
        "skills_medium": df[(df["skills_token_count"] >= 3) & (df["skills_token_count"] < 6)],
        "skills_rich": df[df["skills_token_count"] >= 6],
        "short_desc": df[df["desc_len"] < 200],
        "long_desc": df[df["desc_len"] >= 500],
    }

    per_stratum = max(1, target_rows // len(strata))
    selected_ids: set[str] = set()
    sampled_rows: list[dict] = []

    for name, sub in strata.items():
        available = [r for r in sub.to_dict("records") if r["job_id"] not in selected_ids]
        sample = random.sample(available, min(per_stratum, len(available)))
        for row in sample:
            selected_ids.add(row["job_id"])
            sampled_rows.append(
                {
                    "job_id": row["job_id"],
                    "source": "Jobs by Adzuna",
                    "source_url": _adzuna_url(row["job_id"]),
                    "stratum": name,
                    "title": row.get("title", ""),
                    "description": row.get("description", "")[:2000],
                    "skills_raw": row.get("skills_normalised", ""),
                    "skills_verified": None,
                    "notes": "",
                }
            )

    if len(sampled_rows) < target_rows:
        pool = [r for r in df.to_dict("records") if r["job_id"] not in selected_ids]
        extras = random.sample(pool, min(target_rows - len(sampled_rows), len(pool)))
        for row in extras:
            selected_ids.add(row["job_id"])
            sampled_rows.append(
                {
                    "job_id": row["job_id"],
                    "source": "Jobs by Adzuna",
                    "source_url": _adzuna_url(row["job_id"]),
                    "stratum": "additional",
                    "title": row.get("title", ""),
                    "description": row.get("description", "")[:2000],
                    "skills_raw": row.get("skills_normalised", ""),
                    "skills_verified": None,
                    "notes": "",
                }
            )

    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_SET_PATH.open("w") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled_rows)} rows to {EVAL_SET_PATH}")
    for name in strata:
        count = sum(1 for r in sampled_rows if r["stratum"] == name)
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
