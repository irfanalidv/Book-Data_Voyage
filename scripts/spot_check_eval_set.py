"""Spot-check 5 LLM-labelled rows from eval_set_llm_labelled.jsonl.

Prints each row's posting + LLM labels, asks the human to confirm,
correct, or note disagreement. The result merges human corrections
back in and writes the final eval_set.jsonl.

Random sample (seed-controlled) for reproducibility. Different
seeds let the author do additional spot-check passes if desired.

Run:
    python scripts/spot_check_eval_set.py

Or with a custom seed / sample size:
    python scripts/spot_check_eval_set.py --seed 42 --n 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

LLM_PATH = Path("book/ch13/data/eval_set_llm_labelled.jsonl")
FINAL_PATH = Path("book/ch13/data/eval_set.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    if not LLM_PATH.exists():
        print(f"ERROR: {LLM_PATH} not found. Run label_eval_set_with_llm.py first.")
        return 1

    with LLM_PATH.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    labelled_indices = [i for i, r in enumerate(rows) if r.get("skills_verified") is not None]
    if len(labelled_indices) < args.n:
        print(f"Only {len(labelled_indices)} labelled rows; can't sample {args.n}.")
        return 1

    random.seed(args.seed)
    sample_indices = sorted(random.sample(labelled_indices, args.n))

    print(f"Spot-checking {args.n} rows (seed={args.seed}).")
    print("For each row, the LLM's labels are shown.")
    print("Type 'ok' to accept, or type a corrected JSON list, or 'skip'.")
    print("Examples: ['Python', 'SQL']  or  []  or  ok  or  skip")
    print()

    n_agreed = 0
    n_corrected = 0
    n_skipped = 0
    corrections: dict[int, list[str]] = {}

    for sample_n, idx in enumerate(sample_indices, 1):
        row = rows[idx]
        print("=" * 70)
        print(f"ROW {sample_n}/{args.n}  (eval-set index {idx}, job_id {row.get('job_id')})")
        print("=" * 70)
        print(f"TITLE: {row.get('title', '')}")
        print("\nDESCRIPTION:")
        print(row.get("description", "")[:1500])
        if len(row.get("description", "")) > 1500:
            print("... [truncated]")
        print(f"\nSKILLS_RAW: {row.get('skills_raw', '(empty)')}")
        print(f"\nLLM LABELS: {row['skills_verified']}")
        print(f"LLM NOTES:  {row.get('notes', '(none)')}")
        print()

        while True:
            response = input("Your verdict (ok / [...] / skip): ").strip()
            if response == "ok":
                n_agreed += 1
                break
            if response == "skip":
                n_skipped += 1
                break
            try:
                new_labels = json.loads(response)
                if not isinstance(new_labels, list):
                    print("  Need a JSON list, e.g. ['Python', 'SQL']")
                    continue
                if not all(isinstance(s, str) for s in new_labels):
                    print("  All items must be strings.")
                    continue
                corrections[idx] = new_labels
                n_corrected += 1
                print(f"  Recorded correction: {new_labels}")
                break
            except json.JSONDecodeError:
                print("  Could not parse — type 'ok', 'skip', or valid JSON list.")
        print()

    for idx, labels in corrections.items():
        rows[idx]["skills_verified"] = labels
        existing = rows[idx].get("notes", "")
        human_note = "[HUMAN] Corrected during spot-check."
        rows[idx]["notes"] = (existing + " | " if existing else "") + human_note

    with FINAL_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 70)
    print("Spot-check complete.")
    print(f"  Agreed:    {n_agreed}/{args.n}")
    print(f"  Corrected: {n_corrected}/{args.n}")
    print(f"  Skipped:   {n_skipped}/{args.n}")
    print(f"  Wrote {len(rows)} rows to {FINAL_PATH}")
    print("\nRecord this agreement rate in the chapter prose:")
    print(f"  human-LLM agreement on {args.n}-row spot-check: {n_agreed}/{args.n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
