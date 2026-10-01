"""LLM-based labelling of book/ch13/data/eval_set.jsonl.

Reads each unlabelled row, sends the description + skills_raw + the
full annotation guide as system context, asks for skills_verified as a
JSON list, writes results to eval_set_llm_labelled.jsonl.

Providers: Groq (llama-3.3-70b-versatile) or OpenAI (gpt-4o-mini).

Run:
    python scripts/label_eval_set_with_llm.py --provider openai
    python scripts/label_eval_set_with_llm.py --provider groq

Env:
    OPENAI_API_KEY  (for --provider openai)
    GROQ_API_KEY    (for --provider groq)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

INPUT_PATH = Path("book/ch13/data/eval_set.jsonl")
OUTPUT_PATH = Path("book/ch13/data/eval_set_llm_labelled.jsonl")
ANNOTATION_GUIDE_PATH = Path("book/ch13/data/ANNOTATION_GUIDE.md")
PROVIDERS = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "sleep": 0.5,
        "env": "GROQ_API_KEY",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "sleep": 0.05,
        "env": "OPENAI_API_KEY",
    },
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _build_system_prompt(annotation_guide: str, canonical_skills: tuple) -> str:
    return f"""You are labelling job postings for skill extraction. Follow
these rules exactly:

{annotation_guide}

The canonical skills you may use as labels (case-sensitive, use
exactly these names — no variations):

{", ".join(sorted(canonical_skills))}

For each posting you receive, respond with a JSON object of the form:

{{"skills_verified": ["SkillName1", "SkillName2"], "notes": "any judgement calls"}}

The "skills_verified" list MUST contain only names from the canonical
list above. Aliases (e.g. "k8s", "Postgres") in the posting should
map to their canonical form ("Kubernetes", "PostgreSQL") in the
output. If a skill is mentioned but not in the canonical list, do
NOT include it.

Respond with the JSON object only. No prose, no preamble, no code
fences."""


def _make_client(provider: str) -> tuple[Any, str, float]:
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["env"])
    if not api_key:
        raise SystemExit(f"{cfg['env']} not set. Add it to .env or export it.")
    if provider == "openai":
        from openai import OpenAI

        return OpenAI(api_key=api_key), cfg["model"], cfg["sleep"]
    import groq

    return groq.Groq(api_key=api_key), cfg["model"], cfg["sleep"]


def _label_one_row(client: Any, model: str, system_prompt: str, row: dict) -> dict:
    user_msg = (
        f"Posting title: {row.get('title', '(no title)')}\n\n"
        f"Description:\n{row.get('description', '')[:2000]}\n\n"
        f"Skills field (structured):\n{row.get('skills_raw', '(empty)')}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDERS),
        default="groq",
        help="LLM provider (default: groq)",
    )
    args = parser.parse_args()

    from talentlens.skills import CANONICAL_SKILLS

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}")
        return 1
    if not ANNOTATION_GUIDE_PATH.exists():
        logger.error(f"Annotation guide not found: {ANNOTATION_GUIDE_PATH}")
        return 1

    client, model, rate_sleep = _make_client(args.provider)
    annotation_guide = ANNOTATION_GUIDE_PATH.read_text()
    system_prompt = _build_system_prompt(annotation_guide, CANONICAL_SKILLS)

    with INPUT_PATH.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open() as f:
            prior = {r["job_id"]: r for r in (json.loads(line) for line in f if line.strip())}
        rows = [prior.get(r["job_id"], r) for r in rows]

    pending = [r for r in rows if r.get("skills_verified") is None]
    logger.info(
        f"Labelling {len(pending)} pending rows ({len(rows) - len(pending)} "
        f"already done) with {args.provider}/{model}..."
    )
    labelled_rows = []
    for i, row in enumerate(rows, 1):
        if row.get("skills_verified") is not None:
            labelled_rows.append(row)
            continue
        try:
            result = _label_one_row(client, model, system_prompt, row)
            skills = result.get("skills_verified", [])
            notes = result.get("notes", "")
            skills = [s for s in skills if s in CANONICAL_SKILLS]
        except Exception as e:
            logger.warning(f"  row {i} ({row.get('job_id')}): FAILED — {e}")
            skills = None
            notes = f"LLM labelling failed: {e}"

        new_row = dict(row)
        new_row["skills_verified"] = skills
        existing_notes = row.get("notes", "")
        llm_note_prefix = f"[LLM/{args.provider}]"
        if notes:
            new_row["notes"] = (
                existing_notes + " | " if existing_notes else ""
            ) + f"{llm_note_prefix} {notes}"
        labelled_rows.append(new_row)
        logger.info(f"  row {i:3d}: {skills}")
        time.sleep(rate_sleep)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        for row in labelled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_labelled = sum(1 for r in labelled_rows if r["skills_verified"] is not None)
    logger.info(f"\nWrote {len(labelled_rows)} rows to {OUTPUT_PATH}")
    logger.info(f"Successfully labelled: {n_labelled}/{len(rows)}")

    if n_labelled < len(rows):
        logger.warning(
            f"{len(rows) - n_labelled} rows failed labelling. "
            "Re-run to retry, or hand-label the failures."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
