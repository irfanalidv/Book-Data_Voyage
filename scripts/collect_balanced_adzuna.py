"""Collect a balanced dataset of real postings from the Adzuna API, then clean it.

Five role keywords, up to 400 rows each, deduplicated by job_id. Uses YOUR
Adzuna key (free at developer.adzuna.com), so the data is collected under
your own API agreement. Writes:

    data/raw/jobs_raw.adzuna.csv      raw postings
    data/clean/jobs_clean.large.csv   cleaned by Chapter 6's pipeline

Chapters prefer ``jobs_clean.large.csv`` automatically when it exists
(``talentlens.paths.jobs_clean_path``); delete it to return to the bundled
corpus. The bundled ``jobs_raw.csv`` / ``jobs_clean.csv`` are never touched.

Run::

    make collect-dataset
    # or: PYTHONPATH=. python scripts/collect_balanced_adzuna.py

Uses roughly 40-50 Adzuna API calls. Respects rate-limit backoff in
``BaseCollector._get``. Adzuna's terms require attributing postings with
"Jobs by Adzuna" wherever you display them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch05"))
import ch05_data_collection as ch05  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROLE_QUERIES = [
    ("AI Engineer", "AI Engineer"),
    ("ML Engineer", "machine learning engineer"),
    ("Data Scientist", "data scientist"),
    ("Data Engineer", "data engineer"),
    ("Data Analyst", "data analyst"),
]

TARGET_PER_ROLE = 400


def main() -> int:
    ch05._load_dotenv()

    cfg = ch05.Config(demo_mode=False)
    if not cfg.adzuna_app_id or not cfg.adzuna_api_key:
        logger.error("ADZUNA_APP_ID and ADZUNA_API_KEY required in .env")
        return 1

    collector = ch05.AdzunaCollector(cfg)

    all_postings: list[dict] = []
    seen_ids: set[str] = set()

    for role_label, keyword in ROLE_QUERIES:
        logger.info("=== Collecting %s (keyword=%r) ===", role_label, keyword)
        try:
            postings = collector.collect_by_keyword(keyword, target_rows=TARGET_PER_ROLE)
        except Exception as exc:
            logger.error("  failed: %s", exc)
            continue

        new_count = 0
        for posting in postings:
            job_id = posting.get("job_id")
            if job_id and job_id not in seen_ids:
                seen_ids.add(job_id)
                all_postings.append(posting)
                new_count += 1

        logger.info(
            "  %s: collected %s, new %s, total %s",
            role_label,
            len(postings),
            new_count,
            len(all_postings),
        )

    if not all_postings:
        logger.error("No postings collected. Aborting.")
        return 1

    df = pd.DataFrame(all_postings)
    out_path = _REPO_ROOT / "data" / "raw" / "jobs_raw.adzuna.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info("")
    logger.info("Wrote %s unique rows to %s", len(df), out_path)

    # Clean with Chapter 6's pipeline into the path chapters prefer.
    sys.path.insert(0, str(_REPO_ROOT / "book" / "ch06"))
    import ch06_data_cleaning_preprocessing as ch06  # noqa: E402

    clean_cfg = ch06.Config()
    clean_cfg.raw_path = out_path
    clean_cfg.clean_path = _REPO_ROOT / "data" / "clean" / "jobs_clean.large.csv"
    _, clean_df, clean_path = ch06.run_cleaning_pipeline(clean_cfg, overwrite=True)
    logger.info("Cleaned %s rows -> %s", len(clean_df), clean_path)
    logger.info("Chapters now read this file. Delete it to return to the bundled corpus.")
    logger.info("")
    logger.info("Per-source breakdown:")
    if "source" in df.columns:
        print(df["source"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
