# TalentLens Data Collection Summary

## Overview
- **Total collected:** 600
- **Valid (schema check):** 600
- **After deduplication:** 582
- **Duplicates removed:** 18 (3.0%)
- **Output file:** `data/raw/jobs_raw.demo.csv` (0.3MB)

## By source

| Source | Collected | Valid | Rate |
|--------|-----------|-------|------|
| demo | 600 | 600 | 100.0% |
| **Total** | **600** | **582** | **97.0%** |


## Field coverage

| Field | Coverage |
|-------|----------|
| `job_id` | ✅ 100% |
| `source` | ✅ 100% |
| `title` | ✅ 100% |
| `company` | ✅ 100% |
| `city` | ✅ 100% |
| `country` | ✅ 100% |
| `description` | ✅ 100% |
| `skills_raw` | ✅ 100% |
| `salary_min` | ⚠️ 81% |
| `salary_max` | ⚠️ 81% |
| `currency` | ✅ 100% |
| `is_remote` | ✅ 100% |
| `posted_date` | ✅ 100% |
| `url` | ✅ 100% |


## Next step

Run Chapter 6 to clean this dataset:
```bash
python book/ch06/ch06_data_cleaning_preprocessing.py
```

Outputs: `data/clean/jobs_clean.csv` — ready for EDA (Ch7) and ML (Ch9).
