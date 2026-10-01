# TalentLens Agent Run Report

**Profile:** AI Engineer | Skills: Python, RAG, LLMs, FastAPI
**Status:** done
**Steps:** 5 | **Total time:** 100ms
**Results:** 5 ranked jobs

## Execution trace

| Step | Action | Tool | Output | Time |
|------|--------|------|--------|------|
| 1 | plan_sources | select_sources | selected sources: ['adzuna', 'remoteok'] | 0ms |
| 2 | fetch_adzuna | fetch_from_source | 3 postings fetched | 50ms |
| 3 | fetch_remoteok | fetch_from_source | 2 postings fetched | 50ms |
| 4 | deduplicate | deduplicate | 5 after dedup | 0ms |
| 5 | rank | rank_by_profile | top match: Senior NLP Engineer score=0.983 | 0ms |

## Agent reflections

- Attempt 1: 5 results, top score 0.98. Quality sufficient — finishing.

## Top ranked results

| Rank | Title | Company | Score | Remote |
|------|-------|---------|-------|--------|
| 1 | Senior NLP Engineer | AI Startup (Remote) | 0.983 | Yes |
| 2 | AI Engineer | Series B Fintech | 0.817 | Yes |
| 3 | Research Scientist | AI Lab | 0.500 | No |
| 4 | ML Engineer | Swiggy | 0.167 | No |
| 5 | Data Scientist | CRED | 0.167 | No |