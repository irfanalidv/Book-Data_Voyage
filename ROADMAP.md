# Roadmap

What the book deliberately leaves out, and what is planned for the next edition. If you want to work on one of these, open an issue first so we can agree on the approach.

## Chapter extensions

| Chapter | Item | Why it isn't in this edition |
|---|---|---|
| 10 | A fourth feature group (text features beyond bag-of-words) | The `FEATURE_GROUPS` registry makes it a small addition; the chapter stays focused on three groups and an honest ablation. |
| 11 | BERTopic + UMAP clustering | Better on a large corpus, but adds a sentence-transformers dependency before Chapter 13 introduces it. |
| 13 | Fine-tuned NER on the labelled eval set | Needs a GPU and a training set well beyond the 200-row eval set. |
| 13 | Hindi and Tamil extraction | The current methods are English-only. A multilingual MiniLM model would cover this at a small cost in English accuracy. |
| 13 | A larger skill vocabulary (200–500 skills) | The chapter uses 24 canonical skills. A larger list changes the regex-vs-embedding trade-off and is worth measuring. |
| 13 | LLM extraction at inference time | Highest quality and highest cost; worth a comparison run as token prices fall. |
| 14 | Skill trends on real TalentLens data | Needs six or more months of dated postings. The chapter teaches the methods on COVID and synthetic series until then. |
| 15 | Live polars, dask, and DuckDB benchmarks | The chapter measures pandas optimisations and covers the others with a decision table. |
| 18 | Multi-agent patterns, async tools, persistent memory | Each is its own topic; the chapter keeps one agent and four tools so the loop stays readable. |
| 18 | Caching, streaming, and batching for agents | Named in the chapter as production concerns; implementing them would bury the core loop. |
| 18 | A fresh recording of the agent run | The committed traces come from a Groq model that has since been retired. Replay is exact; `--live` records a new run with a current model. |
| 18 | Tool-calling comparisons across providers | The chapter uses one provider so its measurements stay comparable. |

## Code

- **Serve the role classifier over HTTP.** Chapter 19's API serves search and a salary-band heuristic. Adding the Chapter 9 classifier means adding scikit-learn to the slim image; Chapter 19's "What's next" sketches the route.
- **Label-independent salary imputation.** Chapter 6 imputes missing salaries by role median, and Chapter 10 shows that this leaks the label on real Adzuna data. The chapter teaches the fix; the pipeline default stays unchanged so the book's numbers reproduce.
- **A standalone `talentlens-core`.** The Chapter 22 package wraps chapter modules and needs a repository checkout. It has to vendor that code before a real PyPI release.
- **Stricter tests for the agent loop.** Add a mocked turn that calls two tools in sequence and assert the exact order.
- **The full ruff rule set on `talentlens/`.** CI uses a narrow rule set; `make lint-all` still reports naming findings on scikit-learn-style `X` parameters.

## Dependencies

- **transformers 5.x and torch 2.9+.** These releases fix open security advisories, but they are major upgrades that can shift the embedding and neural-network numbers quoted in Chapters 12, 13, and 16. They will land together with a re-run of those chapters.

## Book production

- **An A–Z index** for the print and EPUB editions, built once the final layout is fixed.
