# Scope

What this book deliberately leaves out, and why. Each item is a good next project if you want to take TalentLens further on your own.

## Extensions to try

| Chapter | Extension | Why the book leaves it out |
|---|---|---|
| 10 | A fourth feature group (text features beyond bag-of-words) | The `FEATURE_GROUPS` registry makes it a small addition; the chapter stays focused on three groups and an honest ablation. |
| 11 | BERTopic + UMAP clustering | Better on a large corpus, but adds a sentence-transformers dependency before Chapter 13 introduces it. |
| 13 | Fine-tuned NER on the labelled eval set | Needs a GPU and a training set well beyond the 200-row eval set. |
| 13 | Hindi and Tamil extraction | The methods are English-only. A multilingual MiniLM model would cover this at a small cost in English accuracy. |
| 13 | A larger skill vocabulary (200–500 skills) | The chapter uses 24 canonical skills. A larger list changes the regex-vs-embedding trade-off and is worth measuring. |
| 13 | LLM extraction at inference time | Highest quality and highest cost; worth a comparison run as token prices fall. |
| 14 | Skill trends on real TalentLens data | Needs six or more months of dated postings. The chapter teaches the methods on COVID and synthetic series. |
| 15 | Live polars, dask, and DuckDB benchmarks | The chapter measures pandas optimisations and covers the others with a decision table. |
| 18 | Multi-agent patterns, async tools, persistent memory | Each is its own topic; the chapter keeps one agent and four tools so the loop stays readable. |
| 18 | Caching, streaming, and batching for agents | Named in the chapter as production concerns; implementing them would bury the core loop. |
| 18 | Prompt and provider mitigations from the failure analysis | `book/ch18/reports/agent_report.md` names them; the chapter measures the failures rather than tuning them away. |
| 18 | Tool-calling comparisons across providers | The chapter uses one provider so its measurements stay comparable. |
| 19 | Serve the Chapter 9 role classifier over HTTP | It means adding scikit-learn to the slim image; Chapter 19's "What's next" sketches the route. |
| 22 | A standalone `talentlens-core` on PyPI | The package wraps chapter modules and needs a repository checkout. It would have to vendor that code first. |

## Deliberate choices

- **Salary imputation by role median.** Chapter 6 imputes missing salaries this way, and Chapter 10 shows that it leaks the label on real Adzuna data. The chapter teaches the fix; the pipeline default stays unchanged so the book's numbers reproduce.
- **Recorded agent run.** The Chapter 18 traces come from a Groq model that has since been retired. Replay is exact; `--live` records a new run with a current model.
- **Narrow lint rules.** CI uses ruff rules E, F, W and I. `make lint-all` also reports naming findings on scikit-learn-style `X` parameters, which the book keeps on purpose because they match the library's conventions.

## Pinned dependencies

Every version is pinned in `requirements-lock.txt` so that each number in the book reproduces exactly. The pinned PyTorch 2.8 and Transformers 4.57 have published security advisories:

- **Transformers:** code execution or file writes when loading, initialising, or saving untrusted models, chat templates, or remote code. Fixes arrive in 5.x, and one advisory has no fixed release yet.
- **PyTorch:** memory corruption in `torch.jit.script`, `torch.lstm_cell`, and `unpack_sequence` when given crafted input. Fixed between 2.9.1 and 2.13.

The book's code loads only well-known public models on your own machine and never passes untrusted input to those functions. Upgrading would shift the embedding and neural-network numbers quoted in Chapters 12, 13, and 16, so the book stays on the versions it was tested with. If you build on this code for production, upgrade both packages, re-run Chapters 12, 13, and 16, and expect small differences in those numbers.
