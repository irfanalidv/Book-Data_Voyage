# Data Voyage
### Building Real AI Systems from Data to Deployment

---

This is the chapter-by-chapter index of the book: 24 chapters in 7 parts, all anchored on a single running project, **TalentLens**, a job market intelligence platform we build from raw data to a deployed API.

For how the project threads through the chapters, see [`RUNNING_PROJECT_TALENTLENS.md`](RUNNING_PROJECT_TALENTLENS.md). For installation and the dataset, see the root [`README.md`](../README.md).

Every chapter folder holds a `README.md` (the chapter text), a runnable script, and generated `reports/`. Every number quoted in a chapter reproduces from a fresh clone on the bundled dataset.

---

## Front matter

- [Copyright](COPYRIGHT.md)
- [Dedication](DEDICATION.md)
- [A Note Before You Begin](FOREWORD.md)
- [Preface](PREFACE.md): why this book exists and how to read it
- [Acknowledgements](ACKNOWLEDGEMENTS.md)
- [Prologue](PROLOGUE.md): what we're building

---

## Part I — Foundations

| # | Chapter | What it covers |
|---|---------|----------------|
| 1 | [The Data Science Landscape](ch01/README.md) | The five roles in data and AI, what each does and pays; environment check |
| 2 | [The Working Engineer's Python](ch02/README.md) | `pyproject.toml`, editable installs, pydantic settings, logging, Makefile |
| 3 | [The Statistics You Actually Need](ch03/README.md) | Mean vs median on skewed pay, IQR, skew, `log1p`, confidence intervals |
| 4 | [Data: Types, Sources, Ethics](ch04/README.md) | Which sources TalentLens may use, the canonical schema, scraping law and ethics |

## Part II — Data to Insight

| # | Chapter | What it covers |
|---|---------|----------------|
| 5 | [Data Collection](ch05/README.md) | Adzuna and RemoteOK collectors, rate limits, validation, the seeded demo corpus |
| 6 | [Data Cleaning](ch06/README.md) | Missing salaries (MCAR/MAR/MNAR), imputation flags, dedup, leakage at cleaning time |
| 7 | [Exploratory Data Analysis](ch07/README.md) | Salary shape, skill frequency, role and remote comparisons on disclosed pay |
| 8 | [Statistical Inference](ch08/README.md) | Mann-Whitney, stratification, chi-squared; a remote premium that is really seniority |

## Part III — Classical ML

| # | Chapter | What it covers |
|---|---------|----------------|
| 9 | [Supervised Learning](ch09/README.md) | TF-IDF role classifier trained without the title; one-standard-error model selection |
| 10 | [Feature Engineering and Selection](ch10/README.md) | Ablations, three selection methods, a negative result, and salary-imputation leakage |
| 11 | [Unsupervised Learning](ch11/README.md) | TF-IDF + KMeans, elbow and silhouette, auditing labels with clusters |
| 12 | [Neural Networks](ch12/README.md) | PyTorch MLPs on four datasets: scaling, overfitting, early stopping, baselines |

## Part IV — Specialist AI

| # | Chapter | What it covers |
|---|---------|----------------|
| 13 | [NLP for Skill Extraction](ch13/README.md) | Regex vs spaCy vs sentence-transformers on a 200-posting labelled eval set |
| 14 | [Time Series](ch14/README.md) | Decomposition, ADF/KPSS stationarity, ARIMA, held-out forecast error |
| 15 | [Scaling Python](ch15/README.md) | Measured pandas optimisations; when polars, dask, or Spark are worth it |

## Part V — The GenAI Stack

| # | Chapter | What it covers |
|---|---------|----------------|
| 16 | [RAG and Vector Search](ch16/README.md) | Sentence embeddings, cosine search, chunking, hybrid ranking, pgvector patterns |
| 17 | [LLM Generation](ch17/README.md) | CV parsing and match explanations; JSON mode, validation, graceful fallback |
| 18 | [Agentic AI](ch18/README.md) | A 120-line tool-calling agent and five measured failure modes |

## Part VI — Shipping

| # | Chapter | What it covers |
|---|---------|----------------|
| 19 | [FastAPI](ch19/README.md) | `/health`, `/api/v1/search`, `/api/v1/classify`, Pydantic contracts, rate limiting |
| 20 | [Docker + Render](ch20/README.md) | A 462 MB slim serving image, layer caching, health checks, `render.yaml` |
| 21 | [CI/CD](ch21/README.md) | GitHub Actions test → lint → Docker → deploy, and Make as the local mirror |
| 22 | [Packaging as a PyPI Library](ch22/README.md) | `talentlens-core`: ranges not pins, semver, trusted publishing |

## Part VII — Career

| # | Chapter | What it covers |
|---|---------|----------------|
| 23 | [Real-World Case Studies](ch23/README.md) | Reflecta, Godam, RAGNav and ragfallback, StackSift: architecture decisions and what broke |
| 24 | [The India Playbook](ch24/README.md) | Five markets, negotiation, remote contracts, and which claims you can measure |

---

## Back matter

- [Glossary](GLOSSARY.md): TalentLens vocabulary and technical terms, each with a chapter pointer
- [References and further reading](REFERENCES.md): the libraries, data sources, and standards the book uses
- [About the Author](ABOUT_THE_AUTHOR.md)

---

## How to read

- **Brand new?** Start with the Preface, Prologue, and Chapter 1, then read in order.
- **Comfortable with Python and basic ML?** Skim Parts I–II and start writing real code at Chapter 5.
- **Already shipping ML and want the GenAI and production stack?** Jump to Part V.
- **Want the career advice first?** Chapter 24 stands alone, though it lands harder after you've built TalentLens.
