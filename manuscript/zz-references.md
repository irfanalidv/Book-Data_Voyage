# References and further reading

Canonical documentation and data sources for the tools, standards, and influences *Data Voyage* uses. Deep versioned doc URLs rot quickly; we link to stable homepages and doc roots. If a link breaks, open an issue on the companion repository.

---

## Core libraries

| Library | Documentation | Role in this book |
|---------|---------------|-------------------|
| **pandas** | https://pandas.pydata.org | Tabular data throughout ingestion, cleaning, EDA, and feature work |
| **NumPy** | https://numpy.org | Arrays, statistics, and numerical primitives under pandas and sklearn |
| **SciPy** | https://scipy.org | Scientific routines supporting stats and ML stacks |
| **statsmodels** | https://www.statsmodels.org | Time-series and statistical tests (Chapter 14; core content when Prophet is optional) |
| **matplotlib** | https://matplotlib.org | Chapter figures and reproducible plots |
| **seaborn** | https://seaborn.pydata.org | Statistical visualisation layered on matplotlib |
| **scikit-learn** | https://scikit-learn.org | Classification, clustering, pipelines, cross-validation, and bundled teaching datasets (Chapters 9–12) |
| **joblib** | https://joblib.readthedocs.io | Serialising trained sklearn pipelines to `.joblib` artefacts |
| **imbalanced-learn** | https://imbalanced-learn.org | Class-imbalance techniques referenced in Chapter 9 (e.g. SMOTE) |
| **PyTorch** | https://pytorch.org | Chapter 12's neural networks; backend for sentence-transformers |
| **transformers** | https://huggingface.co/docs/transformers | Hugging Face model hub access for pretrained encoders |
| **sentence-transformers** | https://www.sbert.net | Dense embeddings for semantic search and skill matching (Chapters 13, 16) |
| **spaCy** | https://spacy.io | Tokenisation, EntityRuler skill extraction (Chapter 13) |
| **Prophet** | https://facebook.github.io/prophet | Optional forecaster named in Chapter 14; statsmodels carries core teaching |
| **polars** | https://pola.rs | Columnar scaling path (Chapter 15) |
| **dask** | https://www.dask.org | Parallel pandas-style execution when data exceeds RAM (Chapter 15) |
| **pyarrow** | https://arrow.apache.org/docs/python | Columnar interchange for polars/dask and future Parquet paths |
| **faiss-cpu** | https://github.com/facebookresearch/faiss | Approximate nearest-neighbour search at scale (optional `vectors` extra) |
| **FastAPI** | https://fastapi.tiangolo.com | TalentLens HTTP API (Chapter 19) |
| **Pydantic** | https://docs.pydantic.dev | Settings and request/response validation (Chapters 2, 19) |
| **uvicorn** | https://uvicorn.dev | ASGI server running the FastAPI app |
| **httpx** | https://www.python-httpx.org | HTTP client for API tests and integrations |
| **OpenAI Python SDK** | https://platform.openai.com/docs | LLM provider option in Chapter 17 |
| **Groq** | https://console.groq.com/docs | Default fast LLM inference for generation and agents (Chapters 17–18) |
| **Anthropic SDK** | https://docs.anthropic.com | Installed with the book's stack; a third provider you can add to Chapter 17's client wrapper |
| **requests** | https://requests.readthedocs.io | HTTP collectors for Adzuna and RemoteOK (Chapter 5) |
| **tenacity** | https://tenacity.readthedocs.io | Retry decorators, shown in Chapter 17 |
| **python-dotenv** | https://github.com/theskumar/python-dotenv | Loading `.env` for local API keys |
| **pytest** | https://docs.pytest.org | Chapter and integration tests (`tests/`, `make test`) |
| **ruff** | https://docs.astral.sh/ruff | Linting and formatting (Chapter 2, CI) |
| **pre-commit** | https://pre-commit.com | Git hooks for style and hygiene (Chapter 2) |
| **Docker** | https://docs.docker.com | Container images for deployment (Chapter 20) |
| **GitHub Actions** | https://docs.github.com/actions | CI/CD pipeline (Chapter 21) |
| **Render** | https://render.com/docs | Managed deployment target for the TalentLens API (Chapter 20) |
| **pgvector** | https://github.com/pgvector/pgvector | PostgreSQL vector extension for production RAG (Chapter 16) |

**Lockfiles:** [`requirements.txt`](https://github.com/irfanalidv/Book-Data_Voyage/blob/main/requirements.txt) declares compatible-release pins; [`requirements-lock.txt`](https://github.com/irfanalidv/Book-Data_Voyage/blob/main/requirements-lock.txt) holds exact versions for CI and readers. Regenerate with `uv` via `make lock`.

**Maintainers we stand on:** Wes McKinney (pandas), the scikit-learn community, Sebastián Ramírez (FastAPI), and Nils Reimers (sentence-transformers).

---

## Data sources

| Source | Entry point | Role in TalentLens |
|--------|-------------|-------------------|
| **Adzuna API** | https://developer.adzuna.com | Live India and global job postings (Chapters 4–5) |
| **RemoteOK API** | https://remoteok.com/api | Remote tech roles JSON feed, no auth (Chapters 4–5) |
| **GitHub Jobs archive (Kaggle)** | https://www.kaggle.com (search "GitHub Jobs") | Historical postings allowed by Chapter 4's policy; collector left as an exercise (Chapter 5) |
| **disease.sh** | https://disease.sh | Free COVID-19 API wrapping the Johns Hopkins series (Chapter 14) |
| **scikit-learn datasets** | https://scikit-learn.org/stable/datasets.html | Digits, Diabetes, California Housing (Chapter 12) |
| **Hugging Face Hub** | https://huggingface.co | Pretrained sentence-transformer and transformer weights |

---

## Standards and practices

| Standard | Reference | Where the book uses it |
|----------|-----------|------------------------|
| **Keep a Changelog** | https://keepachangelog.com | `CHANGELOG.md` format |
| **Semantic Versioning** | https://semver.org | Package and release versioning (Chapter 22) |
| **PyPI Trusted Publishing** | https://docs.pypi.org/trusted-publishers/ | CI releases without long-lived API tokens (Chapter 22) |
| **SPDX license identifiers** | https://spdx.org/licenses | `license` field in [`pyproject.toml`](https://github.com/irfanalidv/Book-Data_Voyage/blob/main/pyproject.toml) (Chapter 22) |
| **Diátaxis** | https://diataxis.fr | Documentation framing behind the book's split between explanation, reference boxes, and how-to steps |

---

## Chart and narrative craft

| Influence | Reference | Where named |
|-----------|-----------|-------------|
| **Cole Nussbaumer Knaflic**, *Storytelling with Data* (Wiley, 2015) | Annotate takeaways on charts; remove clutter; show negative examples | The book's figure conventions (Chapters 3, 7) |

---

## From the author

Libraries and systems built or operated in production and discussed as case studies (see the Acknowledgements):

- **RAGNav**: open-source hybrid retrieval, https://github.com/irfanalidv/RAGNav (Chapter 23)
- **ragfallback**: reliability layer and CI gate for RAG pipelines, https://github.com/irfanalidv/ragfallback (Chapter 23)
- **StackSift**: B2B product intelligence API, https://stacksift.in/ (Chapter 23)
- **Reflecta**: https://www.getreflecta.com/ and **Godam**: https://www.getgodam.com/ (Chapter 23)
- **AgentEnsemble**, **AgentCare**, **scrapeflow-py**: PyPI libraries under the author's GitHub account
- **Reflecta**, **Godam**: production case studies (Chapter 23)
- **talentlens-core**: the library packaged in Chapter 22

**Research:** Two peer-reviewed papers in IJAINN (2025); see the author's GitHub profile for titles.

---

If a documentation link above fails, open an issue on [github.com/irfanalidv/Book-Data_Voyage](https://github.com/irfanalidv/Book-Data_Voyage) with the broken URL and where you found it in the book.
