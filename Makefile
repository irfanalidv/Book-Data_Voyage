# TalentLens — Project Makefile
#
# Every task in this project runs through make.
# One command interface, consistent between local dev and CI.
#
# Usage:
#   make help          — list all commands
#   make install       — set up the project
#   make test          — run all tests
#   make lint          — check code quality
#   make run           — start the API server
#   make docker-build  — build the Docker image
#   make deploy-check  — validate everything before deploying
#
# Requirements: Python 3.11+, pip, (optionally) Docker

.PHONY: help install install-dev spacy-model verify-api-deps lock test test-fast test-ch01 test-ch02 test-ch03 test-ch04 test-ch05 test-ch06 test-ch07 test-ch08 test-ch09 test-ch16 test-ch17 test-ch19 \
        test-ch20 test-ch21 test-ch22 test-ch24 lint lint-all format type-check \
        run run-docker docker docker-build docker-run docker-stop \
        collect clean-data train-role eda rag llm api deploy-check clean clean-figures clean-cache \
        chapter-05 chapter-06 chapter-07 chapter-09 chapter-16 chapter-17 chapter-19 chapter-20 chapter-21 chapter-22 chapter-24 \
        all-chapters publish-check collect-dataset fetch-dataset fetch-dataset-force publish-dataset manuscript epub pdf

# ── Colours for terminal output ──────────────────────────────────────────────
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
RESET  := \033[0m
BOLD   := \033[1m

# ── Project config ────────────────────────────────────────────────────────────
PYTHON     := python3
PIP        := pip3
PYTEST     := python3 -m pytest
UVICORN    := uvicorn
PORT       := 8000
IMAGE_NAME := talentlens
IMAGE_TAG  := latest
BOOK_DIR   := book
TEST_DIR   := tests
# Required for `book.ch19...` imports (Dockerfile, uvicorn, CI)
export PYTHONPATH := $(abspath .)

# ── Default target ────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo ""
	@echo "$(BOLD)TalentLens — Data Voyage Book Project$(RESET)"
	@echo "$(BOLD)=====================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)Setup:$(RESET)"
	@grep -E '^(install|install-dev)[^:]*:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Testing:$(RESET)"
	@grep -E '^test[^:]*:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Code quality:$(RESET)"
	@grep -E '^(lint|format|type-check)[^:]*:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Running the API:$(RESET)"
	@grep -E '^(run|docker)[^:]*:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Book chapters:$(RESET)"
	@grep -E '^(eda|rag|api|chapter|all-chapters)[^:]*:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Deployment:$(RESET)"
	@grep -E '^(deploy|publish)[^:]*:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Maintenance:$(RESET)"
	@grep -E '^clean[^:]*:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
lock: ## Regenerate requirements-lock.txt and requirements-dev-lock.txt from abstract pins
	@echo "$(GREEN)Compiling lockfiles with uv...$(RESET)"
	uv pip compile requirements.txt --output-file requirements-lock.txt
	uv pip compile requirements-dev.txt --output-file requirements-dev-lock.txt
	@echo "$(GREEN)Lockfiles updated. Commit requirements-lock.txt and requirements-dev-lock.txt.$(RESET)"

install: ## Install runtime dependencies (exact versions from lockfile)
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements-lock.txt
	$(PIP) install -e .
	@$(MAKE) spacy-model
	@echo "$(GREEN)Done. Editable install active — 'import talentlens' works. Run 'make run' for the API.$(RESET)"

spacy-model: ## Download spaCy English model for Chapter 13 (en_core_web_sm)
	@echo "$(GREEN)Downloading spaCy model en_core_web_sm...$(RESET)"
	$(PYTHON) -m spacy download en_core_web_sm

install-dev: install ## Install runtime + development dependencies
	@echo "$(GREEN)Installing dev dependencies...$(RESET)"
	$(PIP) install -r requirements-dev-lock.txt
	@echo "$(GREEN)Dev environment ready. Run 'make test' to verify.$(RESET)"

verify-api-deps: ## Verify requirements-api.txt is sufficient for the ch19 serving image
	@echo "$(GREEN)Verifying slim runtime deps in isolated venv...$(RESET)"
	rm -rf /tmp/talentlens-slim-test
	$(PYTHON) -m venv /tmp/talentlens-slim-test
	/tmp/talentlens-slim-test/bin/pip install --quiet -r requirements-api.txt
	/tmp/talentlens-slim-test/bin/pip install --quiet --no-deps -e .  # same as the Dockerfile
	/tmp/talentlens-slim-test/bin/python -c "import book.ch19.ch19_fastapi_deployment; print('OK: ch19 imports cleanly with slim deps')"
	rm -rf /tmp/talentlens-slim-test
	@echo "$(GREEN)Verified.$(RESET)"

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run all tests with coverage report
	@echo "$(GREEN)Running all tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ \
		-v \
		--tb=short \
		--cov=$(BOOK_DIR) \
		--cov-report=term-missing \
		--cov-report=html:reports/coverage \
		-q
	@echo "$(GREEN)Coverage report: reports/coverage/index.html$(RESET)"

test-fast: ## Run tests without coverage (faster)
	@echo "$(GREEN)Running tests (no coverage)...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=short -q

test-ch01: ## Run Chapter 1 tests (environment check + role distribution chart)
	$(PYTEST) book/ch01/tests/ -v --tb=short

test-ch02: ## Run Chapter 2 tests (config, paths, package import)
	$(PYTEST) book/ch02/tests/ -v --tb=short

test-ch03: ## Run Chapter 3 tests (synthetic salary statistics + figures)
	$(PYTEST) book/ch03/tests/ -v --tb=short

test-ch04: ## Run Chapter 4 tests (data sources + schema normalisation)
	$(PYTEST) book/ch04/tests/ -v --tb=short

test-ch05: ## Run Chapter 5 tests (data collection pipeline)
	$(PYTEST) $(TEST_DIR)/test_ch05.py -v --tb=short

test-ch06: ## Run Chapter 6 tests (data cleaning pipeline)
	$(PYTEST) book/ch06/tests/ -v --tb=short

test-ch07: ## Run Chapter 7 tests (EDA)
	$(PYTEST) $(TEST_DIR)/test_ch07.py -v --tb=short

test-ch08: ## Run Chapter 8 tests (statistical inference)
	$(PYTEST) book/ch08/tests/ -v --tb=short

test-ch09: ## Run Chapter 9 tests (role classifier)
	$(PYTEST) $(TEST_DIR)/test_ch09.py -v --tb=short
	$(PYTEST) $(TEST_DIR)/test_ch09.py -v --tb=short

test-ch16: ## Run Chapter 16 tests (RAG + vector search)
	$(PYTEST) $(TEST_DIR)/test_ch16.py -v --tb=short

test-ch17: ## Run Chapter 17 tests (LLM generation layer)
	$(PYTEST) $(TEST_DIR)/test_ch17.py -v --tb=short

test-ch19: ## Run Chapter 19 tests (FastAPI)
	$(PYTEST) $(TEST_DIR)/test_ch19.py -v --tb=short

test-ch20: ## Run Chapter 20 tests (Docker + Render)
	$(PYTEST) $(TEST_DIR)/test_ch20.py -v --tb=short

test-ch21: ## Run Chapter 21 tests (CI/CD pipeline validator)
	$(PYTEST) $(TEST_DIR)/test_ch21.py -v --tb=short

test-ch22: ## Run Chapter 22 tests (talentlens-core package)
	$(PYTEST) $(TEST_DIR)/test_ch22.py -v --tb=short

test-ch24: ## Run Chapter 24 tests (Career analysis)
	$(PYTEST) $(TEST_DIR)/test_ch24.py -v --tb=short

# ── Code quality ──────────────────────────────────────────────────────────────
lint: ## Check code style (same scope and rules as CI lint job)
	@echo "$(GREEN)Linting (CI scope)...$(RESET)"
	$(PYTHON) -m ruff check $(BOOK_DIR)/ $(TEST_DIR)/ talentlens/ \
		--select=E,F,W,I --ignore=E501
	@echo "$(YELLOW)Black check (CI warns only — does not fail the job)...$(RESET)"
	-$(PYTHON) -m black $(BOOK_DIR)/ $(TEST_DIR)/ --line-length 100 --check
	@echo "$(GREEN)Lint passed.$(RESET)"

# Full pyproject [tool.ruff.lint] rules — NOT a must-pass gate.
# Currently fails on chapter scripts (N806: ML X/y/ax, matplotlib ax; B905 zip strict; etc.).
# We deliberately do not enforce this in CI: naming rules like N806 fight idiomatic ML code.
# Use only as an exploratory strict check before tightening config — not `make lint`.
lint-all: ## Full ruff rule set (exploratory; not CI; often fails — see comment above)
	@echo "$(GREEN)Linting (full pyproject rules)...$(RESET)"
	$(PYTHON) -m ruff check $(BOOK_DIR)/ $(TEST_DIR)/ talentlens/
	@echo "$(GREEN)Lint-all passed.$(RESET)"

format: ## Format code with black
	@echo "$(GREEN)Formatting...$(RESET)"
	$(PYTHON) -m black $(BOOK_DIR)/ $(TEST_DIR)/ --line-length 100
	@echo "$(GREEN)Formatting done.$(RESET)"

format-check: ## Check formatting without making changes (used in CI)
	$(PYTHON) -m black $(BOOK_DIR)/ $(TEST_DIR)/ --line-length 100 --check

type-check: ## Run mypy on talentlens/ (strict; chapter scripts are excluded in pyproject.toml)
	@echo "$(GREEN)Type checking...$(RESET)"
	$(PYTHON) -m mypy talentlens/ --ignore-missing-imports
	@echo "$(GREEN)Type check passed.$(RESET)"

# ── Running the API ───────────────────────────────────────────────────────────
run: ## Start the TalentLens API server (localhost:8000)
	@echo "$(GREEN)Starting TalentLens API...$(RESET)"
	@echo "$(YELLOW)Docs: http://localhost:$(PORT)/docs$(RESET)"
	@echo "$(YELLOW)Health: http://localhost:$(PORT)/health$(RESET)"
	$(UVICORN) book.ch19.ch19_fastapi_deployment:app --host 127.0.0.1 --port $(PORT)

run-reload: ## Start API with auto-reload (development mode)
	@echo "$(GREEN)Starting TalentLens API (dev mode)...$(RESET)"
	$(UVICORN) book.ch19.ch19_fastapi_deployment:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload \
		--log-level debug

# ── Docker ────────────────────────────────────────────────────────────────────
docker: docker-build ## Alias for docker-build (root README)

docker-build: ## Build the Docker image
	@echo "$(GREEN)Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)...$(RESET)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "$(GREEN)Image built:$(RESET)"
	docker images $(IMAGE_NAME)

docker-run: ## Run the Docker container locally
	@echo "$(GREEN)Starting container on port $(PORT)...$(RESET)"
	docker run -d \
		--name talentlens-local \
		-p $(PORT):8000 \
		-e PORT=8000 \
		$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "$(GREEN)Container started. Waiting for health check...$(RESET)"
	@sleep 3
	curl -s http://localhost:$(PORT)/health | $(PYTHON) -m json.tool

docker-stop: ## Stop and remove the local container
	docker stop talentlens-local 2>/dev/null || true
	docker rm talentlens-local 2>/dev/null || true
	@echo "$(GREEN)Container stopped.$(RESET)"

docker-logs: ## Show logs from the running container
	docker logs talentlens-local -f

docker-shell: ## Open a shell inside the running container
	docker exec -it talentlens-local /bin/bash

docker-size: ## Show the image size breakdown by layer
	docker history $(IMAGE_NAME):$(IMAGE_TAG) --human --format "{{.Size}}\t{{.CreatedBy}}"

# ── Book chapters ─────────────────────────────────────────────────────────────
collect: ## Run Chapter 5 data collection → data/raw/jobs_raw.csv
	@echo "$(GREEN)Running Chapter 5: data collection...$(RESET)"
	$(PYTHON) $(BOOK_DIR)/ch05/ch05_data_collection.py
	@echo "$(GREEN)Raw jobs: data/raw/jobs_raw.csv$(RESET)"

clean-data: ## Run Chapter 6 cleaning → data/clean/jobs_clean.csv
	@echo "$(GREEN)Running Chapter 6: data cleaning...$(RESET)"
	$(PYTHON) $(BOOK_DIR)/ch06/ch06_data_cleaning_preprocessing.py
	@echo "$(GREEN)Clean jobs: data/clean/jobs_clean.csv$(RESET)"

# ── Dataset versioning (Chapter 5/6 outputs) ────────────────────────────────
collect-dataset: ## Collect up to 400 real postings per role with YOUR Adzuna key -> data/clean/jobs_clean.large.csv
	@echo "$(GREEN)Collecting from Adzuna (needs ADZUNA_APP_ID / ADZUNA_API_KEY in .env)...$(RESET)"
	PYTHONPATH=. $(PYTHON) scripts/collect_balanced_adzuna.py

fetch-dataset: ## (author only) Download the author's dataset release — needs repo access
	@echo "$(GREEN)Checking dataset manifest (talentlens/data_versions.json)...$(RESET)"
	@$(PYTHON) scripts/fetch_dataset.py
	@if [ -f data/clean/jobs_clean.large.csv ]; then \
		echo "$(GREEN)Large dataset ready at data/clean/jobs_clean.large.csv (bundled file untouched).$(RESET)"; \
	fi

fetch-dataset-force: ## Re-download dataset even if cached file matches the manifest
	$(PYTHON) scripts/fetch_dataset.py --force

# Author-only: requires live ADZUNA_* keys and a first-time live ch05 run.
# Most readers and contributors never need this target. See scripts/publish_dataset.py
# for what it does and why it's gated behind real API keys.
publish-dataset: ## (author only — requires live Adzuna keys) Publish a new dataset version
	@echo "$(YELLOW)Publishing requires live API keys and a fresh version tag.$(RESET)"
	@echo "$(YELLOW)Pass: make publish-dataset VERSION=v1.0.0-dataset-2 DESCRIPTION=\"...\"$(RESET)"
	$(PYTHON) scripts/publish_dataset.py --version "$(VERSION)" --description "$(DESCRIPTION)"

train-role: ## Run Chapter 9 role classifier → models/role_classifier.joblib
	@echo "$(GREEN)Running Chapter 9: role classifier training...$(RESET)"
	$(PYTHON) $(BOOK_DIR)/ch09/ch09_supervised_learning.py
	@echo "$(GREEN)Model: models/role_classifier.joblib$(RESET)"

eda: ## Run Chapter 7 EDA pipeline (salary + skill analysis)
	@echo "$(GREEN)Running Chapter 7: EDA...$(RESET)"
	$(PYTHON) $(BOOK_DIR)/ch07/ch07_exploratory_data_analysis.py
	@echo "$(GREEN)Figures: $(BOOK_DIR)/ch07/reports/figures/$(RESET)"

rag: ## Run Chapter 16 RAG pipeline (semantic search)
	@echo "$(GREEN)Running Chapter 16: RAG + Vector Search...$(RESET)"
	$(PYTHON) $(BOOK_DIR)/ch16/ch16_rag_vector_search.py
	@echo "$(GREEN)Search results: $(BOOK_DIR)/ch16/reports/search_demo_results.md$(RESET)"

llm: ## Run Chapter 17 LLM generation demo (stub unless GROQ_API_KEY / OPENAI_API_KEY set)
	@echo "$(GREEN)Running Chapter 17: LLM generation (demo / stub)...$(RESET)"
	$(PYTHON) $(BOOK_DIR)/ch17/ch17_llm_generation.py
	@echo "$(GREEN)Reports: $(BOOK_DIR)/ch17/reports/$(RESET)"

api: ## Run Chapter 19 inline checks + architecture figure (no long-running server)
	@echo "$(GREEN)Running Chapter 19: API checks + diagram...$(RESET)"
	$(PYTHON) -c "from book.ch19.ch19_fastapi_deployment import _run_inline_tests, plot_api_architecture; _run_inline_tests(); plot_api_architecture()"

chapter-05: test-ch05 collect ## Run Chapter 5 tests + data collection
chapter-06: clean-data ## Run Chapter 6 cleaning pipeline
chapter-07: test-ch07 eda ## Run Chapter 7 tests + script
chapter-09: test-ch09 train-role ## Run Chapter 9 tests + train classifier
chapter-16: test-ch16 rag ## Run Chapter 16 tests + script
chapter-17: test-ch17 llm ## Run Chapter 17 tests + LLM generation script (stub by default)
chapter-19: test-ch19 api ## Run Chapter 19 tests + script
chapter-20: test-ch20 ## Run Chapter 20 tests + deployment validation
	$(PYTHON) $(BOOK_DIR)/ch20/ch20_docker_deployment.py
chapter-21: test-ch21 ## Run Chapter 21 tests + CI/CD diagrams + setup guide
	$(PYTHON) $(BOOK_DIR)/ch21/ch21_cicd_pipeline.py
chapter-22: test-ch22 publish-check ## Run Chapter 22 tests + package build smoke
chapter-24: test-ch24 ## Run Chapter 24 tests + career analysis
	$(PYTHON) $(BOOK_DIR)/ch24/ch24_career_market_analysis.py

all-chapters: ## Run all chapter scripts in sequence
	@echo "$(BOLD)Running all TalentLens chapters...$(RESET)"
	@$(MAKE) chapter-05 || echo "$(RED)Chapter 5 failed$(RESET)"
	@$(MAKE) chapter-06 || echo "$(RED)Chapter 6 failed$(RESET)"
	@$(MAKE) chapter-07 || echo "$(RED)Chapter 7 failed$(RESET)"
	@$(MAKE) chapter-09 || echo "$(RED)Chapter 9 failed$(RESET)"
	@$(MAKE) chapter-16 || echo "$(RED)Chapter 16 failed$(RESET)"
	@$(MAKE) chapter-17 || echo "$(RED)Chapter 17 failed$(RESET)"
	@$(MAKE) chapter-19 || echo "$(RED)Chapter 19 failed$(RESET)"
	@$(MAKE) chapter-20 || echo "$(RED)Chapter 20 failed$(RESET)"
	@$(MAKE) chapter-21 || echo "$(RED)Chapter 21 failed$(RESET)"
	@$(MAKE) chapter-22 || echo "$(RED)Chapter 22 failed$(RESET)"
	@$(MAKE) chapter-24 || echo "$(RED)Chapter 24 failed$(RESET)"
	@echo "$(GREEN)All chapters complete. Check reports/ for output.$(RESET)"

# ── Deployment ────────────────────────────────────────────────────────────────
deploy-check: lint test-fast docker-build ## Full pre-deploy validation (lint + test + docker build)
	@echo ""
	@echo "$(GREEN)$(BOLD)Deploy check passed.$(RESET)"
	@echo "$(GREEN)Safe to push — Render will auto-deploy from main.$(RESET)"

publish-check: ## Verify the PyPI package builds cleanly (Chapter 22)
	@echo "$(GREEN)Checking talentlens-core package build...$(RESET)"
	$(PIP) install -q build
	cd $(BOOK_DIR)/ch22 && $(PYTHON) -m build
	@echo "$(GREEN)Wheel + sdist in book/ch22/dist/$(RESET)"

epub: manuscript ## Build dist/data-voyage.epub for Amazon KDP, Google Play, Apple Books (needs pandoc)
	$(PYTHON) scripts/build_epub.py

pdf: manuscript ## Build dist/data-voyage.pdf, the typeset ebook PDF (needs pandoc + Google Chrome)
	$(PYTHON) scripts/build_pdf.py

manuscript: ## Build Leanpub manuscript/ from book/ (Book.txt + rewritten image paths)
	@echo "$(GREEN)Building Leanpub manuscript/…$(RESET)"
	$(PYTHON) scripts/build_manuscript.py
	@echo "$(GREEN)Done. Connect Leanpub GitHub mode to manuscript/Book.txt.$(RESET)"

# ── Maintenance ───────────────────────────────────────────────────────────────
clean: clean-cache clean-figures ## Remove all generated files
	@echo "$(GREEN)Project cleaned.$(RESET)"

clean-figures: ## Remove generated figures (they're regenerated by chapter scripts)
	@echo "$(YELLOW)Removing generated figures...$(RESET)"
	find $(BOOK_DIR) -path "*/reports/figures/*.png" -delete
	@echo "$(GREEN)Figures removed.$(RESET)"

clean-cache: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Cache cleared.$(RESET)"

clean-embeddings: ## Remove cached embedding files (forces re-embedding on next run)
	find $(BOOK_DIR) -name "*.npy" -delete
	find $(BOOK_DIR) -name "*.db" -delete
	@echo "$(YELLOW)Embeddings cleared — next run will re-embed (slow).$(RESET)"

# ── Shortcuts ─────────────────────────────────────────────────────────────────
t: test-fast    ## Alias: t = test-fast
l: lint         ## Alias: l = lint
r: run          ## Alias: r = run
d: docker-build ## Alias: d = docker-build
