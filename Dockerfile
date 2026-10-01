# TalentLens API — production-oriented image (Chapter 20 + 21)
# syntax=docker/dockerfile:1

# Pin by digest, not tag alone — tags can be rewritten silently (supply-chain risk).
# Regenerate after `docker pull python:3.12-slim` when intentionally upgrading the base.
FROM python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Layer order: slim runtime manifest first for better build cache.
# requirements-api.txt is the runtime-only set (no torch/transformers/spaCy/
# scikit-learn) — the ch19 API does not import the ML stack at runtime.
COPY requirements-api.txt pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code: stable package and data before churny book/ (cache-friendly)
COPY talentlens/ /app/talentlens/
# Editable install of the talentlens package ONLY. --no-deps because
# pyproject.toml declares the full ML stack (matplotlib, scikit-learn,
# requests, ...) that the API does not use at runtime; the slim runtime set
# was already installed from requirements-api.txt above.
RUN pip install --no-cache-dir --no-deps -e .
COPY data/clean/jobs_clean.csv /app/data/clean/jobs_clean.csv
COPY book/ /app/book/

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

# JSON-array exec form; Render sets PORT — default 8000 for local Docker
CMD ["sh", "-c", "exec uvicorn book.ch19.ch19_fastapi_deployment:app --host 0.0.0.0 --port ${PORT:-8000}"]
