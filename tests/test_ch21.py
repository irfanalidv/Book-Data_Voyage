"""
Tests for Chapter 21: CI/CD Pipeline
Run: pytest tests/test_ch21.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch21"))

from ch21_cicd_pipeline import (  # noqa: E402
    Config,
    ValidationResult,
    validate_ci_workflow,
    validate_makefile,
    write_setup_guide,
)


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.figures_dir = tmp_path / "figures"
    c.reports_dir = tmp_path / "reports"
    c.figures_dir.mkdir(parents=True)
    c.reports_dir.mkdir(parents=True)
    return c


@pytest.fixture
def good_workflow(tmp_path) -> Path:
    content = """
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pytest tests/ -v

  docker:
    needs: [test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          cache-from: type=gha
          cache-to: type=gha,mode=max
          push: false
      - run: |
          docker run -d -p 8000:8000 myapp:latest
          curl -f http://localhost:8000/health

  deploy:
    needs: [test, docker]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - run: curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
"""
    p = tmp_path / "ci.yml"
    p.write_text(content)
    return p


@pytest.fixture
def bad_workflow(tmp_path) -> Path:
    content = """
name: CI
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - run: curl -X POST "https://api.render.com/deploy/srv-xxx?key=actual-secret-abc123xyz"
"""
    p = tmp_path / "ci.yml"
    p.write_text(content)
    return p


@pytest.fixture
def good_makefile(tmp_path) -> Path:
    content = """.PHONY: test lint run docker-build deploy-check help clean

help: ## Show help
\t@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  %-20s %s\\n", $$1, $$2}'

test: ## Run tests
\tpytest tests/ -v

lint: ## Run linter
\truff check book/ tests/

run: ## Start the API
\tpython book/ch19/ch19_fastapi_deployment.py

docker-build: ## Build Docker image
\tdocker build -t talentlens:latest .

deploy-check: lint test docker-build ## Full pre-deploy check
\t@echo "Ready to deploy"

clean: ## Remove cache files
\tfind . -name __pycache__ -exec rm -rf {} +
"""
    p = tmp_path / "Makefile"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# CI workflow validation
# ---------------------------------------------------------------------------


class TestValidateCIWorkflow:
    def test_missing_file_returns_error(self, tmp_path):
        results = validate_ci_workflow(tmp_path / "nonexistent.yml")
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].severity == "error"

    def test_good_workflow_has_no_errors(self, good_workflow):
        results = validate_ci_workflow(good_workflow)
        errors = [r for r in results if not r.passed and r.severity == "error"]
        assert (
            len(errors) == 0
        ), f"Good workflow should have no errors: {[e.message for e in errors]}"

    def test_bad_workflow_fails_critical_checks(self, bad_workflow):
        results = validate_ci_workflow(bad_workflow)
        errors = [r for r in results if not r.passed and r.severity == "error"]
        assert len(errors) > 0

    def test_concurrency_check(self, tmp_path):
        # Missing concurrency
        p = tmp_path / "ci.yml"
        p.write_text("name: CI\non:\n  push:\njobs:\n  test:\n    runs-on: ubuntu-latest\n")
        results = validate_ci_workflow(p)
        r = next((r for r in results if r.check == "concurrency"), None)
        assert r is not None and not r.passed

    def test_concurrency_check_pass(self, good_workflow):
        results = validate_ci_workflow(good_workflow)
        r = next((r for r in results if r.check == "concurrency"), None)
        assert r is not None and r.passed

    def test_deploy_needs_test_check(self, good_workflow):
        results = validate_ci_workflow(good_workflow)
        r = next((r for r in results if r.check == "deploy-needs-test"), None)
        assert r is not None and r.passed

    def test_no_hardcoded_secrets_detected(self, bad_workflow):
        results = validate_ci_workflow(bad_workflow)
        r = next((r for r in results if r.check == "no-hardcoded-secrets"), None)
        assert r is not None and not r.passed

    def test_pip_cache_check(self, good_workflow):
        results = validate_ci_workflow(good_workflow)
        r = next((r for r in results if r.check == "pip-cache"), None)
        assert r is not None and r.passed

    def test_docker_cache_check(self, good_workflow):
        results = validate_ci_workflow(good_workflow)
        r = next((r for r in results if r.check == "docker-cache"), None)
        assert r is not None and r.passed

    def test_matrix_testing_check(self, good_workflow):
        results = validate_ci_workflow(good_workflow)
        r = next((r for r in results if r.check == "matrix-testing"), None)
        assert r is not None and r.passed

    def test_returns_validation_result_objects(self, good_workflow):
        results = validate_ci_workflow(good_workflow)
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_all_severities_valid(self, good_workflow):
        results = validate_ci_workflow(good_workflow)
        valid_severities = {"error", "warning", "info"}
        for r in results:
            assert r.severity in valid_severities


# ---------------------------------------------------------------------------
# Makefile validation
# ---------------------------------------------------------------------------


class TestValidateMakefile:
    def test_missing_file_returns_warning(self, tmp_path):
        results = validate_makefile(tmp_path / "nonexistent")
        assert len(results) == 1
        assert not results[0].passed

    def test_good_makefile_has_no_errors(self, good_makefile):
        results = validate_makefile(good_makefile)
        errors = [r for r in results if not r.passed and r.severity == "error"]
        assert len(errors) == 0

    def test_essential_targets_detected(self, good_makefile):
        results = validate_makefile(good_makefile)
        target_results = {r.check: r for r in results if r.check.startswith("target-")}
        for target in ["test", "lint", "run", "help", "clean"]:
            assert f"target-{target}" in target_results
            assert target_results[f"target-{target}"].passed

    def test_phony_detected(self, good_makefile):
        results = validate_makefile(good_makefile)
        r = next((r for r in results if r.check == "phony-declared"), None)
        assert r is not None and r.passed

    def test_missing_target_fails(self, tmp_path):
        p = tmp_path / "Makefile"
        p.write_text("test:\n\tpytest\n")  # Missing other targets
        results = validate_makefile(p)
        missing = [r for r in results if not r.passed and r.check.startswith("target-")]
        assert len(missing) > 0


# ---------------------------------------------------------------------------
# Setup guide
# ---------------------------------------------------------------------------


class TestWriteSetupGuide:
    def test_creates_markdown_file(self, cfg, good_workflow, good_makefile):
        ci_results = validate_ci_workflow(good_workflow)
        make_results = validate_makefile(good_makefile)
        out = write_setup_guide(cfg, ci_results, make_results)
        assert out.exists()
        assert out.suffix == ".md"

    def test_guide_contains_key_sections(self, cfg, good_workflow, good_makefile):
        ci_results = validate_ci_workflow(good_workflow)
        make_results = validate_makefile(good_makefile)
        out = write_setup_guide(cfg, ci_results, make_results)
        content = out.read_text()
        for section in ["Step 1", "Step 2", "Step 3", "Secrets", "make test"]:
            assert section in content, f"Missing section: {section}"

    def test_guide_shows_ready_status_for_good_configs(self, cfg, good_workflow, good_makefile):
        ci_results = validate_ci_workflow(good_workflow)
        make_results = validate_makefile(good_makefile)
        out = write_setup_guide(cfg, ci_results, make_results)
        content = out.read_text()
        assert "Ready" in content


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_creation_defaults(self):
        r = ValidationResult("test-check", True, "All good")
        assert r.check == "test-check"
        assert r.passed is True
        assert r.severity == "warning"

    def test_custom_severity(self):
        for sev in ("error", "warning", "info"):
            r = ValidationResult("c", False, "m", sev)
            assert r.severity == sev

    def test_message_stored(self):
        msg = "Very specific error message about the thing"
        r = ValidationResult("c", False, msg)
        assert r.message == msg
