"""
Chapter 21: CI/CD - GitHub Actions and Makefile
Data Voyage - Building TalentLens

TalentLens milestone: automated pipeline - every push to main runs tests,
builds Docker, and deploys to Render without manual steps.

This script validates the CI configuration files and generates
pipeline architecture diagrams.

Run:
    python book/ch21/ch21_cicd_pipeline.py

Outputs:
    book/ch21/reports/figures/ch21_pipeline_architecture.png
    book/ch21/reports/figures/ch21_pipeline_timing.png
    book/ch21/reports/cicd_setup_guide.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _find_repo_file(candidates: list[Path]) -> Path:
    """Return the first existing path from candidates, else the first."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


@dataclass
class Config:
    ci_workflow_path: Path = field(
        default_factory=lambda: _find_repo_file(
            [
                _REPO_ROOT / ".github" / "workflows" / "ci.yml",
                _THIS_DIR / ".github" / "workflows" / "ci.yml",
            ]
        )
    )
    pr_workflow_path: Path = field(
        default_factory=lambda: _find_repo_file(
            [
                _REPO_ROOT / ".github" / "workflows" / "pr-check.yml",
                _THIS_DIR / ".github" / "workflows" / "pr-check.yml",
            ]
        )
    )
    makefile_path: Path = field(
        default_factory=lambda: _find_repo_file(
            [
                _REPO_ROOT / "Makefile",
                _THIS_DIR / "Makefile",
            ]
        )
    )
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"


# ---------------------------------------------------------------------------
# CI config validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    check: str
    passed: bool
    message: str
    severity: str = "warning"


def validate_ci_workflow(path: Path) -> list[ValidationResult]:
    """Validate a GitHub Actions workflow file for common issues.

    Args:
        path: Path to the workflow YAML file.

    Returns:
        List of ValidationResult objects.
    """
    if not path.exists():
        return [ValidationResult("file-exists", False, f"Workflow not found at {path}", "error")]

    content = path.read_text()
    results: list[ValidationResult] = []

    def check(name: str, cond: bool, ok: str, fail: str, sev: str = "warning") -> None:
        results.append(ValidationResult(name, cond, ok if cond else fail, sev))

    # Concurrency configured
    check(
        "concurrency",
        "concurrency:" in content and "cancel-in-progress: true" in content,
        "Concurrency configured — parallel deploys won't race",
        "Add concurrency.cancel-in-progress: true to prevent racing deploys",
        "error",
    )

    # Tests before deploy
    deploy_section = content[content.find("deploy:") :] if "deploy:" in content else ""
    check(
        "deploy-needs-test",
        "needs:" in deploy_section and "test" in deploy_section,
        "Deploy job depends on test — broken code won't deploy",
        "Deploy job should have 'needs: [test]' to prevent deploying broken code",
        "error",
    )

    # Secrets not hardcoded (no obvious tokens in file)
    # Look for hardcoded secrets: API keys, tokens, URL params with secret-looking values
    _secret_patterns = [
        r"sk-[A-Za-z0-9]{20,}",  # OpenAI sk- keys
        r"gsk_[A-Za-z0-9]{20,}",  # Groq keys
        r"[?&]key=[A-Za-z0-9_-]{8,}",  # URL ?key=secret params
        r"token[_\-]?[A-Za-z0-9]{16,}",  # token patterns
        r"Bearer [A-Za-z0-9._\-]{16,}",  # Bearer tokens
        r"Authorization.*[A-Za-z0-9]{24,}",  # Auth headers with long values
    ]
    obvious_secrets = []
    for _pat in _secret_patterns:
        obvious_secrets.extend(re.findall(_pat, content, re.IGNORECASE))
    check(
        "no-hardcoded-secrets",
        len(obvious_secrets) == 0,
        "No hardcoded secrets detected",
        f"Possible hardcoded secrets found: {obvious_secrets[:2]}",
        "error",
    )

    # Uses secrets for deploy
    check(
        "uses-secrets",
        "secrets." in content,
        "Secrets referenced via ${{ secrets.* }} — not hardcoded",
        "Consider using ${{ secrets.* }} for any sensitive values",
        "warning",
    )

    # pip caching
    check(
        "pip-cache",
        "cache: pip" in content or "cache-dependency-path" in content,
        "pip caching configured — faster CI runs",
        "Add 'cache: pip' to setup-python to speed up dependency installation",
        "warning",
    )

    # Docker layer caching
    check(
        "docker-cache",
        "type=gha" in content or "cache-from" in content,
        "Docker layer caching configured — faster image builds",
        "Add cache-from/cache-to to docker/build-push-action for faster builds",
        "warning",
    )

    # Health check in docker job
    check(
        "container-health-check",
        "/health" in content and ("curl" in content or "wget" in content),
        "Container health verified in pipeline",
        "Add a health check step after starting the container in the docker job",
        "warning",
    )

    # Matrix testing
    check(
        "matrix-testing",
        "matrix:" in content and "python-version" in content,
        "Matrix testing across Python versions",
        "Consider testing across multiple Python versions with strategy.matrix",
        "info",
    )

    return results


def validate_makefile(path: Path) -> list[ValidationResult]:
    """Validate Makefile for essential targets.

    Args:
        path: Path to Makefile.

    Returns:
        List of ValidationResult objects.
    """
    if not path.exists():
        return [ValidationResult("file-exists", False, f"Makefile not found at {path}", "warning")]

    content = path.read_text()
    results: list[ValidationResult] = []

    def check(name: str, cond: bool, ok: str, fail: str, sev: str = "warning") -> None:
        results.append(ValidationResult(name, cond, ok if cond else fail, sev))

    essential_targets = ["test", "lint", "run", "docker-build", "deploy-check", "help", "clean"]
    for target in essential_targets:
        check(
            f"target-{target}",
            f"{target}:" in content,
            f"Target '{target}' defined",
            f"Add a '{target}:' target to the Makefile",
            "warning",
        )

    check(
        "phony-declared",
        ".PHONY" in content,
        ".PHONY declared — targets won't conflict with files",
        "Add .PHONY declaration for non-file targets",
        "warning",
    )

    check(
        "help-target",
        "help:" in content and "##" in content,
        "Help target with ## annotations — self-documenting",
        "Add ## comments after targets for automatic help generation",
        "info",
    )

    return results


def print_results(results: list[ValidationResult], title: str) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  {title.upper()}\n{sep}")
    for r in results:
        icon = "PASS" if r.passed else ("ERR " if r.severity == "error" else "WARN")
        logger.info(f"  [{icon}] {r.check:<30} {r.message}")
    errors = sum(1 for r in results if not r.passed and r.severity == "error")
    warnings = sum(1 for r in results if not r.passed and r.severity == "warning")
    passed = sum(1 for r in results if r.passed)
    logger.info(f"\n  {passed} passed | {warnings} warnings | {errors} errors")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def plot_pipeline_architecture(cfg: Config) -> Path:
    """CI/CD architecture: what runs on a push to main and on a pull request."""
    from talentlens.diagrams import Box, Diagram

    d = Diagram(6.6, 3.3, "CI/CD: what runs on every push to main")
    d.box("push", Box(0.1, 1.85, 1.05, 0.6, "git push", "to main", "input"))
    d.box("pr", Box(0.1, 0.4, 1.05, 0.6, "pull request", "into main", "input"))
    d.group(1.4, 1.32, 1.5, 1.62, "in parallel")
    d.box("t311", Box(1.5, 2.3, 1.3, 0.36, "test, Python 3.11", kind="step"))
    d.box("t312", Box(1.5, 1.88, 1.3, 0.36, "test, Python 3.12", kind="step"))
    d.box("lint", Box(1.5, 1.46, 1.3, 0.36, "lint: ruff + black", kind="step"))
    d.box(
        "docker",
        Box(3.15, 1.79, 1.2, 0.72, "docker", "build the image,\nhit /health + 2 routes", "step"),
    )
    d.box("deploy", Box(4.6, 1.79, 0.95, 0.72, "deploy", "Render\ndeploy hook", "step"))
    d.box("live", Box(5.75, 1.79, 0.75, 0.72, "Live", "public URL", "output"))
    d.box("prcheck", Box(1.4, 0.4, 1.5, 0.6, "tests + lint", "nothing deploys", "good"))
    d.arrow("push", "t312", start=(1.15, 2.15), end=(1.4, 2.15))
    d.arrow("t312", "docker", start=(2.9, 2.15), end=(3.15, 2.15))
    d.arrow("docker", "deploy")
    d.arrow("deploy", "live")
    d.arrow("pr", "prcheck")
    d.note(
        3.15,
        0.7,
        "Docker waits for all three test and lint jobs.\nDocker and deploy run only on a push to main;\na failed job stops everything after it.",
    )
    out = d.save(cfg.figures_dir / "ch21_pipeline_architecture.png")
    logger.info(f"Saved: {out}")
    return out


def plot_pipeline_timing(cfg: Config) -> Path:
    """Gantt-style chart showing job timing and parallelism."""
    fig, ax = plt.subplots(figsize=(12, 5))

    jobs = [
        ("test (3.11)", 0, 107, "#4CAF50"),
        ("test (3.12)", 0, 112, "#66BB6A"),
        ("lint", 0, 38, "#FF9800"),
        ("docker build", 115, 415, "#2196F3"),
        ("deploy", 418, 441, "#F44336"),
    ]

    for i, (label, start, end, color) in enumerate(jobs):
        ax.barh(i, end - start, left=start, color=color, alpha=0.75, edgecolor="white", height=0.55)
        ax.text(
            start + (end - start) / 2,
            i,
            f"{end-start}s",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        ax.text(-5, i, label, ha="right", va="center", fontsize=10)

    # Dependency markers
    ax.axvline(115, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(115, len(jobs) - 0.3, "test+lint\npassed", ha="center", fontsize=8, color="gray")
    ax.axvline(418, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(418, len(jobs) - 0.3, "docker\npassed", ha="center", fontsize=8, color="gray")

    # Total time marker
    ax.axvline(441, color="#4CAF50", linestyle="-", linewidth=2, alpha=0.8)
    ax.text(
        443, -0.8, "Live\n~7 min total", ha="left", fontsize=9, fontweight="bold", color="#4CAF50"
    )

    ax.set_xlabel("Time (seconds from push)", fontsize=11)
    ax.set_yticks([])
    ax.set_title(
        "CI/CD Pipeline Timing — Parallelism cuts wall-clock time", fontsize=13, fontweight="bold"
    )
    ax.set_xlim(-80, 520)
    ax.set_ylim(-1.2, len(jobs) + 0.5)

    # Parallelism annotation
    ax.annotate(
        "",
        xy=(112, 3.8),
        xytext=(0, 3.8),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.5),
    )
    ax.text(56, 4.1, "test + lint run in parallel", ha="center", fontsize=8.5, color="#555555")

    plt.tight_layout()
    out = cfg.figures_dir / "ch21_pipeline_timing.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Setup guide
# ---------------------------------------------------------------------------


def write_setup_guide(
    cfg: Config, ci_results: list[ValidationResult], make_results: list[ValidationResult]
) -> Path:
    """Write a CI/CD setup guide with current validation status.

    Args:
        cfg: Config with reports_dir.
        ci_results: CI workflow validation results.
        make_results: Makefile validation results.

    Returns:
        Path to saved guide.
    """
    ci_errors = sum(1 for r in ci_results if not r.passed and r.severity == "error")
    make_errors = sum(1 for r in make_results if not r.passed and r.severity == "error")

    ci_status = "Ready" if ci_errors == 0 else f"{ci_errors} issues to fix"
    make_status = "Ready" if make_errors == 0 else f"{make_errors} issues to fix"

    guide = f"""# TalentLens CI/CD Setup Guide

## Current status
| File | Status |
|------|--------|
| `.github/workflows/ci.yml` | {ci_status} |
| `Makefile` | {make_status} |

## Step 1 — Commit the CI files

```bash
git add .github/workflows/ci.yml .github/workflows/pr-check.yml Makefile
git commit -m "feat: add CI/CD pipeline (GitHub Actions + Makefile)"
git push origin main
```

Visit **GitHub → Actions tab** to watch the first run.

## Step 2 — Add GitHub Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions**

| Secret name | Where to get it | Required? |
|------------|-----------------|-----------|
| `RENDER_DEPLOY_HOOK_URL` | Render → service → Settings → Deploy Hook | Yes |
| `TALENTLENS_PRODUCTION_URL` | Your Render service URL | For verification |

## Step 3 — Verify the pipeline

```bash
# Locally — mirror what CI does
make test-fast    # same as CI test job
make lint         # CI lint gate (ruff E,F,W,I; black warns, does not fail)
make docker-build # same as CI docker job
# optional: make lint-all — full pyproject ruff; exploratory, often fails (N806 vs ML X/y/ax)

# If all green:
git push origin main  # pipeline runs automatically
```

**Lint tiers:** `make lint` matches CI (ruff E,F,W,I on book/tests/talentlens, E501 ignored; black warns only). `make lint-all` runs the full pyproject ruleset — exploratory, not a gate; N806 conflicts with ML conventions (X, y, ax).

## Step 4 — Set up branch protection

Go to: **GitHub repo → Settings → Branches → Add rule for `main`**

Recommended rules:
- [x] Require status checks to pass before merging
  - Required checks: `Test (3.11)`, `Code Quality`
- [x] Require branches to be up to date before merging
- [x] Require pull request reviews (1 approver)

This prevents anyone (including yourself) from pushing directly to
`main` without passing CI. All changes go through PRs.

## What to expect

**On every `git push origin main`:**
```
✅ Test (3.11)     ~2 min
✅ Test (3.12)     ~2 min  (parallel)
✅ Code Quality    ~40s    (parallel)
✅ Docker Build    ~5 min
✅ Deploy          ~3 min
Total: ~10 min from push to production
```

**On every pull request:**
```
✅ Tests           ~2 min
✅ Lint            ~40s
Total: ~2 min feedback on your PR
```

**If a step fails:**
- Pipeline stops — nothing broken reaches production
- GitHub sends an email notification
- Check Actions tab for the failing step and its logs
- Fix, push, pipeline re-runs automatically

## Common setup issues

**"Permission denied" on Makefile targets:**
```bash
chmod +x  # Not needed — make runs shell directly
# If the issue is on Windows: use WSL or Git Bash
```

**"No such file: .github/workflows/ci.yml":**
```bash
# Create the directories first
mkdir -p .github/workflows
# Then commit the YAML files
```

**Pipeline runs but deploy doesn't trigger:**
```bash
# Check that RENDER_DEPLOY_HOOK_URL secret is set
# Check that the deploy job has: if: github.ref == 'refs/heads/main'
# Check the deploy job's 'needs' list includes all preceding jobs
```

**Docker build fails in CI but works locally:**
```bash
# Most common causes:
# 1. Different Docker version — CI uses latest stable
# 2. Platform mismatch — CI uses linux/amd64, Mac uses arm64
# Add to Dockerfile or build args: --platform linux/amd64
```
"""

    out = cfg.reports_dir / "cicd_setup_guide.md"
    out.write_text(guide, encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 60)
    logger.info("  CHAPTER 21: CI/CD PIPELINE")
    logger.info("  GitHub Actions + Makefile")
    logger.info("=" * 60)

    cfg = Config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n[1/5] Validating CI workflow...")
    ci_results = validate_ci_workflow(cfg.ci_workflow_path)
    print_results(ci_results, "GitHub Actions CI Workflow Validation")

    logger.info("\n[2/5] Validating Makefile...")
    make_results = validate_makefile(cfg.makefile_path)
    print_results(make_results, "Makefile Validation")

    logger.info("\n[3/5] Generating pipeline architecture diagram...")
    plot_pipeline_architecture(cfg)

    logger.info("\n[4/5] Generating pipeline timing diagram...")
    plot_pipeline_timing(cfg)

    logger.info("\n[5/5] Writing setup guide...")
    write_setup_guide(cfg, ci_results, make_results)

    ci_errors = sum(1 for r in ci_results if not r.passed and r.severity == "error")
    make_errors = sum(1 for r in make_results if not r.passed and r.severity == "error")

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 21 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  CI workflow:  {'READY' if ci_errors == 0 else f'{ci_errors} errors'}")
    logger.info(f"  Makefile:     {'READY' if make_errors == 0 else f'{make_errors} errors'}")
    logger.info(f"  Figures  → {cfg.figures_dir}/")
    logger.info(f"  Guide    → {cfg.reports_dir}/cicd_setup_guide.md")
    logger.info("\n  Next: git push origin main → watch the pipeline run")
    logger.info("  See book README for the full chapter map (TalentLens milestones).")


if __name__ == "__main__":
    main()
