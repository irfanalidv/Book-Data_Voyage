"""
Chapter 22: Packaging as a PyPI Library - talentlens-core
Data Voyage - Building TalentLens

TalentLens milestone: extract the reusable core into talentlens-core,
a real Python package anyone can pip install. Covers pyproject.toml,
versioning, the GitHub Actions release pipeline, and PyPI publish.

Run: python book/ch22/ch22_pypi_library.py

Outputs:
    book/ch22/reports/figures/ch22_package_architecture.png
    book/ch22/reports/figures/ch22_release_pipeline.png
    book/ch22/reports/pypi_publish_guide.md
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)


@dataclass
class Config:
    package_dir: Path = _THIS_DIR / "talentlens_core"
    pyproject_path: Path = _THIS_DIR / "pyproject.toml"
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"


# ---------------------------------------------------------------------------
# Package validation
# ---------------------------------------------------------------------------


def validate_package_structure(cfg: Config) -> list[dict]:
    """Check that talentlens_core has the required files and structure.

    Args:
        cfg: Config with package_dir and pyproject_path.

    Returns:
        List of check result dicts with name, passed, message.
    """
    results = []

    def check(name: str, condition: bool, ok: str, fail: str) -> None:
        results.append({"name": name, "passed": condition, "message": ok if condition else fail})

    # Package directory
    check(
        "package_dir_exists",
        cfg.package_dir.exists(),
        f"talentlens_core/ found at {cfg.package_dir}",
        f"talentlens_core/ not found at {cfg.package_dir}",
    )

    # __init__.py
    init = cfg.package_dir / "__init__.py"
    check(
        "has_init",
        init.exists(),
        "__init__.py present",
        "__init__.py missing — package won't import correctly",
    )

    # pyproject.toml
    check(
        "has_pyproject",
        cfg.pyproject_path.exists(),
        "pyproject.toml present",
        "pyproject.toml missing — required for modern Python packaging",
    )

    if cfg.pyproject_path.exists():
        toml_content = cfg.pyproject_path.read_text()
        check(
            "pyproject_has_name",
            "[project]" in toml_content and "name" in toml_content,
            "pyproject.toml has [project] section with name",
            "pyproject.toml missing [project] section",
        )
        check(
            "pyproject_has_version",
            "version" in toml_content,
            "pyproject.toml has version",
            "pyproject.toml missing version field",
        )
        check(
            "pyproject_has_description",
            "description" in toml_content,
            "pyproject.toml has description",
            "pyproject.toml missing description",
        )

    # Key module files
    for module in ["classification", "search", "cv"]:
        module_path = cfg.package_dir / f"{module}.py"
        check(
            f"module_{module}",
            module_path.exists(),
            f"talentlens_core/{module}.py present",
            f"talentlens_core/{module}.py missing",
        )

    # Can we import it?
    try:
        sys.path.insert(0, str(_THIS_DIR))
        import talentlens_core  # noqa

        check(
            "importable",
            True,
            f"talentlens_core imports successfully (v{getattr(talentlens_core, '__version__', 'unknown')})",
            "",
        )
    except ImportError as e:
        check("importable", False, "", f"Import failed: {e}")
    finally:
        if str(_THIS_DIR) in sys.path:
            sys.path.remove(str(_THIS_DIR))

    return results


def print_validation_results(results: list[dict]) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("  PACKAGE VALIDATION")
    logger.info("=" * 60)
    for r in results:
        icon = "PASS" if r["passed"] else "FAIL"
        logger.info(f"  [{icon}] {r['name']:<30} {r['message']}")
    passed = sum(1 for r in results if r["passed"])
    logger.info(f"\n  {passed}/{len(results)} checks passed")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def plot_package_architecture(cfg: Config) -> Path:
    """talentlens_core's public API and the chapter modules each wrapper imports."""
    from talentlens.diagrams import Box, Diagram

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    d = Diagram(6.6, 3.75, "talentlens_core: a thin public API over chapter code")
    d.box(
        "user", Box(0.1, 1.55, 1.25, 0.8, "Your code", "from talentlens_core\nimport ...", "input")
    )
    d.group(1.65, 0.2, 2.55, 3.1, "talentlens_core/  (exported in __init__.py)")
    modules = [
        (
            "cls",
            "classification.py",
            "load_role_classifier()\npredict_job_role()",
            "ch09",
            "Ch 9 classifier",
        ),
        ("srch", "search.py", "connect_vector_store()", "ch16", "Ch 16 VectorStore"),
        ("cv", "cv.py", "create_cv_parser()", "ch17", "Ch 17 CVParser"),
        ("app", "app.py", "create_app()", "ch19", "Ch 19 build_app()"),
    ]
    for k, (key, title, sub, ch, target) in enumerate(modules):
        y = 2.45 - k * 0.66
        d.box(key, Box(1.8, y, 2.25, 0.56, title, sub, "step"))
        d.box(ch, Box(4.75, y, 1.75, 0.56, target, f"book/{ch}/", "store"))
        d.arrow(key, ch, dashed=True)
    d.note(1.8, 0.33, "paths.py: find_repo_root(), ensure_repo_on_sys_path()", size=6.8)
    d.arrow("user", "cls", start=(1.35, 2.05), end=(1.8, 2.6))
    d.arrow("user", "app", start=(1.35, 1.85), end=(1.8, 0.7))
    d.note(4.75, 3.15, "wraps (imports at call time)", size=7)
    out = d.save(cfg.figures_dir / "ch22_package_architecture.png")
    logger.info(f"Saved: {out}")
    return out


def plot_release_pipeline(cfg: Config) -> Path:
    """Release flow in .github/workflows/publish.yml: tag -> build -> TestPyPI -> PyPI."""
    from talentlens.diagrams import Box, Diagram

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    d = Diagram(6.7, 2.35, "Releasing talentlens-core from a git tag")
    w, h, y = 1.15, 0.72, 0.95
    d.box("tag", Box(0.1, y, w, h, "git tag", "talentlens-core-\nv0.1.0", "input"))
    d.box("build", Box(1.45, y, w, h, "build job", "version check,\nbuild, twine check", "step"))
    d.box("test", Box(2.8, y, w, h, "TestPyPI", "trusted\npublishing", "step"))
    d.box("pypi", Box(4.15, y, w, h, "PyPI", "trusted\npublishing", "output"))
    d.box("user", Box(5.5, y, w, h, "pip install", "talentlens-core", "input"))
    d.arrow("tag", "build")
    d.arrow("build", "test")
    d.arrow("test", "pypi")
    d.arrow("pypi", "user")
    d.note(
        0.1,
        0.4,
        "No API tokens are stored: PyPI trusts this repository's publish workflow (OIDC).\n"
        "PyPI is published only after TestPyPI succeeds, and the build fails if the tag\n"
        "does not match the version in pyproject.toml.",
    )
    out = d.save(cfg.figures_dir / "ch22_release_pipeline.png")
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Publish guide
# ---------------------------------------------------------------------------


def write_publish_guide(results: list[dict], cfg: Config) -> Path:
    """Write the step-by-step PyPI publish guide."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r["passed"])
    ready = passed == len(results)

    guide = f"""# talentlens-core — PyPI Publish Guide

## Package validation: {passed}/{len(results)} checks passed {'✅' if ready else '⚠️'}

## Step 1 — Install build tools

```bash
pip install build twine
```

## Step 2 — Build the distribution

```bash
cd book/ch22
python -m build
# Creates: dist/talentlens_core-X.Y.Z-py3-none-any.whl
#          dist/talentlens_core-X.Y.Z.tar.gz
```

## Step 3 — Test on TestPyPI first

```bash
# Register at test.pypi.org (free)
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ talentlens-core
python -c "import talentlens_core; print(talentlens_core.__version__)"
```

## Step 4 — Publish to real PyPI

```bash
# Register at pypi.org (free)
# Generate an API token under Account Settings
twine upload dist/*
# Enter: __token__ as username, your token as password
```

## Step 5 — Verify

```bash
pip install talentlens-core
python -c "from talentlens_core import classify_role; print(classify_role('ML Engineer'))"
```

## Step 6 — Set up automated releases (GitHub Actions)

Add `PYPI_API_TOKEN` to GitHub repo Secrets.
The workflow in `book/ch22/.github/workflows/publish.yml` triggers on
`git tag v*` and publishes automatically.

```bash
git tag v1.0.1
git push origin v1.0.1
# GitHub Actions runs tests → build → publish
```

## Versioning convention

- `v1.0.0` — first public release
- `v1.0.1` — bug fix
- `v1.1.0` — new feature (backwards compatible)
- `v2.0.0` — breaking change

## Your package URL

Once published: https://pypi.org/project/talentlens-core/
"""
    out = cfg.reports_dir / "pypi_publish_guide.md"
    out.write_text(guide, encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = Config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  CHAPTER 22: PACKAGING AS A PYPI LIBRARY")
    logger.info("  talentlens-core — From Repo to pip install")
    logger.info("=" * 60)

    logger.info("\n[1/4] Validating package structure...")
    results = validate_package_structure(cfg)
    print_validation_results(results)

    logger.info("\n[2/4] Plotting package architecture...")
    plot_package_architecture(cfg)

    logger.info("\n[3/4] Plotting release pipeline...")
    plot_release_pipeline(cfg)

    logger.info("\n[4/4] Writing publish guide...")
    write_publish_guide(results, cfg)

    passed = sum(1 for r in results if r["passed"])
    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 22 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Package checks: {passed}/{len(results)} passed")
    logger.info(f"  Figures: {cfg.figures_dir}/")
    logger.info(f"  Guide:   {cfg.reports_dir}/pypi_publish_guide.md")
    logger.info("\n  To publish: pip install build twine && python -m build")
    logger.info("  Then: twine upload dist/*")
    logger.info("\nNext: Chapter 23 — Real-World Case Studies")


if __name__ == "__main__":
    main()
