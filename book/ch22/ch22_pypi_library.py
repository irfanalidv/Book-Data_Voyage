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
    """Diagram showing talentlens_core module structure and public API."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color="#2196F3"):
        r = plt.Rectangle(
            (x, y), w, h, facecolor=color, alpha=0.18, edgecolor=color, linewidth=2, zorder=2
        )
        ax.add_patch(r)
        ax.text(
            x + w / 2,
            y + h / 2 + (0.15 if sub else 0),
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=3,
        )
        if sub:
            ax.text(
                x + w / 2,
                y + h / 2 - 0.22,
                sub,
                ha="center",
                va="center",
                fontsize=8,
                color="#555",
                zorder=3,
            )

    def arr(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#555", lw=1.5)
        )

    # User
    box(0.2, 3.0, 1.8, 1.0, "User code", "pip install", "#9C27B0")
    # Package
    box(2.5, 0.5, 7.0, 6.0, "talentlens_core", "", "#2196F3")
    ax.text(6.0, 6.2, "talentlens_core/", ha="center", fontsize=9, color="#2196F3", style="italic")
    # Modules
    box(2.8, 3.8, 2.8, 1.5, "classification.py", "RoleClassifier\npredict_role()", "#4CAF50")
    box(6.0, 3.8, 2.8, 1.5, "search.py", "VectorStore\nsemantic_search()", "#4CAF50")
    box(2.8, 1.5, 2.8, 1.5, "cv.py", "CVParser\nparse_cv()", "#4CAF50")
    box(6.0, 1.5, 2.8, 1.5, "paths.py", "REPO_ROOT\nDATA_DIR", "#FF9800")
    # PyPI
    box(10.0, 3.0, 1.8, 1.0, "PyPI", "pip install\ntalentlens-core", "#607D8B")

    arr(2.0, 3.5, 2.5, 4.5)
    arr(2.0, 3.5, 2.5, 2.2)
    arr(9.5, 3.5, 10.0, 3.5)

    ax.set_title("talentlens-core Package Architecture", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out = cfg.figures_dir / "ch22_package_architecture.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_release_pipeline(cfg: Config) -> Path:
    """GitHub Actions release pipeline diagram."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color="#2196F3"):
        r = plt.Rectangle(
            (x, y), w, h, facecolor=color, alpha=0.18, edgecolor=color, linewidth=2, zorder=2
        )
        ax.add_patch(r)
        ax.text(
            x + w / 2,
            y + h / 2 + (0.15 if sub else 0),
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=3,
        )
        if sub:
            ax.text(
                x + w / 2,
                y + h / 2 - 0.22,
                sub,
                ha="center",
                va="center",
                fontsize=8,
                color="#555",
                zorder=3,
            )

    def arr(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#555", lw=1.8)
        )
        if label:
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 0.2, label, ha="center", fontsize=8, color="#555"
            )

    box(0.2, 1.8, 2.0, 1.4, "git tag v1.0.1", "triggers workflow", "#9C27B0")
    box(2.8, 1.8, 2.0, 1.4, "pytest tests/", "must pass", "#4CAF50")
    box(5.2, 1.8, 2.0, 1.4, "python -m build", "creates .whl\n+ .tar.gz", "#FF9800")
    box(7.6, 1.8, 2.0, 1.4, "twine check", "validates dist", "#FF9800")
    box(10.0, 1.8, 2.6, 1.4, "twine upload PyPI", "PYPI_TOKEN secret", "#2196F3")

    arr(2.2, 2.5, 2.8, 2.5, "push tag")
    arr(4.8, 2.5, 5.2, 2.5, "pass")
    arr(7.2, 2.5, 7.6, 2.5, "build ok")
    arr(9.6, 2.5, 10.0, 2.5, "valid")

    # Annotations
    ax.text(
        11.3,
        1.2,
        "pip install\ntalentlens-core\nworks",
        ha="center",
        fontsize=8.5,
        color="#2196F3",
        fontweight="bold",
    )
    ax.set_title(
        "talentlens-core Release Pipeline — GitHub Actions", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = cfg.figures_dir / "ch22_release_pipeline.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
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
