"""
Chapter 20: Docker + Render Deployment
Data Voyage - Building TalentLens

TalentLens milestone: package the FastAPI service from Chapter 19 as an OCI
image, lint the Dockerfile against production rules, document Render deploy,
and emit checklists and figures for the book.

Run from repository root:

    python book/ch20/ch20_docker_deployment.py

Outputs:

    book/ch20/reports/deploy_checklist.md
    book/ch20/reports/figures/ch20_lint_results.png
    book/ch20/reports/figures/ch20_layer_cache_diagram.png
    book/ch20/reports/figures/ch20_image_size_comparison.png
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_THIS = Path(__file__).resolve().parent
_REPO_ROOT = _THIS.parents[1]


# ---------------------------------------------------------------------------
# Dockerfile linter - 10 production rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintResult:
    """Outcome of a single Dockerfile rule."""

    rule_id: str
    description: str
    passed: bool
    detail: str


def _text(content: str) -> str:
    return content.replace("\r\n", "\n")


def rule_slim_base(content: str) -> LintResult:
    """Use a slim Python base image to reduce attack surface and image size."""
    ok = bool(
        re.search(
            r"^FROM\s+python:\d+\.\d+-slim(@sha256:[a-f0-9]+)?",
            content,
            re.MULTILINE | re.IGNORECASE,
        )
    )
    return LintResult(
        "R01_slim_base",
        "FROM uses python:X.Y-slim (optionally @sha256 digest; not full or unofficial bases)",
        ok,
        "Found slim base." if ok else "Expected e.g. FROM python:3.12-slim or ...-slim@sha256:...",
    )


def rule_pip_no_cache(content: str) -> LintResult:
    """Avoid bloating layers with pip's HTTP cache."""
    ok = bool(re.search(r"pip\s+install[^\n]*--no-cache-dir", content))
    return LintResult(
        "R02_pip_no_cache_dir",
        "pip install uses --no-cache-dir",
        ok,
        "pip uses --no-cache-dir." if ok else "Add --no-cache-dir to pip install",
    )


def rule_python_unbuffered(content: str) -> LintResult:
    """Ensure logs flush immediately in containers."""
    ok = "PYTHONUNBUFFERED=1" in content or 'PYTHONUNBUFFERED="1"' in content
    return LintResult(
        "R03_pythonunbuffered",
        "PYTHONUNBUFFERED=1 is set",
        ok,
        "Unbuffered I/O enabled." if ok else "Set ENV PYTHONUNBUFFERED=1",
    )


def rule_python_no_bytecode(content: str) -> LintResult:
    """Reduce noise and image churn from .pyc writes."""
    ok = "PYTHONDONTWRITEBYTECODE=1" in content or 'PYTHONDONTWRITEBYTECODE="1"' in content
    return LintResult(
        "R04_pythondontwritebytecode",
        "PYTHONDONTWRITEBYTECODE=1 is set",
        ok,
        "Bytecode writes disabled." if ok else "Set ENV PYTHONDONTWRITEBYTECODE=1",
    )


def rule_non_root_user(content: str) -> LintResult:
    """Processes should not run as root inside the container."""
    if "USER" not in content:
        return LintResult("R05_non_root_user", "Non-root USER directive", False, "Missing USER")
    for m in re.finditer(r"^USER\s+(\S+)\s*$", content, re.MULTILINE | re.IGNORECASE):
        name = m.group(1).lower()
        if name not in ("root", "0"):
            return LintResult(
                "R05_non_root_user",
                "Non-root USER directive",
                True,
                f"Runs as {m.group(1)}.",
            )
    return LintResult(
        "R05_non_root_user",
        "Non-root USER directive",
        False,
        "Use USER appuser (or similar), not root",
    )


def rule_healthcheck(content: str) -> LintResult:
    """Orchestrators need a probe that matches the real process."""
    ok = bool(re.search(r"^HEALTHCHECK\b", content, re.MULTILINE | re.IGNORECASE))
    return LintResult(
        "R06_healthcheck",
        "HEALTHCHECK defined",
        ok,
        "Health probe present." if ok else "Add HEALTHCHECK hitting /health",
    )


def rule_cmd_exec_form(content: str) -> LintResult:
    """Prefer JSON-array CMD for correct signal handling."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CMD "):
            ok = stripped.startswith("CMD [")
            return LintResult(
                "R07_cmd_exec_form",
                "CMD uses JSON exec form (CMD [...])",
                ok,
                "CMD is exec form." if ok else 'Use CMD ["executable", ...] not shell CMD',
            )
    return LintResult("R07_cmd_exec_form", "CMD uses JSON exec form", False, "No CMD found")


def rule_bind_host(content: str) -> LintResult:
    """Inside a container, bind 0.0.0.0 so port publishing works."""
    ok = "0.0.0.0" in content
    return LintResult(
        "R08_bind_all_interfaces",
        "Server binds 0.0.0.0",
        ok,
        "uvicorn/gunicorn binds 0.0.0.0." if ok else "Use --host 0.0.0.0",
    )


def rule_layer_order(content: str) -> LintResult:
    """Copy dependency manifests before bulk COPY so pip layer caches."""
    lines = content.splitlines()
    req_idx = next(
        (
            i
            for i, L in enumerate(lines)
            if re.search(r"COPY\s+.*requirements(-[a-z-]+)?\.txt", L, re.I)
        ),
        None,
    )
    book_idx = next(
        (i for i, L in enumerate(lines) if re.search(r"COPY\s+.*\bbook/", L, re.I)), None
    )
    if req_idx is None:
        return LintResult(
            "R09_layer_order",
            "requirements before app COPY",
            False,
            "No COPY requirements*.txt (e.g. requirements.txt, -lock, -api)",
        )
    if book_idx is None:
        return LintResult(
            "R09_layer_order", "requirements before app COPY", True, "No COPY book/ — skipped"
        )
    ok = req_idx < book_idx
    return LintResult(
        "R09_layer_order",
        "COPY requirements + pip install before COPY application code",
        ok,
        "Good layer order." if ok else "Move requirements COPY + pip install above large COPYs",
    )


def rule_no_obvious_secrets(content: str) -> LintResult:
    """Fail obvious secret literals (not exhaustive - use secret scanners in CI)."""
    bad = re.search(r"(API_KEY|SECRET|TOKEN)\s*=\s*['\"]?[a-zA-Z0-9_\-]{20,}", content)
    ok = bad is None
    return LintResult(
        "R10_no_obvious_secrets",
        "No long inline secret-like assignments",
        ok,
        "No obvious secret literals." if ok else "Move secrets to env / Render dashboard",
    )


LINT_RULES = [
    rule_slim_base,
    rule_pip_no_cache,
    rule_python_unbuffered,
    rule_python_no_bytecode,
    rule_non_root_user,
    rule_healthcheck,
    rule_cmd_exec_form,
    rule_bind_host,
    rule_layer_order,
    rule_no_obvious_secrets,
]


def lint_dockerfile(dockerfile_text: str) -> list[LintResult]:
    """Run all Dockerfile production rules."""
    text = _text(dockerfile_text)
    return [fn(text) for fn in LINT_RULES]


def lint_repo_dockerfile(path: Path | None = None) -> list[LintResult]:
    p = path or (_REPO_ROOT / "Dockerfile")
    return lint_dockerfile(p.read_text(encoding="utf-8"))


def docker_cli_available() -> bool:
    """True if `docker` is on PATH (CI sandboxes may omit it)."""
    return shutil.which("docker") is not None


def docker_build_smoke(image_tag: str = "talentlens:ch20-test") -> tuple[bool, str]:
    """Optional local `docker build` - skipped if Docker unavailable."""
    if not docker_cli_available():
        return False, "docker CLI not available"
    try:
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                image_tag,
                "-f",
                str(_REPO_ROOT / "Dockerfile"),
                str(_REPO_ROOT),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return True, f"built {image_tag}"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Deploy checklist
# ---------------------------------------------------------------------------


def write_deploy_checklist(
    lint_results: list[LintResult],
    out_path: Path | None = None,
) -> Path:
    """Write Markdown deploy checklist for Render + Docker."""
    out = out_path or (_THIS / "reports" / "deploy_checklist.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    all_pass = all(r.passed for r in lint_results)
    lint_lines = "\n".join(
        f"- [{'x' if r.passed else ' '}] **{r.rule_id}** — {r.description}: {r.detail}"
        for r in lint_results
    )

    body = f"""# TalentLens — Docker & Render deploy checklist

_Generated by Chapter 20 (`ch20_docker_deployment.py`)._

## Dockerfile lint ({len(lint_results)} rules)

**Overall:** {'**PASS** — all rules satisfied.' if all_pass else '**ATTENTION** — fix failing rules before shipping.'}

{lint_lines}

## Local Docker

1. [ ] `docker build -t talentlens-api:test .` from repository root
2. [ ] `docker run --rm -p 8000:8000 -e PORT=8000 talentlens-api:test`
3. [ ] Open `http://localhost:8000/health` — expect `{{"status":"ok",...}}`
4. [ ] `docker run ... curl` or browser check `http://localhost:8000/docs`

## Render (Blueprint)

1. [ ] Push this repo to GitHub (Render pulls from Git)
2. [ ] New **Blueprint** → select repo → Render reads `render.yaml`
3. [ ] Confirm **Dockerfile path** `./Dockerfile` and context `.`
4. [ ] Set any API keys in the Render dashboard (**Environment**), not in git
5. [ ] First deploy: watch logs until `Uvicorn running`
6. [ ] Hit `https://<service>.onrender.com/health` after cold start completes
7. [ ] **Free tier:** expect 30–60s cold start after idle sleep

## Environment variables

- Use `.env.example` as the template; keep real `.env` local only.
- On Render, mirror variables from `.env.example` with **sync: false** for secrets.

## Rollback

- Render: **Manual Deploy** → select previous successful deploy.
- Docker registry: retag previous digest if you push versioned images.

---
*Next: wire Makefile + GitHub Actions so every push runs tests and optional deploy.*
"""
    out.write_text(body, encoding="utf-8")
    logger.info("Wrote %s", out)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_lint_results(results: list[LintResult], out_dir: Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    out_dir = out_dir or (_THIS / "reports" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r.rule_id.replace("R0", "").replace("_", " ") for r in results]
    colours = ["#4CAF50" if r.passed else "#F44336" for r in results]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = range(len(results))
    ax.barh(y, [1] * len(results), color=colours, edgecolor="white", height=0.65)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_title("Dockerfile lint — 10 production rules", fontsize=13, fontweight="bold")
    for i, r in enumerate(results):
        ax.text(
            0.5,
            i,
            "PASS" if r.passed else "FAIL",
            va="center",
            ha="center",
            color="white",
            fontweight="bold",
            fontsize=8,
        )
    fig.tight_layout()
    p = out_dir / "ch20_lint_results.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)
    return p


def plot_layer_cache_diagram(out_dir: Path | None = None) -> Path:
    """Illustrate why dependency layers cache independently."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    out_dir = out_dir or (_THIS / "reports" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def layer(y, h, label, color):
        r = mpatches.FancyBboxPatch(
            (0.3, y), 9.4, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#333"
        )
        ax.add_patch(r)
        ax.text(5, y + h / 2, label, ha="center", va="center", fontsize=7.5, fontweight="bold")

    layer(3.6, 0.9, "Layer 3: COPY book/ … (invalidates often)", "#FFCC80")
    layer(2.4, 0.9, "Layer 2: RUN pip install … (cached until requirements change)", "#A5D6A7")
    layer(1.2, 0.9, "Layer 1: COPY requirements.txt (tiny manifest)", "#90CAF9")

    ax.set_title("Docker layer cache: order matters", fontsize=10.5, fontweight="bold", pad=8)
    ax.text(
        5,
        0.35,
        "Change app code: only Layer 3 rebuilds. Change dependencies: Layers 2 and 3 rebuild.",
        ha="center",
        fontsize=7.5,
        color="#555",
    )
    p = out_dir / "ch20_layer_cache_diagram.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)
    return p


def plot_image_size_comparison(out_dir: Path | None = None) -> Path:
    """Measured image sizes: full ML stack vs the slim runtime set (Chapter 20)."""
    import matplotlib.pyplot as plt

    out_dir = out_dir or (_THIS / "reports" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Full book stack\n(requirements-lock.txt)", "Slim runtime\n(requirements-api.txt)"]
    # Measured with `docker image ls` on python:3.12-slim: the full-stack build
    # during the book's first deployment attempt, the slim build on the current
    # Dockerfile. Re-measure after changing either requirements file.
    sizes = [6300, 462]
    colours = ["#EF5350", "#42A5F5"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(labels, sizes, color=colours, edgecolor="white")
    ax.set_ylabel("Image size (MB)")
    ax.set_title("Serving image: ship only what the API imports", fontsize=13, fontweight="bold")
    for i, v in enumerate(sizes):
        label = f"{v / 1000:.1f} GB" if v >= 1000 else f"{v} MB"
        ax.text(i, v + 80, label, ha="center", fontweight="bold")
    fig.tight_layout()
    p = out_dir / "ch20_image_size_comparison.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def plot_deployment_architecture(out_dir: Path | None = None) -> Path:
    """From git push to a public URL: GitHub Actions, Render, container, health check."""
    from talentlens.diagrams import Box, Diagram

    out_dir = out_dir or (_THIS / "reports" / "figures")
    d = Diagram(6.6, 2.3, "From git push to a public URL")
    w, h, y = 1.1, 0.72, 0.98
    d.box("dev", Box(0.1, y, w, h, "Developer", "git push to main", "input"))
    d.box("ci", Box(1.4, y, w, h, "GitHub Actions", "tests, lint,\nimage check", "step"))
    d.box("render", Box(2.7, y, w, h, "Render", "builds the\nDockerfile", "step"))
    d.box("app", Box(4.0, y, w, h, "Container", "python:3.12-slim\nuvicorn on $PORT", "output"))
    d.box("users", Box(5.3, y, w, h, "Users", "HTTPS requests", "input"))
    d.arrow("dev", "ci")
    d.arrow("ci", "render")
    d.arrow("render", "app")
    d.arrow("users", "app")
    d.note(
        0.1,
        0.42,
        "Render deploys only after CI passes (the deploy hook is the last job).\n"
        "Render's health check calls GET /health on the same app users hit,\n"
        "and the Dockerfile HEALTHCHECK probes it every 30 s.",
    )
    path = d.save(out_dir / "ch20_deployment_architecture.png")
    logger.info("Saved: %s", path)
    return path


def plot_docker_layer_cache(out_dir: Path | None = None) -> Path:
    """The repository Dockerfile, step by step, marking which layers a code edit rebuilds."""
    from talentlens.diagrams import Box, Diagram

    out_dir = out_dir or (_THIS / "reports" / "figures")
    d = Diagram(6.6, 3.55, "Layer order decides what a code change rebuilds")
    d.note(0.15, 3.05, "Copy everything first", ha="left", size=8.5)
    d.note(3.45, 3.05, "The TalentLens Dockerfile", ha="left", size=8.5)
    slow = [
        ("FROM python:3.12-slim", "base image", "good"),
        ("COPY . /app", "changes on every commit", "bad"),
        ("RUN pip install -r ...", "re-runs on every commit", "bad"),
    ]
    ours = [
        ("FROM python:3.12-slim", "base image, pinned by digest", "good"),
        ("COPY requirements-api.txt ...", "changes rarely", "good"),
        ("RUN pip install -r ...", "cached unless deps change", "good"),
        ("COPY talentlens/ + pip -e .", "rebuilt when talentlens/ changes", "bad"),
        ("COPY data/ and book/", "rebuilt when you edit code", "bad"),
    ]
    for col, steps, x in (("s", slow, 0.15), ("o", ours, 3.45)):
        for i, (title, sub, kind) in enumerate(steps):
            d.box(f"{col}{i}", Box(x, 2.38 - i * 0.55, 3.0, 0.46, title, sub, kind))
    d.note(
        0.15,
        0.62,
        "Green: reused from cache.  Red: rebuilt.\n"
        "Edit a Python file and only the last layers of\nthe right-hand order rebuild.",
        size=7.2,
    )
    path = d.save(out_dir / "ch20_docker_layer_cache.png")
    logger.info("Saved: %s", path)
    return path


def main() -> None:
    logger.info("Chapter 20 — Docker + Render")
    results = lint_repo_dockerfile()
    for r in results:
        logger.info("  %s %s", "OK " if r.passed else "FAIL", r.rule_id)

    write_deploy_checklist(results)
    plot_lint_results(results)
    plot_layer_cache_diagram()
    plot_image_size_comparison()
    plot_deployment_architecture()
    plot_docker_layer_cache()

    if docker_cli_available():
        logger.info(
            "Docker CLI detected — optional smoke build skipped by default (set CH20_DOCKER_BUILD=1)."
        )
    import os

    if os.environ.get("CH20_DOCKER_BUILD") == "1":
        ok, msg = docker_build_smoke()
        logger.info("docker build: %s — %s", ok, msg)


if __name__ == "__main__":
    main()
