"""
Tests for Chapter 20: Docker + Render deployment.

    pytest tests/test_ch20.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "book" / "ch20"))

import ch20_docker_deployment as ch20  # noqa: E402
from ch20_docker_deployment import (  # noqa: E402
    LINT_RULES,
    LintResult,
    docker_build_smoke,
    docker_cli_available,
    lint_dockerfile,
    lint_repo_dockerfile,
    plot_image_size_comparison,
    plot_layer_cache_diagram,
    plot_lint_results,
    write_deploy_checklist,
)


def test_lint_rules_count():
    assert len(LINT_RULES) == 10


def test_lint_result_dataclass():
    r = LintResult("X", "desc", True, "ok")
    assert r.rule_id == "X" and r.passed is True


def test_repo_dockerfile_all_pass():
    results = lint_repo_dockerfile()
    assert len(results) == 10
    assert all(r.passed for r in results)


def test_r01_fails_without_slim():
    df = 'FROM python:3.12\nRUN pip install --no-cache-dir x\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1\nUSER app\nHEALTHCHECK CMD ["true"]\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n'
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R01_slim_base"][0]
    assert r.passed is False


def test_r02_fails_without_no_cache():
    df = (
        "FROM python:3.12-slim\nRUN pip install x\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1\n"
        'USER app\nHEALTHCHECK CMD ["true"]\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R02_pip_no_cache_dir"][0]
    assert r.passed is False


def test_r03_requires_unbuffered():
    df = (
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir x\nENV PYTHONDONTWRITEBYTECODE=1\n"
        'USER app\nHEALTHCHECK CMD ["true"]\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R03_pythonunbuffered"][0]
    assert r.passed is False


def test_r04_requires_no_bytecode_env():
    df = (
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir x\nENV PYTHONUNBUFFERED=1\n"
        'USER app\nHEALTHCHECK CMD ["true"]\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R04_pythondontwritebytecode"][0]
    assert r.passed is False


def test_r05_rejects_root_user():
    df = (
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir x\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1\n"
        'USER root\nHEALTHCHECK CMD ["true"]\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R05_non_root_user"][0]
    assert r.passed is False


def test_r06_requires_healthcheck():
    df = (
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir x\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1\n"
        'USER app\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R06_healthcheck"][0]
    assert r.passed is False


def test_r07_requires_exec_cmd():
    df = (
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir x\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1\n"
        'USER app\nHEALTHCHECK CMD ["true"]\nCMD uvicorn x:app --host 0.0.0.0\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R07_cmd_exec_form"][0]
    assert r.passed is False


def test_r08_requires_bind_all():
    df = (
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir x\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1\n"
        'USER app\nHEALTHCHECK CMD ["true"]\nCMD ["uvicorn","x:app","--host","127.0.0.1"]\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R08_bind_all_interfaces"][0]
    assert r.passed is False


def test_r09_bad_layer_order():
    df = (
        "FROM python:3.12-slim\nCOPY book/ /app/book/\nCOPY requirements.txt /\n"
        "RUN pip install --no-cache-dir -r /requirements.txt\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1\n"
        'USER app\nHEALTHCHECK CMD ["true"]\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R09_layer_order"][0]
    assert r.passed is False


def test_r10_detects_inline_secret():
    df = (
        "FROM python:3.12-slim\nENV OPENAI_API_KEY=sk-12345678901234567890123456789012\n"
        "RUN pip install --no-cache-dir x\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1\n"
        'USER app\nHEALTHCHECK CMD ["true"]\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n'
    )
    r = [x for x in lint_dockerfile(df) if x.rule_id == "R10_no_obvious_secrets"][0]
    assert r.passed is False


def test_write_deploy_checklist_creates_file(tmp_path):
    results = lint_dockerfile("FROM python:3.12-slim\n")
    p = write_deploy_checklist(results, tmp_path / "dc.md")
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "Render" in text
    assert "Docker" in text


def test_docker_cli_available_returns_bool():
    assert isinstance(docker_cli_available(), bool)


def test_plot_lint_results_writes_png(tmp_path):
    res = lint_repo_dockerfile()
    out = plot_lint_results(res, tmp_path)
    assert out.exists() and out.stat().st_size > 100


def test_plot_layer_cache_writes_png(tmp_path):
    out = plot_layer_cache_diagram(tmp_path)
    assert out.exists() and out.stat().st_size > 100


def test_plot_image_comparison_writes_png(tmp_path):
    out = plot_image_size_comparison(tmp_path)
    assert out.exists() and out.stat().st_size > 100


def test_docker_build_smoke_skips_without_cli(monkeypatch):
    monkeypatch.setattr(ch20.shutil, "which", lambda _cmd=None: None)
    ok, msg = docker_build_smoke()
    assert ok is False
    assert "not available" in msg


def test_lint_repo_custom_path(tmp_path):
    df = tmp_path / "Dockerfile.mini"
    df.write_text(
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir x\nENV PYTHONUNBUFFERED=1\n"
        'ENV PYTHONDONTWRITEBYTECODE=1\nUSER nobody\nHEALTHCHECK CMD ["true"]\n'
        'COPY requirements.txt /\nCOPY book/ /b/\nCMD ["sh","-c","uvicorn x:app --host 0.0.0.0"]\n',
        encoding="utf-8",
    )
    res = ch20.lint_repo_dockerfile(df)
    assert len(res) == 10
