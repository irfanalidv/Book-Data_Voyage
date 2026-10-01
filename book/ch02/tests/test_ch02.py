"""Chapter 2 tests - config, paths, package import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from talentlens import __version__, get_settings
from talentlens.config import Settings
from talentlens.paths import find_repo_root


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = Settings()
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == tmp_path


def test_paths_repo_root_is_findable() -> None:
    root = find_repo_root()
    assert (root / "pyproject.toml").is_file()


def test_talentlens_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_main_script_exits_zero() -> None:
    repo_root = find_repo_root()
    script = repo_root / "book" / "ch02" / "ch02_python_setup.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Chapter 2 project scaffold" in result.stdout
