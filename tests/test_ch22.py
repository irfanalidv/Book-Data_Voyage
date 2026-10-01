"""
Tests for Chapter 22: talentlens-core package (monorepo wiring).
Run: pytest tests/test_ch22.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch22"))

from talentlens_core import (  # noqa: E402
    __version__,
    connect_vector_store,
    create_app,
    create_cv_parser,
    find_repo_root,
    predict_job_role,
)
from talentlens_core.paths import ensure_repo_on_sys_path  # noqa: E402


def test_version():
    assert __version__ == "0.1.0"


def test_find_repo_root():
    root = find_repo_root()
    assert root is not None
    assert (root / "book" / "ch19" / "ch19_fastapi_deployment.py").is_file()


def test_ensure_repo_on_sys_path():
    assert ensure_repo_on_sys_path() is not None
    assert str(_REPO_ROOT) in sys.path


def test_create_app():
    app = create_app(rate_limit_max=500)
    assert app.title == "TalentLens API"


def test_create_cv_parser_stub():
    parser = create_cv_parser(provider="stub")
    assert parser is not None


def test_connect_vector_store():
    store = connect_vector_store()
    assert store._conn is not None
    store.close()


def test_predict_job_role_with_model(tmp_path, monkeypatch):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100)),
            ("clf", LogisticRegression(max_iter=200, random_state=0)),
        ]
    )
    texts = ["machine learning pytorch models production"] * 6 + ["spark etl sql pipelines"] * 6
    labels = ["ML Engineer"] * 6 + ["Data Engineer"] * 6
    pipe.fit(texts, labels)
    model_path = tmp_path / "role_classifier.joblib"
    joblib.dump(pipe, model_path)
    monkeypatch.setenv("TALENTLENS_ROLE_MODEL", str(model_path))
    out = predict_job_role(
        {"title": "ML Engineer", "description": "pytorch models", "company": "Co"}
    )
    # predict_job_role returns (label, confidence) tuple
    assert isinstance(out, (tuple, dict)) and len(out) == 2


def test_predict_job_role_missing_file(monkeypatch):
    monkeypatch.setenv("TALENTLENS_ROLE_MODEL", "/nonexistent/joblib")
    with pytest.raises(FileNotFoundError):
        predict_job_role({"title": "x", "description": "y", "company": "z"})
