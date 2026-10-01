"""
Tests for Chapter 9: Supervised Learning - Role Classifier
Run: pytest tests/test_ch09.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

_THIS_FILE = Path(__file__).resolve()
for _candidate in [
    _THIS_FILE.parents[1] / "book" / "ch09",
    _THIS_FILE.parents[1],
]:
    if (_candidate / "ch09_supervised_learning.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from ch09_supervised_learning import (  # noqa: E402
    Config,
    _generate_demo_data,
    build_feature_text,
    build_pipeline,
    load_data,
    predict_role,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.model_path = tmp_path / "models" / "role_classifier.joblib"
    c.figures_dir = tmp_path / "figures"
    c.reports_dir = tmp_path / "reports"
    # Point data paths at nonexistent files to force demo data
    c.clean_data_path = tmp_path / "nonexistent_clean.csv"
    c.fallback_data_path = tmp_path / "nonexistent_raw.csv"
    c.cv_folds = 2  # fast for tests
    return c


@pytest.fixture
def small_cfg(tmp_path):
    """Config with small demo dataset for fast tests."""
    c = Config()
    c.model_path = tmp_path / "models" / "role_classifier.joblib"
    c.figures_dir = tmp_path / "figures"
    c.reports_dir = tmp_path / "reports"
    c.clean_data_path = tmp_path / "nonexistent.csv"
    c.fallback_data_path = tmp_path / "nonexistent.csv"
    c.cv_folds = 2
    c.tfidf_max_features = 500  # tiny vocabulary for speed
    return c


@pytest.fixture
def demo_df(cfg):
    return _generate_demo_data(cfg)


@pytest.fixture
def trained_pipeline(small_cfg):
    """Train a small pipeline for inference tests."""
    df = _generate_demo_data(small_cfg)
    df["feature_text"] = df.apply(build_feature_text, axis=1)
    from sklearn.model_selection import train_test_split  # noqa: E402

    X_train, _, y_train, _ = train_test_split(
        df["feature_text"],
        df["role_label"],
        test_size=0.2,
        random_state=42,
        stratify=df["role_label"],
    )
    pipeline = build_pipeline(
        LogisticRegression(max_iter=200, random_state=42, class_weight="balanced"),
        small_cfg,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Feature text
# ---------------------------------------------------------------------------


class TestBuildFeatureText:
    def test_title_is_NOT_included(self):
        row = pd.Series(
            {
                "title": "DISTINCTIVE_TITLE_XYZ",
                "skills_normalised": "Python|PyTorch",
                "description": "build models",
            }
        )
        text = build_feature_text(row)
        assert "DISTINCTIVE_TITLE_XYZ" not in text, (
            "Title leaked into feature_text — see ch09's 'Common mistakes' "
            "section on the labels/features leak."
        )

    def test_contains_skills(self):
        row = pd.Series(
            {
                "title": "X",
                "skills_normalised": "PyTorch|RAG",
                "description": "",
            }
        )
        text = build_feature_text(row)
        assert "PyTorch" in text
        assert "RAG" in text

    def test_skills_pipe_separators_replaced(self):
        row = pd.Series(
            {
                "title": "X",
                "skills_normalised": "alpha|beta|gamma",
                "description": "",
            }
        )
        text = build_feature_text(row)
        assert "|" not in text
        for skill in ("alpha", "beta", "gamma"):
            assert skill in text

    def test_skills_repeated_for_weight(self):
        row = pd.Series(
            {
                "title": "X",
                "skills_normalised": "unique_skill_z",
                "description": "",
            }
        )
        text = build_feature_text(row)
        assert text.count("unique_skill_z") >= 2

    def test_description_included(self):
        row = pd.Series(
            {
                "title": "X",
                "skills_normalised": "",
                "description": "transformer models",
            }
        )
        text = build_feature_text(row)
        assert "transformer models" in text

    def test_long_description_truncated(self):
        row = pd.Series(
            {
                "title": "X",
                "skills_normalised": "",
                "description": "word " * 1000,
            }
        )
        text = build_feature_text(row)
        assert len(text) < 5000

    def test_handles_none_fields(self):
        row = pd.Series(
            {
                "title": None,
                "skills_normalised": None,
                "description": None,
            }
        )
        text = build_feature_text(row)
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Demo data generation
# ---------------------------------------------------------------------------


class TestGenerateDemoData:
    def test_returns_dataframe(self, cfg):
        df = _generate_demo_data(cfg)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, cfg):
        df = _generate_demo_data(cfg)
        for col in ["title", "description", "skills_normalised", "role_category"]:
            assert col in df.columns

    def test_all_five_roles_present(self, cfg):
        df = _generate_demo_data(cfg)
        assert set(df["role_label"].unique()) == set(cfg.role_labels)

    def test_roughly_balanced_classes(self, cfg):
        df = _generate_demo_data(cfg)
        counts = df["role_label"].value_counts()
        ratio = counts.max() / counts.min()
        assert ratio < 2.0, f"Class imbalance too high: {ratio:.2f}x"

    def test_descriptions_non_empty(self, cfg):
        df = _generate_demo_data(cfg)
        assert (df["description"].str.len() > 10).all()


# ---------------------------------------------------------------------------
# Pipeline building
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    def test_returns_sklearn_pipeline(self, small_cfg):
        model = LogisticRegression(max_iter=100)
        pipeline = build_pipeline(model, small_cfg)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_tfidf_and_clf(self, small_cfg):
        model = LogisticRegression(max_iter=100)
        pipeline = build_pipeline(model, small_cfg)
        assert "tfidf" in pipeline.named_steps
        assert "clf" in pipeline.named_steps

    def test_pipeline_fits_and_predicts(self, small_cfg):
        model = LogisticRegression(max_iter=200, class_weight="balanced")
        pipeline = build_pipeline(model, small_cfg)
        df = _generate_demo_data(small_cfg)
        df["feature_text"] = df.apply(build_feature_text, axis=1)
        pipeline.fit(df["feature_text"], df["role_label"])
        preds = pipeline.predict(df["feature_text"][:5])
        assert len(preds) == 5
        assert all(p in small_cfg.role_labels for p in preds)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class TestPredictRole:
    def test_returns_tuple(self, trained_pipeline):
        job = {
            "title": "ML Engineer",
            "description": "Machine learning model training",
            "skills_raw": "Python,scikit-learn",
        }
        result = predict_role(job, trained_pipeline)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_valid_label(self, trained_pipeline, small_cfg):
        job = {
            "title": "Data Analyst",
            "description": "SQL dashboards business intelligence",
            "skills_raw": "SQL,Tableau",
        }
        role, _ = predict_role(job, trained_pipeline)
        assert role in small_cfg.role_labels

    def test_confidence_in_range(self, trained_pipeline):
        job = {
            "title": "AI Engineer",
            "description": "LLM RAG pgvector production",
            "skills_raw": "Python,LLMs,RAG",
        }
        _, conf = predict_role(job, trained_pipeline)
        assert 0.0 <= conf <= 1.0

    def test_ai_engineer_signals_predict_correctly(self, trained_pipeline):
        job = {
            "title": "AI Engineer",
            "description": "Build LLM applications with RAG and vector databases. LangChain FastAPI.",
            "skills_raw": "Python,LLMs,RAG,FastAPI,pgvector,LangChain",
        }
        role, conf = predict_role(job, trained_pipeline)
        # Should predict AI Engineer or ML Engineer (both are correct-adjacent)
        assert role in ["AI Engineer", "ML Engineer"]

    def test_data_analyst_signals_predict_correctly(self, trained_pipeline):
        job = {
            "title": "Data Analyst",
            "description": "SQL dashboards, Tableau reporting, business intelligence, KPI tracking.",
            "skills_raw": "SQL,Excel,Tableau,Power BI,business intelligence",
        }
        role, _ = predict_role(job, trained_pipeline)
        assert role in ["Data Analyst", "Data Scientist"]  # both reasonable

    def test_handles_empty_job(self, trained_pipeline, small_cfg):
        role, conf = predict_role({}, trained_pipeline)
        assert role in small_cfg.role_labels
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Full data loading
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_loads_demo_when_no_csv(self, cfg):
        df = load_data(cfg)
        assert len(df) > 0

    def test_adds_role_label_column(self, cfg):
        df = load_data(cfg)
        assert "role_label" in df.columns

    def test_adds_feature_text_column(self, cfg):
        df = load_data(cfg)
        assert "feature_text" in df.columns

    def test_all_labels_valid(self, cfg):
        df = load_data(cfg)
        invalid = df[~df["role_label"].isin(cfg.role_labels)]
        assert len(invalid) == 0, f"Invalid labels found: {invalid['role_label'].unique()}"

    def test_feature_text_non_empty(self, cfg):
        df = load_data(cfg)
        assert (df["feature_text"].str.len() > 0).all()

    def test_loads_from_csv_when_available(self, tmp_path, cfg):
        # Create a minimal CSV
        df_sample = _generate_demo_data(cfg).head(100)
        csv_path = tmp_path / "jobs_clean.csv"
        df_sample.to_csv(csv_path, index=False)
        cfg.clean_data_path = csv_path
        df = load_data(cfg)
        assert len(df) == 100


class TestSelectModel:
    """One-standard-error rule: prefer the simpler model within noise of the best."""

    def test_simpler_model_wins_within_one_std(self):
        from book.ch09.ch09_supervised_learning import select_model

        results = {
            "Logistic Regression": {"mean_f1": 0.867, "std_f1": 0.05},
            "Random Forest": {"mean_f1": 0.869, "std_f1": 0.058},
            "SGD Classifier": {"mean_f1": 0.70, "std_f1": 0.06},
        }
        assert select_model(results) == "Logistic Regression"

    def test_clear_winner_is_kept(self):
        from book.ch09.ch09_supervised_learning import select_model

        results = {
            "Logistic Regression": {"mean_f1": 0.70, "std_f1": 0.01},
            "Random Forest": {"mean_f1": 0.90, "std_f1": 0.01},
        }
        assert select_model(results) == "Random Forest"
