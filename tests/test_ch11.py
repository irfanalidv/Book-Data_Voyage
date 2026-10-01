"""Tests for Chapter 11: Beyond Logistic Regression (Part A - clustering).

Reference: book/ch11/README.md
Source: book/ch11/ch11_unsupervised_learning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch11"))

from ch11_unsupervised_learning import (  # noqa: E402
    Config,
    build_text_matrix,
    cluster_jobs,
    describe_clusters,
    find_optimal_k,
)

from talentlens.paths import jobs_clean_path  # noqa: E402


@pytest.fixture
def jobs_cluster_df() -> pd.DataFrame:
    path = jobs_clean_path()
    if not path.exists():
        pytest.skip(f"jobs_clean not found at {path}")
    return pd.read_csv(path, nrows=200)


@pytest.fixture
def cfg() -> Config:
    return Config(k_min=3, k_max=5, final_k=3)


class TestClusteringPipeline:
    def test_kmeans_at_three_clusters_labels_shape(self, jobs_cluster_df, cfg):
        X, _ = build_text_matrix(jobs_cluster_df, cfg)
        labels, _ = cluster_jobs(X, cfg)
        assert len(labels) == X.shape[0]
        assert set(labels) <= {0, 1, 2}

    def test_silhouette_score_in_valid_range(self, jobs_cluster_df, cfg):
        X, _ = build_text_matrix(jobs_cluster_df, cfg)
        km = KMeans(n_clusters=3, random_state=cfg.random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(500, X.shape[0]), random_state=42)
        assert -1.0 <= sil <= 1.0

    def test_find_optimal_k_returns_parallel_lists(self, jobs_cluster_df, cfg):
        k_vals, inertias, sils = find_optimal_k(build_text_matrix(jobs_cluster_df, cfg)[0], cfg)
        assert k_vals == [3, 4, 5]
        assert len(inertias) == len(sils) == 3
        assert all(-1.0 <= s <= 1.0 for s in sils)

    def test_describe_clusters_dict_keys(self, jobs_cluster_df, cfg):
        X, _ = build_text_matrix(jobs_cluster_df, cfg)
        labels, _ = cluster_jobs(X, cfg)
        descriptions = describe_clusters(jobs_cluster_df, labels)
        assert len(descriptions) == cfg.final_k
        for desc in descriptions:
            assert set(desc.keys()) == {
                "cluster_id",
                "size",
                "top_roles",
                "median_salary_l",
                "remote_pct",
            }
            assert desc["size"] > 0


@pytest.mark.skip(
    reason=(
        "Part B uses MNIST/Housing/Diabetes in book/ch12/ch12_deep_learning_fundamentals.py, "
        "not TalentLens; no Part B unit tests in this file."
    )
)
def test_part_b_neural_networks_deferred():
    """Placeholder documenting explicit skip of Part B in ch11 tests."""
