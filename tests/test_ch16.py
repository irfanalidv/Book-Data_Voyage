"""
Tests for Chapter 16: RAG and Vector Databases.

Run from repository root:

    pytest tests/test_ch16.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch16"))

# Check if sentence-transformers is usable (network may be restricted)
try:
    from sentence_transformers import SentenceTransformer as _ST

    _ST("all-MiniLM-L6-v2")  # will fail if HF blocked
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

from ch16_rag_vector_search import (  # noqa: E402
    Config,
    EmbeddingModel,
    VectorStore,
    _generate_demo_jobs,
    chunk_text,
    hybrid_search,
    index_jobs,
    keyword_score,
)


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.db_path = tmp_path / "test.db"
    c.figures_dir = tmp_path / "figures"
    c.reports_dir = tmp_path / "reports"
    c.figures_dir.mkdir(parents=True)
    c.reports_dir.mkdir(parents=True)
    return c


@pytest.fixture
def model():
    return EmbeddingModel(dim=384)


@pytest.fixture
def store(cfg):
    s = VectorStore(cfg.db_path, cfg.embedding_dim)
    s.connect()
    yield s
    s.close()


@pytest.fixture
def populated_store(store, model, cfg):
    """Store with 10 indexed jobs."""
    df = _generate_demo_jobs().head(10)
    index_jobs(df, store, model, cfg)
    return store


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "hello world this is short"
        result = chunk_text(text, chunk_size=100)
        assert result == [text]

    def test_long_text_splits_into_multiple_chunks(self):
        text = " ".join(["word"] * 500)
        result = chunk_text(text, chunk_size=200, overlap=50)
        assert len(result) > 1

    def test_chunks_overlap(self):
        text = " ".join([f"word{i}" for i in range(300)])
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        last_words_chunk0 = set(chunks[0].split()[-50:])
        first_words_chunk1 = set(chunks[1].split()[:50])
        assert len(last_words_chunk0 & first_words_chunk1) > 0

    def test_empty_text_returns_list_with_empty_string(self):
        result = chunk_text("", chunk_size=100)
        assert result == [""]

    def test_chunk_size_respected(self):
        text = " ".join(["word"] * 600)
        chunks = chunk_text(text, chunk_size=200, overlap=0)
        for chunk in chunks:
            assert len(chunk.split()) <= 200


@pytest.mark.skipif(
    not _SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed or HuggingFace not reachable",
)
class TestEmbeddingModel:
    def test_embed_returns_correct_shape(self, model):
        texts = ["hello world", "machine learning", "python fastapi"]
        result = model.embed(texts)
        assert result.shape == (3, 384)

    def test_embed_normalised_vectors(self, model):
        texts = ["test text for normalisation check"]
        result = model.embed(texts, normalize=True)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="Semantic ordering requires sentence-transformers (fallback is hash-based).",
    )
    def test_similar_texts_closer_than_different(self, model):
        texts = [
            "machine learning engineer Python PyTorch",
            "ML engineer deep learning Python",
            "chef de cuisine fine dining Paris",
        ]
        embeddings = model.embed(texts, normalize=True)
        sim_similar = float(embeddings[0] @ embeddings[1])
        sim_different = float(embeddings[0] @ embeddings[2])
        assert sim_similar > sim_different

    def test_deterministic_output(self, model):
        texts = ["same text produces same embedding"]
        e1 = model.embed(texts)
        e2 = model.embed(texts)
        np.testing.assert_array_equal(e1, e2)


@pytest.mark.skipif(
    not _SENTENCE_TRANSFORMERS_AVAILABLE, reason="HuggingFace not reachable in this environment"
)
class TestVectorStore:
    def test_connect_creates_tables(self, store):
        cursor = store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "jobs" in tables
        assert "embeddings" in tables

    def test_upsert_and_search(self, store, model):
        emb = model.embed(["Python NLP engineer remote"], normalize=True)[0]
        store.upsert(
            "job_001",
            0,
            "Python NLP engineer remote",
            emb,
            {
                "title": "NLP Engineer",
                "company": "AI Co",
                "salary_min": 2000000,
                "salary_max": 3000000,
                "is_remote": True,
                "skills": "Python|NLP",
            },
        )
        store.build_index()
        query_emb = model.embed(["Python natural language processing"], normalize=True)[0]
        results = store.search(query_emb, top_k=1, threshold=0.0)
        assert len(results) == 1
        assert results[0]["title"] == "NLP Engineer"

    def test_search_before_build_index_raises(self, store):
        with pytest.raises(RuntimeError):
            store.search(np.zeros(384))

    def test_similarity_within_bounds(self, populated_store, model):
        query = model.embed(["machine learning"], normalize=True)[0]
        results = populated_store.search(query, top_k=5, threshold=0.0)
        for r in results:
            assert -1.0 <= r["similarity"] <= 1.0

    def test_results_sorted_by_similarity(self, populated_store, model):
        query = model.embed(["NLP transformer models"], normalize=True)[0]
        results = populated_store.search(query, top_k=5, threshold=0.0)
        sims = [r["similarity"] for r in results]
        assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))

    def test_deduplication_by_job_id(self, store, model):
        emb = model.embed(["chunk one of the job description"], normalize=True)[0]
        emb2 = model.embed(["chunk two of the job description more detail"], normalize=True)[0]
        store.upsert(
            "job_dup",
            0,
            "chunk one",
            emb,
            {"title": "Test Job", "company": "Co", "is_remote": False},
        )
        store.upsert(
            "job_dup",
            1,
            "chunk two",
            emb2,
            {"title": "Test Job", "company": "Co", "is_remote": False},
        )
        store.build_index()
        query = model.embed(["job description chunk"], normalize=True)[0]
        results = store.search(query, top_k=10, threshold=0.0)
        job_ids = [r["job_id"] for r in results]
        assert job_ids.count("job_dup") <= 1


class TestKeywordScore:
    def test_perfect_match(self):
        score = keyword_score("python nlp", "python nlp engineer")
        assert score == 1.0

    def test_no_match(self):
        score = keyword_score("rust golang", "python machine learning")
        assert score == 0.0

    def test_partial_match(self):
        score = keyword_score("python nlp fastapi", "python engineer")
        assert 0 < score < 1.0

    def test_empty_query(self):
        score = keyword_score("", "some text")
        assert score == 0.0


@pytest.mark.skipif(
    not _SENTENCE_TRANSFORMERS_AVAILABLE, reason="HuggingFace not reachable in this environment"
)
class TestHybridSearch:
    def test_hybrid_score_fusion(self, populated_store, model, cfg):
        query = "Python NLP transformers remote"
        qemb = model.embed([query], normalize=True)[0]
        out = hybrid_search(query, qemb, populated_store, alpha=cfg.hybrid_alpha, top_k=5)
        assert len(out) <= 5
        for row in out:
            assert "hybrid_score" in row
            assert "keyword_score" in row


class TestDemoJobs:
    def test_returns_dataframe(self):
        df = _generate_demo_jobs()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = _generate_demo_jobs()
        for col in ["job_id", "title", "description", "is_remote"]:
            assert col in df.columns

    def test_unique_job_ids(self):
        df = _generate_demo_jobs()
        assert df["job_id"].nunique() == len(df)

    def test_reasonable_salaries(self):
        df = _generate_demo_jobs()
        assert (df["salary_min"] > 0).all()
        assert (df["salary_max"] > df["salary_min"]).all()
