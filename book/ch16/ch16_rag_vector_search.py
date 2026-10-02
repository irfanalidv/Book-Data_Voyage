"""
Chapter 16: RAG and Vector Databases
Data Voyage - Building TalentLens

TalentLens milestone: add semantic job search - a user describes their
background in plain English and gets back ranked job matches by meaning,
not just keyword overlap.

Run:
    python book/ch16/ch16_rag_vector_search.py

Outputs:
    book/ch16/data/embeddings/jobs_embeddings.npy  (cached)
    book/ch16/reports/figures/ch16_similarity_distribution.png
    book/ch16/reports/figures/ch16_search_quality.png
    book/ch16/reports/figures/ch16_vector_space_2d.png
    book/ch16/reports/search_demo_results.md
"""

from __future__ import annotations  # noqa: E402

import hashlib  # noqa: E402
import logging  # noqa: E402
import sqlite3  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)
SAVE_DPI = 300


def _default_jobs_clean_path() -> Path:
    from talentlens.paths import jobs_clean_path

    primary = jobs_clean_path()
    if primary.exists():
        return primary
    return _THIS_DIR / "data" / "clean" / "jobs_clean.csv"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    data_path: Path = field(default_factory=_default_jobs_clean_path)
    embeddings_dir: Path = _THIS_DIR / "data" / "embeddings"
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"
    db_path: Path = _THIS_DIR / "data" / "talentlens_vectors.db"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 200  # tokens per chunk (approximate, we use words)
    chunk_overlap: int = 50  # overlap between chunks
    batch_size: int = 64  # for embedding batches
    top_k: int = 10  # results to return
    similarity_threshold: float = 0.60
    hybrid_alpha: float = 0.7  # weight for vector score in hybrid (0=keyword only, 1=vector only)


# ---------------------------------------------------------------------------
# Embedding model (lightweight wrapper)
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Wraps sentence-transformers with caching and batching.

    Uses all-MiniLM-L6-v2 (384 dims, fast, CPU-friendly).
    Falls back to a random projection model if sentence-transformers
    is not installed, so the chapter runs without extra dependencies.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self._fallback = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: E402

            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using deterministic random projection fallback for demonstration. "
                "Install with: pip install sentence-transformers"
            )
            self._fallback = True

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Embed a list of texts into normalised vectors.

        Args:
            texts: List of strings to embed.
            normalize: If True, L2-normalise each vector (required for cosine similarity).

        Returns:
            NumPy array of shape (len(texts), dim).
        """
        self._load()
        if self._fallback:
            return self._fallback_embed(texts, normalize)

        vecs = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 200,
        )
        return np.array(vecs, dtype=np.float32)

    def _fallback_embed(self, texts: list[str], normalize: bool) -> np.ndarray:
        """Deterministic random projection - same input always gives same output.

        This is NOT a real embedding model. It's a deterministic hash-based
        projection that preserves some local structure for demonstration only.
        Replace with sentence-transformers for real use.
        """
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            np.random.default_rng(seed)  # warm up seed
            # Add word-level components so similar texts have similar vectors
            words = text.lower().split()
            for word in words:
                word_seed = int(hashlib.md5(word.encode()).hexdigest(), 16) % (2**32)
                word_rng = np.random.default_rng(word_seed)
                vecs[i] += word_rng.standard_normal(self.dim).astype(np.float32)
            if len(words) > 0:
                vecs[i] /= len(words)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms
        return vecs


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words to overlap between chunks.

    Returns:
        List of text chunks. Returns [text] if text is short enough.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
        if end == len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# Vector store (SQLite-backed for local development)
# ---------------------------------------------------------------------------


class VectorStore:
    """SQLite-backed vector store with cosine similarity search.

    Uses exact nearest-neighbour search via NumPy matrix operations.
    For production with >100k documents, swap to pgvector or Qdrant.

    The interface matches what a pgvector-backed store would expose,
    so swapping backends requires no changes to calling code.
    """

    def __init__(self, db_path: Path, dim: int = 384) -> None:
        self.db_path = db_path
        self.dim = dim
        self._conn: Optional[sqlite3.Connection] = None
        self._embedding_matrix: Optional[np.ndarray] = None
        self._id_map: Optional[list[int]] = None

    def connect(self) -> None:
        """Open the SQLite connection and create tables if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id      TEXT UNIQUE,
                title       TEXT,
                company     TEXT,
                salary_min  REAL,
                salary_max  REAL,
                is_remote   INTEGER,
                skills      TEXT,
                chunk_index INTEGER DEFAULT 0,
                chunk_text  TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                job_row_id  INTEGER PRIMARY KEY REFERENCES jobs(id),
                vector_blob BLOB NOT NULL
            )
        """)
        self._conn.commit()

    def upsert(
        self, job_id: str, chunk_index: int, chunk_text: str, embedding: np.ndarray, metadata: dict
    ) -> None:
        """Insert or update a job chunk and its embedding.

        Args:
            job_id: Unique identifier for the job posting.
            chunk_index: Index of this chunk within the job's text.
            chunk_text: The text of this chunk.
            embedding: Normalised embedding vector of shape (dim,).
            metadata: Dict with title, company, salary_min, salary_max, is_remote, skills.
        """
        cursor = self._conn.execute(
            """INSERT OR REPLACE INTO jobs
               (job_id, title, company, salary_min, salary_max, is_remote, skills, chunk_index, chunk_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"{job_id}_{chunk_index}",
                metadata.get("title", ""),
                metadata.get("company", ""),
                metadata.get("salary_min"),
                metadata.get("salary_max"),
                int(metadata.get("is_remote", False)),
                metadata.get("skills", ""),
                chunk_index,
                chunk_text,
            ),
        )
        row_id = cursor.lastrowid
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (job_row_id, vector_blob) VALUES (?, ?)",
            (row_id, embedding.astype(np.float32).tobytes()),
        )
        self._conn.commit()

    def build_index(self) -> None:
        """Load all embeddings into memory for fast similarity search.

        Call this after batch inserting documents, before searching.
        Loads embedding matrix into RAM - O(n * dim * 4 bytes).
        For 50k documents at 384 dims: ~75MB. Acceptable.
        """
        rows = self._conn.execute(
            "SELECT j.id, e.vector_blob FROM jobs j JOIN embeddings e ON j.id = e.job_row_id"
        ).fetchall()

        if not rows:
            logger.warning("No embeddings in store — upsert documents before building index.")
            return

        self._id_map = [r[0] for r in rows]
        vecs = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        self._embedding_matrix = np.stack(vecs)
        logger.info(
            f"Index built: {len(self._id_map):,} vectors, shape {self._embedding_matrix.shape}"
        )

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10, threshold: float = 0.0
    ) -> list[dict]:
        """Find top-k most similar documents to query_embedding.

        Args:
            query_embedding: Normalised query vector of shape (dim,).
            top_k: Number of results to return.
            threshold: Minimum similarity score to include in results.

        Returns:
            List of dicts with keys: job_id, title, company, salary_min,
            salary_max, is_remote, skills, similarity, chunk_text.
        """
        if self._embedding_matrix is None:
            raise RuntimeError("Call build_index() before search().")

        # Cosine similarity = dot product (vectors are normalised)
        similarities = self._embedding_matrix @ query_embedding

        # Get top_k indices, filtering by threshold
        top_indices = np.argsort(similarities)[::-1][: top_k * 3]  # over-fetch, deduplicate by job
        top_indices = [i for i in top_indices if similarities[i] >= threshold]

        results = []
        seen_base_ids: set[str] = set()
        for idx in top_indices:
            if len(results) >= top_k:
                break
            row_id = self._id_map[idx]
            row = self._conn.execute(
                "SELECT job_id, title, company, salary_min, salary_max, is_remote, skills, chunk_text "
                "FROM jobs WHERE id = ?",
                (row_id,),
            ).fetchone()
            if row is None:
                continue
            # Deduplicate by base job_id (strip chunk suffix)
            base_id = row[0].rsplit("_", 1)[0]
            if base_id in seen_base_ids:
                continue
            seen_base_ids.add(base_id)
            results.append(
                {
                    "job_id": base_id,
                    "title": row[1],
                    "company": row[2],
                    "salary_min": row[3],
                    "salary_max": row[4],
                    "is_remote": bool(row[5]),
                    "skills": row[6],
                    "similarity": float(similarities[idx]),
                    "chunk_text": row[7],
                }
            )

        return results

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


def keyword_score(query: str, text: str) -> float:
    """Compute simple TF-based keyword overlap score.

    Args:
        query: Search query string.
        text: Document text to score against.

    Returns:
        Score between 0 and 1 based on query term overlap.
    """
    query_terms = set(query.lower().split())
    doc_terms = text.lower().split()
    if not query_terms or not doc_terms:
        return 0.0
    matches = sum(1 for term in query_terms if term in set(doc_terms))
    return matches / len(query_terms)


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    store: VectorStore,
    alpha: float = 0.7,
    top_k: int = 10,
) -> list[dict]:
    """Combine vector similarity and keyword matching via score fusion.

    Args:
        query: Plain text search query.
        query_embedding: Pre-computed normalised query embedding.
        store: VectorStore to search.
        alpha: Weight for vector score (1-alpha = weight for keyword score).
        top_k: Number of results to return.

    Returns:
        Ranked list of job dicts with hybrid_score added.
    """
    # Get more candidates than we need for fusion
    vector_results = store.search(query_embedding, top_k=top_k * 3, threshold=0.0)

    for result in vector_results:
        kw = keyword_score(query, f"{result['title']} {result['skills']} {result['chunk_text']}")
        result["keyword_score"] = kw
        result["hybrid_score"] = alpha * result["similarity"] + (1 - alpha) * kw

    hybrid_results = sorted(vector_results, key=lambda r: r["hybrid_score"], reverse=True)
    return hybrid_results[:top_k]


# ---------------------------------------------------------------------------
# Data loading and indexing
# ---------------------------------------------------------------------------


def load_or_generate_jobs(cfg: Config) -> pd.DataFrame:
    """Load TalentLens job data or generate demo data.

    Args:
        cfg: Config with data_path.

    Returns:
        DataFrame of job postings.
    """
    if cfg.data_path.exists():
        logger.info(f"Loading real data from {cfg.data_path}")
        return pd.read_csv(cfg.data_path)

    logger.warning("Real data not found — generating demo data for Chapter 16.")
    return _generate_demo_jobs()


def _generate_demo_jobs() -> pd.DataFrame:
    """Generate realistic job postings for demo.

    DEMO DATA - replace with real TalentLens pipeline output.

    Returns:
        DataFrame with 500 synthetic job postings.
    """
    rng = np.random.default_rng(17)
    n = 500

    job_templates = [
        (
            "Senior NLP Engineer",
            "We are looking for an NLP engineer with experience in transformer models, "
            "RAG systems, and production ML. You will build and deploy language models for our AI products. "
            "Strong Python, PyTorch, and FastAPI skills required. Remote-friendly.",
            True,
            2_800_000,
            3_500_000,
            "Python|PyTorch|NLP|RAG|FastAPI|Transformers",
        ),
        (
            "ML Engineer",
            "Join our ML team to build recommendation and ranking systems. "
            "You will work with large-scale data using Spark and deploy models via MLflow. "
            "Experience with scikit-learn, XGBoost, and cloud infrastructure required.",
            False,
            2_200_000,
            3_000_000,
            "Python|scikit-learn|Spark|MLflow|XGBoost|Cloud",
        ),
        (
            "AI Engineer - LLMs",
            "Build production LLM applications including chatbots and document Q&A. "
            "Experience with OpenAI API, LangChain, vector databases, and FastAPI. "
            "Remote position with global team. Strong Python required.",
            True,
            3_000_000,
            4_200_000,
            "Python|LLMs|FastAPI|pgvector|LangChain|RAG",
        ),
        (
            "Data Scientist",
            "Analyse user behaviour data to drive product decisions. "
            "Build predictive models using Python and SQL. Present findings to stakeholders. "
            "Strong statistics and data visualisation skills required.",
            False,
            1_600_000,
            2_400_000,
            "Python|SQL|pandas|scikit-learn|Statistics|Tableau",
        ),
        (
            "MLOps Engineer",
            "Own the machine learning infrastructure. Build CI/CD pipelines for models, "
            "implement monitoring and drift detection, manage Docker and Kubernetes deployments. "
            "Experience with MLflow, Airflow, and cloud platforms essential.",
            True,
            2_500_000,
            3_500_000,
            "Python|Docker|Kubernetes|MLflow|Airflow|CI/CD",
        ),
        (
            "Computer Vision Engineer",
            "Build image recognition and object detection systems. "
            "Experience with PyTorch, OpenCV, and CNN architectures. "
            "Deploy models to edge devices and cloud APIs.",
            False,
            2_000_000,
            3_000_000,
            "Python|PyTorch|OpenCV|CNN|YOLO|TensorFlow",
        ),
        (
            "Data Engineer",
            "Build data pipelines and warehouses. Work with Spark, dbt, and Airflow. "
            "Design schemas, manage ETL processes, ensure data quality. "
            "Strong SQL and cloud data warehouse experience required.",
            False,
            1_800_000,
            2_600_000,
            "Python|SQL|Spark|dbt|Airflow|BigQuery",
        ),
        (
            "Research Scientist - NLP",
            "Conduct research on large language models. Publish papers and "
            "implement state-of-the-art architectures. PhD preferred. Strong Python, PyTorch, "
            "and academic writing skills. Hybrid work arrangement.",
            False,
            3_500_000,
            6_000_000,
            "Python|PyTorch|Research|NLP|Transformers|RLHF",
        ),
    ]

    rows = []
    for i in range(n):
        template = job_templates[rng.integers(0, len(job_templates))]
        title, desc, remote, sal_min, sal_max, skills = template
        # Add slight variation (independent scaling can invert min/max - fix below)
        sal_min = int(sal_min * (0.8 + rng.random() * 0.4))
        sal_max = int(sal_max * (0.8 + rng.random() * 0.4))
        lo, hi = min(sal_min, sal_max), max(sal_min, sal_max)
        sal_min, sal_max = lo, hi if hi > lo else (lo, lo + 50_000)
        rows.append(
            {
                "job_id": f"job_{i:04d}",
                "title": title,
                "description": desc,
                "is_remote": remote,
                "salary_min": sal_min,
                "salary_max": sal_max,
                "skills_normalised": skills,
                "company": f"Company_{rng.integers(1, 50):02d}",
            }
        )
    return pd.DataFrame(rows)


def index_jobs(df: pd.DataFrame, store: VectorStore, model: EmbeddingModel, cfg: Config) -> None:
    """Chunk job descriptions, embed, and store in vector store.

    Args:
        df: Job postings DataFrame.
        store: VectorStore to write into.
        model: EmbeddingModel to use.
        cfg: Config with chunk_size, chunk_overlap, batch_size.
    """
    logger.info(f"Indexing {len(df):,} jobs...")
    all_chunks: list[tuple[str, int, str, dict]] = []  # (job_id, chunk_idx, text, metadata)

    for _, row in df.iterrows():
        job_id = str(row.get("job_id", row.name))
        desc = str(row.get("description", ""))
        chunks = chunk_text(desc, cfg.chunk_size, cfg.chunk_overlap)
        for ci, chunk in enumerate(chunks):
            all_chunks.append(
                (
                    job_id,
                    ci,
                    chunk,
                    {
                        "title": row.get("title", ""),
                        "company": row.get("company", ""),
                        "salary_min": row.get("salary_min"),
                        "salary_max": row.get("salary_max"),
                        "is_remote": bool(row.get("is_remote", False)),
                        "skills": str(row.get("skills_normalised", "")),
                    },
                )
            )

    # Batch embed
    texts = [c[2] for c in all_chunks]
    logger.info(f"Embedding {len(texts):,} chunks (batch_size={cfg.batch_size})...")
    t0 = time.time()
    embeddings = model.embed(texts, normalize=True)
    elapsed = time.time() - t0
    logger.info(f"Embedding complete: {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/sec)")

    # Store
    for i, (job_id, ci, chunk_text_str, meta) in enumerate(all_chunks):
        store.upsert(job_id, ci, chunk_text_str, embeddings[i], meta)

    store.build_index()
    logger.info("Indexing complete.")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_similarity_distribution(results_all: list[dict], cfg: Config) -> Path:
    """Histogram of similarity scores across all search results.

    Args:
        results_all: List of all search result dicts.
        cfg: Config with figures_dir, similarity_threshold.

    Returns:
        Path to saved figure.
    """
    scores = [r["similarity"] for r in results_all]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.hist(scores, bins=30, edgecolor="white", color="#2196F3", alpha=0.8)
    ax.axvline(
        cfg.similarity_threshold,
        color="#F44336",
        linewidth=2,
        linestyle="--",
        label=f"Threshold ({cfg.similarity_threshold})",
    )
    above = sum(1 for s in scores if s >= cfg.similarity_threshold)
    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Number of Results", fontsize=12)
    ax.set_title(
        "Distribution of Similarity Scores — TalentLens Search Demo", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.annotate(
        f"{above} results above threshold ({above/len(scores)*100:.0f}%)\n"
        f"Below threshold: shown as 'related but low match'",
        xy=(cfg.similarity_threshold, ax.get_ylim()[1] * 0.7),
        xytext=(cfg.similarity_threshold + 0.05, ax.get_ylim()[1] * 0.8),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )

    plt.tight_layout()
    out = cfg.figures_dir / "ch16_similarity_distribution.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_search_quality(
    queries: list[str],
    results_per_query: list[list[dict]],
    cfg: Config,
) -> Path:
    """Bar chart showing top similarity score per query.

    Args:
        queries: List of search query strings.
        results_per_query: List of result lists, one per query.
        cfg: Config with figures_dir.

    Returns:
        Path to saved figure.
    """
    top_scores = [results[0]["similarity"] if results else 0.0 for results in results_per_query]
    short_queries = [textwrap.shorten(q, 38, placeholder="…") for q in queries]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = ["#4CAF50" if s >= 0.75 else "#FF9800" if s >= 0.60 else "#F44336" for s in top_scores]
    bars = ax.barh(range(len(queries)), top_scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(short_queries, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Similarity of the top result", fontsize=9)
    ax.set_title("Search quality: top similarity per query", fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 0.85)
    ax.axvline(cfg.similarity_threshold, color="gray", linestyle=":", linewidth=1.5)

    for bar, score in zip(bars, top_scores):
        ax.text(
            score + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=8,
        )

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="#4CAF50", label="Strong match (>0.75)"),
        plt.Rectangle((0, 0), 1, 1, fc="#FF9800", label="Good match (0.60–0.75)"),
        plt.Rectangle((0, 0), 1, 1, fc="#F44336", label="Weak match (<0.60)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.4, -0.16),
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )

    plt.tight_layout()
    out = cfg.figures_dir / "ch16_search_quality.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_vector_space_2d(
    df: pd.DataFrame,
    model: EmbeddingModel,
    cfg: Config,
    n_samples: int = 200,
) -> Path:
    """2D PCA projection of job embeddings, coloured by role category.

    Args:
        df: Job postings DataFrame with title and description.
        model: EmbeddingModel.
        cfg: Config with figures_dir.
        n_samples: How many jobs to embed and plot (keep small for speed).

    Returns:
        Path to saved figure.
    """
    from sklearn.decomposition import PCA  # noqa: E402

    sample = df.sample(min(n_samples, len(df)), random_state=42)
    if "description" in sample.columns:
        texts = (
            sample["title"].astype(str) + " " + sample["description"].fillna("").astype(str)
        ).tolist()
    else:
        texts = sample["title"].astype(str).tolist()
    embeddings = model.embed(texts, normalize=True)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    # Use title as a rough category label
    def categorise(title: str) -> str:
        t = title.lower()
        if "nlp" in t or "language" in t or "text" in t:
            return "NLP"
        if "vision" in t or "image" in t or "cv" in t:
            return "Computer Vision"
        if "mlops" in t or "infra" in t or "platform" in t:
            return "MLOps"
        if "data engineer" in t or "pipeline" in t:
            return "Data Engineering"
        if "research" in t or "scientist" in t:
            return "Research"
        return "ML/AI General"

    categories = [categorise(t) for t in sample["title"].tolist()]
    cat_series = pd.Series(categories)
    unique_cats = cat_series.unique()
    palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)}

    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    for cat in unique_cats:
        mask = cat_series == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color_map[cat],
            label=cat,
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel(
        f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12
    )
    ax.set_ylabel(
        f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12
    )
    ax.set_title(
        "Job embeddings in 2D (PCA): similar roles sit together",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(
        fontsize=7.5,
        title="Role category",
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    fig.text(
        0.01,
        0.01,
        "Each point is a job posting; nearby points have similar meaning.",
        fontsize=7.5,
        color="gray",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out = cfg.figures_dir / "ch16_vector_space_2d.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Search demo
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "5 years Python experience, NLP and transformer models, want remote role",
    "data engineer with Spark and dbt, building ETL pipelines, Bangalore",
    "computer vision engineer, object detection, PyTorch, edge deployment",
    "MLOps background, Kubernetes, model monitoring, CI/CD for ML",
    "research scientist, large language models, RLHF, publications",
    "junior data analyst, SQL and Excel, visualisation, business insights",
]


def run_search_demo(
    store: VectorStore,
    model: EmbeddingModel,
    cfg: Config,
) -> tuple[list[list[dict]], str]:
    """Run demo searches and return results.

    Args:
        store: Populated and indexed VectorStore.
        model: EmbeddingModel.
        cfg: Config with top_k, similarity_threshold.

    Returns:
        Tuple of (results_per_query list, markdown report string).
    """
    all_results: list[list[dict]] = []
    report_lines = ["# TalentLens Semantic Search — Demo Results\n"]

    for query in DEMO_QUERIES:
        query_emb = model.embed([query], normalize=True)[0]
        # Use threshold=0 so we always get ranked hits for charts and the report.
        # (With the hash fallback embedder, scores often sit below 0.60 - a hard
        # threshold would empty every query and the search-quality plot would read 0.)
        results = store.search(query_emb, top_k=cfg.top_k, threshold=0.0)
        all_results.append(results)

        report_lines.append(f"\n## Query: `{query}`\n")
        if not results:
            report_lines.append("No results above similarity threshold.\n")
            continue

        for rank, r in enumerate(results[:5], 1):
            salary_str = (
                f"₹{r['salary_min']/100_000:.0f}L–₹{r['salary_max']/100_000:.0f}L"
                if r["salary_min"] and r["salary_max"]
                else "Not disclosed"
            )
            report_lines.append(
                f"**Rank {rank}:** {r['title']} ({r['company']})  \n"
                f"Similarity: **{r['similarity']:.3f}** | Salary: {salary_str} | "
                f"Remote: {'Yes' if r['is_remote'] else 'No'}  \n"
                f"Skills: `{r['skills']}`\n"
            )

        _print_block(
            f"Query: {query[:60]}",
            [
                f"Rank {i+1}: {r['title']} | sim={r['similarity']:.3f} | "
                f"remote={'Y' if r['is_remote'] else 'N'}"
                for i, r in enumerate(results[:5])
            ],
        )

    return all_results, "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _print_block(title: str, lines: list[str]) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  {title.upper()}\n{sep}")
    for line in lines:
        logger.info(f"  {line}")


def _ensure_dirs(cfg: Config) -> None:
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Chapter 16 RAG pipeline."""
    logger.info("=" * 60)
    logger.info("  CHAPTER 16: RAG AND VECTOR DATABASES")
    logger.info("  TalentLens — Semantic Job Search")
    logger.info("=" * 60)

    cfg = Config()
    _ensure_dirs(cfg)

    # 1. Load data
    logger.info("\n[1/6] Loading job postings...")
    df = load_or_generate_jobs(cfg)
    logger.info(f"Loaded {len(df):,} job postings")

    # 2. Initialise embedding model
    logger.info("\n[2/6] Initialising embedding model...")
    model = EmbeddingModel(cfg.embedding_model, cfg.embedding_dim)

    # 3. Build vector store
    logger.info("\n[3/6] Building vector store...")
    if cfg.db_path.exists():
        cfg.db_path.unlink()
        logger.info(f"Cleared stale vector store: {cfg.db_path}")
    store = VectorStore(cfg.db_path, cfg.embedding_dim)
    store.connect()
    index_jobs(df, store, model, cfg)

    # 4. Run search demo
    logger.info("\n[4/6] Running search demo...")
    all_results, report_md = run_search_demo(store, model, cfg)

    report_path = cfg.reports_dir / "search_demo_results.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info(f"Saved: {report_path}")

    # 5. Visualisations
    logger.info("\n[5/6] Generating visualisations...")
    flat_results = [r for rs in all_results for r in rs]
    if flat_results:
        plot_similarity_distribution(flat_results, cfg)
    plot_search_quality(DEMO_QUERIES, all_results, cfg)
    plot_vector_space_2d(df, model, cfg)

    # 6. Cleanup
    store.close()

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 16 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Search results → {cfg.reports_dir}/search_demo_results.md")
    logger.info(f"  Figures        → {cfg.figures_dir}/")
    logger.info("\nNext: Chapter 17 — Add the LLM generation layer.")
    logger.info("We'll take these retrieved jobs and generate personalised match explanations.")


if __name__ == "__main__":
    main()
