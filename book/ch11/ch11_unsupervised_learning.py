"""
Chapter 11: Unsupervised Learning -- Discovering Hidden Job Archetypes
Data Voyage -- Building TalentLens

TalentLens milestone: cluster job postings to find hidden structure
beyond the five labelled categories.

Run: python book/ch11/ch11_unsupervised_learning.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from talentlens.paths import jobs_clean_path

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
    clean_path: Path = field(default_factory=jobs_clean_path)
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"
    k_min: int = 3
    k_max: int = 12
    final_k: int = 7
    tfidf_max_features: int = 5_000
    # Fixed seed - cluster_report.md must be byte-stable across consecutive runs.
    random_state: int = 42


def _cluster_median_salary_l(cdf: pd.DataFrame) -> float:
    """Median annual salary in lakhs for a cluster (bundled CSV uses salary_annual_inr)."""
    if "salary_annual_inr" in cdf.columns:
        annual = cdf["salary_annual_inr"].dropna()
        if len(annual):
            return round(float(annual.median()) / 100_000, 1)
    if "salary_min" in cdf.columns:
        sal_min = cdf["salary_min"].dropna()
        if len(sal_min):
            return round(float(sal_min.median()) / 100_000, 1)
    return 0.0


def build_text_matrix(df: pd.DataFrame, cfg: Config):
    text = (
        df["title"].fillna("")
        + " "
        + df.get("skills_normalised", pd.Series([""] * len(df))).fillna("").str.replace("|", " ")
        + " "
        + df["description"].fillna("").str[:300]
    )
    vec = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
    )
    X = vec.fit_transform(text)
    logger.info(f"TF-IDF: {X.shape[0]:,} docs x {X.shape[1]:,} features")
    return X, vec


def find_optimal_k(X, cfg: Config):
    k_vals, inertias, sils = [], [], []
    for k in range(cfg.k_min, cfg.k_max + 1):
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels, sample_size=min(1000, X.shape[0]), random_state=42)
        sils.append(sil)
        k_vals.append(k)
        logger.info(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.3f}")
    return k_vals, inertias, sils


def cluster_jobs(X, cfg: Config):
    km = KMeans(n_clusters=cfg.final_k, random_state=cfg.random_state, n_init=20)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(1000, X.shape[0]), random_state=42)
    logger.info(f"Final clustering: k={cfg.final_k}, silhouette={sil:.3f}")
    return labels, km


def describe_clusters(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    df = df.copy()
    df["cluster"] = labels
    descriptions = []
    for cid in sorted(df["cluster"].unique()):
        cdf = df[df["cluster"] == cid]
        role_dist = {}
        if "role_label" in cdf.columns:
            role_dist = cdf["role_label"].value_counts().head(3).to_dict()
        elif "title" in cdf.columns:
            role_dist = cdf["title"].value_counts().head(3).to_dict()
        med_sal = _cluster_median_salary_l(cdf)
        remote_pct = cdf.get("is_remote", pd.Series([False] * len(cdf))).mean() * 100
        descriptions.append(
            {
                "cluster_id": cid,
                "size": len(cdf),
                "top_roles": role_dist,
                "median_salary_l": round(med_sal, 1),
                "remote_pct": round(remote_pct, 1),
            }
        )
        logger.info(
            f"  Cluster {cid}: {len(cdf)} jobs | salary ₹{med_sal:.1f}L | "
            f"remote {remote_pct:.0f}% | roles: {list(role_dist.keys())[:2]}"
        )
    return descriptions


def plot_elbow_silhouette(k_vals, inertias, sils, cfg: Config) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k_vals, inertias, "o-", color="#2196F3", linewidth=2, markersize=7)
    axes[0].axvline(cfg.final_k, color="#F44336", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[0].text(
        cfg.final_k + 0.15,
        max(inertias) * 0.95,
        f"k={cfg.final_k} chosen",
        fontsize=9,
        color="#F44336",
    )
    axes[0].set_xlabel("Number of clusters (k)", fontsize=12)
    axes[0].set_ylabel("Inertia", fontsize=11)
    axes[0].set_title("Elbow Method", fontsize=12, fontweight="bold")
    axes[1].plot(k_vals, sils, "s-", color="#4CAF50", linewidth=2, markersize=7)
    axes[1].axvline(cfg.final_k, color="#F44336", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[1].set_xlabel("Number of clusters (k)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score", fontsize=11)
    axes[1].set_title("Silhouette Score", fontsize=12, fontweight="bold")
    plt.suptitle("TalentLens Job Clustering -- Choosing k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = cfg.figures_dir / "ch11_elbow_silhouette.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_cluster_pca(X, labels, descriptions, cfg: Config) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    arr = X.toarray() if hasattr(X, "toarray") else X
    coords = pca.fit_transform(arr)
    sample_size = min(800, len(labels))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(labels), sample_size, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(labels))))
    fig, ax = plt.subplots(figsize=(11, 8))
    for cid in sorted(set(labels)):
        mask = labels[idx] == cid
        desc = next((d for d in descriptions if d["cluster_id"] == cid), {})
        top_role = list(desc.get("top_roles", {}).keys())[0] if desc.get("top_roles") else f"C{cid}"
        ax.scatter(
            coords[idx][mask, 0],
            coords[idx][mask, 1],
            c=[colors[cid]],
            label=f"C{cid}: {top_role}",
            alpha=0.6,
            s=30,
            edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.set_title("Job Clusters -- PCA 2D Projection", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    out = cfg.figures_dir / "ch11_cluster_pca.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_cluster_profiles(descriptions: list[dict], cfg: Config) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    ids = [d["cluster_id"] for d in descriptions]
    sizes = [d["size"] for d in descriptions]
    salaries = [d["median_salary_l"] for d in descriptions]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar([f"C{i}" for i in ids], sizes, color="#2196F3", alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("Job postings", fontsize=11)
    axes[0].set_title("Cluster Size", fontsize=12, fontweight="bold")
    axes[1].bar([f"C{i}" for i in ids], salaries, color="#4CAF50", alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("Median Salary (₹ lakhs)", fontsize=11)
    axes[1].set_title("Median Salary per Cluster", fontsize=12, fontweight="bold")
    for i, (cid, sal) in enumerate(zip(ids, salaries)):
        axes[1].text(i, sal + 0.2, f"₹{sal:.1f}L", ha="center", fontsize=9, fontweight="bold")
    plt.suptitle("TalentLens Cluster Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = cfg.figures_dir / "ch11_cluster_profiles.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def write_cluster_report(descriptions, k_vals, sils, cfg: Config) -> Path:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    best_k = k_vals[sils.index(max(sils))]
    lines = [
        "# TalentLens Job Cluster Report",
        f"\nBest k by silhouette: {best_k} (score: {max(sils):.3f})",
        f"Chosen k: {cfg.final_k}\n",
        "| Cluster | Size | Salary | Remote | Top roles |",
        "|---------|------|--------|--------|-----------|",
    ]
    for d in descriptions:
        top = ", ".join(list(d["top_roles"].keys())[:2])
        lines.append(
            f"| C{d['cluster_id']} | {d['size']:,} | "
            f"₹{d['median_salary_l']}L | {d['remote_pct']:.0f}% | {top} |"
        )
    out = cfg.reports_dir / "cluster_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


def main() -> None:
    cfg = Config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("  CHAPTER 11: UNSUPERVISED LEARNING")
    logger.info("  TalentLens -- Discovering Hidden Job Archetypes")
    logger.info("=" * 60)
    if cfg.clean_path.exists():
        df = pd.read_csv(cfg.clean_path)
    else:
        logger.warning("Clean data not found -- using minimal demo data")
        rng = np.random.default_rng(cfg.random_state)
        n = 300
        roles = ["ML Engineer", "AI Engineer", "Data Scientist", "Data Engineer", "Data Analyst"]
        df = pd.DataFrame(
            {
                "title": [roles[i % len(roles)] for i in range(n)],
                "description": ["production machine learning models python pytorch fastapi"] * n,
                "skills_normalised": ["Python|PyTorch|ML"] * n,
                "salary_min": rng.integers(800_000, 4_000_000, n).astype(float),
                "is_remote": rng.random(n) > 0.6,
                "company": ["Nimbus Fintech"] * n,
            }
        )
    logger.info(f"Loaded {len(df):,} rows")
    X, vec = build_text_matrix(df, cfg)
    k_vals, inertias, sils = find_optimal_k(X, cfg)
    labels, km = cluster_jobs(X, cfg)
    descriptions = describe_clusters(df, labels)
    plot_elbow_silhouette(k_vals, inertias, sils, cfg)
    plot_cluster_pca(X, labels, descriptions, cfg)
    plot_cluster_profiles(descriptions, cfg)
    write_cluster_report(descriptions, k_vals, sils, cfg)
    logger.info("  CHAPTER 11 COMPLETE")
    logger.info(f"  Figures: {cfg.figures_dir}/")
    logger.info("  Next: Chapter 12 -- Deep Learning Fundamentals")


if __name__ == "__main__":
    main()
