#!/usr/bin/env python3
"""
Chapter 11: Unsupervised Learning
Data Voyage: Discovering Patterns in Unlabeled Data

This script covers essential unsupervised learning concepts and algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.datasets import make_blobs, make_moons, make_circles
import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("CHAPTER 11: UNSUPERVISED LEARNING")
    print("=" * 80)
    print()

    # Section 11.1: Unsupervised Learning Overview
    print("11.1 UNSUPERVISED LEARNING OVERVIEW")
    print("-" * 50)
    demonstrate_unsupervised_overview()

    # Section 11.2: Clustering Algorithms
    print("\n11.2 CLUSTERING ALGORITHMS")
    print("-" * 40)
    demonstrate_clustering()

    # Section 11.3: Dimensionality Reduction
    print("\n11.3 DIMENSIONALITY REDUCTION")
    print("-" * 40)
    demonstrate_dimensionality_reduction()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Unsupervised learning overview and concepts")
    print("✅ Clustering algorithms and evaluation")
    print("✅ Dimensionality reduction techniques")
    print()
    print("Next: Chapter 12 - Deep Learning Fundamentals")
    print("=" * 80)


def demonstrate_unsupervised_overview():
    """Demonstrate unsupervised learning overview and concepts."""
    print("Unsupervised Learning Overview:")
    print("-" * 40)

    print("Unsupervised learning discovers hidden patterns and structures")
    print("in data without predefined labels or target variables.")
    print()

    # 1. Types of Unsupervised Learning
    print("1. TYPES OF UNSUPERVISED LEARNING:")
    print("-" * 40)

    ul_types = {
        "Clustering": {
            "description": "Grouping similar data points together",
            "algorithms": ["K-means", "Hierarchical", "DBSCAN", "Gaussian Mixture"],
            "applications": [
                "Customer segmentation",
                "Image compression",
                "Document clustering",
            ],
        },
        "Dimensionality Reduction": {
            "description": "Reducing number of features while preserving information",
            "algorithms": ["PCA", "t-SNE", "UMAP", "Autoencoders"],
            "applications": [
                "Data visualization",
                "Feature compression",
                "Noise reduction",
            ],
        },
        "Association Rule Mining": {
            "description": "Finding relationships between variables",
            "algorithms": ["Apriori", "FP-Growth", "Eclat"],
            "applications": [
                "Market basket analysis",
                "Recommendation systems",
                "Cross-selling",
            ],
        },
        "Anomaly Detection": {
            "description": "Identifying unusual patterns or outliers",
            "algorithms": ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"],
            "applications": ["Fraud detection", "Quality control", "Network security"],
        },
    }

    for ul_type, details in ul_types.items():
        print(f"{ul_type}:")
        print(f"  Description: {details['description']}")
        print(f"  Algorithms: {', '.join(details['algorithms'])}")
        print(f"  Applications: {', '.join(details['applications'])}")
        print()

    # 2. Create synthetic datasets for demonstrations
    print("2. CREATING SYNTHETIC DATASETS:")
    print("-" * 35)

    np.random.seed(42)
    n_samples = 1000

    # Dataset 1: Well-separated clusters
    X_blobs, y_blobs = make_blobs(
        n_samples=n_samples, centers=4, cluster_std=1.0, random_state=42
    )

    # Dataset 2: Moon-shaped clusters
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

    # Dataset 3: Circular clusters
    X_circles, y_circles = make_circles(
        n_samples=n_samples, noise=0.1, factor=0.5, random_state=42
    )

    print("✅ Created 3 synthetic datasets:")
    print(f"  Blobs dataset: {X_blobs.shape} (4 well-separated clusters)")
    print(f"  Moons dataset: {X_moons.shape} (2 crescent-shaped clusters)")
    print(f"  Circles dataset: {X_circles.shape} (2 concentric circles)")
    print()

    # Store datasets for later use
    global synthetic_datasets
    synthetic_datasets = {
        "blobs": (X_blobs, y_blobs),
        "moons": (X_moons, y_moons),
        "circles": (X_circles, y_circles),
    }

    # 3. Data Visualization
    print("3. DATA VISUALIZATION:")
    print("-" * 25)

    plt.figure(figsize=(15, 5))

    # Blobs dataset
    plt.subplot(1, 3, 1)
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap="viridis", alpha=0.7)
    plt.title("Blobs Dataset (4 clusters)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Moons dataset
    plt.subplot(1, 3, 2)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="plasma", alpha=0.7)
    plt.title("Moons Dataset (2 clusters)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Circles dataset
    plt.subplot(1, 3, 3)
    plt.scatter(
        X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap="cividis", alpha=0.7
    )
    plt.title("Circles Dataset (2 clusters)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig("unsupervised_datasets.png", dpi=300, bbox_inches="tight")
    print("✅ Dataset visualizations saved as 'unsupervised_datasets.png'")
    plt.close()


def demonstrate_clustering():
    """Demonstrate clustering algorithms and evaluation."""
    print("Clustering Algorithms:")
    print("-" * 40)

    if "synthetic_datasets" not in globals():
        print("Synthetic datasets not available. Please run overview first.")
        return

    # 1. K-Means Clustering
    print("1. K-MEANS CLUSTERING:")
    print("-" * 25)

    X_blobs, y_blobs = synthetic_datasets["blobs"]

    # Apply K-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_blobs)

    # Evaluate clustering
    silhouette_avg = silhouette_score(X_blobs, kmeans_labels)
    calinski_avg = calinski_harabasz_score(X_blobs, kmeans_labels)

    print(f"✅ K-means clustering applied to blobs dataset")
    print(f"  Number of clusters: 4")
    print(f"  Silhouette score: {silhouette_avg:.3f}")
    print(f"  Calinski-Harabasz score: {calinski_avg:.1f}")
    print(f"  Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    print()

    # 2. Hierarchical Clustering
    print("2. HIERARCHICAL CLUSTERING:")
    print("-" * 30)

    X_moons, y_moons = synthetic_datasets["moons"]

    # Apply hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=2, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X_moons)

    # Evaluate clustering
    silhouette_avg = silhouette_score(X_moons, hierarchical_labels)
    calinski_avg = calinski_harabasz_score(X_moons, hierarchical_labels)

    print(f"✅ Hierarchical clustering applied to moons dataset")
    print(f"  Number of clusters: 2")
    print(f"  Linkage method: ward")
    print(f"  Silhouette score: {silhouette_avg:.3f}")
    print(f"  Calinski-Harabasz score: {calinski_avg:.1f}")
    print()

    # 3. DBSCAN Clustering
    print("3. DBSCAN CLUSTERING:")
    print("-" * 25)

    X_circles, y_circles = synthetic_datasets["circles"]

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_circles)

    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print(f"✅ DBSCAN clustering applied to circles dataset")
    print(f"  Epsilon: 0.3")
    print(f"  Min samples: 5")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    print()

    # 4. Clustering Visualization
    print("4. CLUSTERING VISUALIZATION:")
    print("-" * 30)

    plt.figure(figsize=(15, 5))

    # K-means results
    plt.subplot(1, 3, 1)
    plt.scatter(
        X_blobs[:, 0], X_blobs[:, 1], c=kmeans_labels, cmap="viridis", alpha=0.7
    )
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="red",
        marker="x",
        s=200,
        linewidths=3,
        label="Centroids",
    )
    plt.title("K-means Clustering (Blobs Dataset)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # Hierarchical clustering results
    plt.subplot(1, 3, 2)
    plt.scatter(
        X_moons[:, 0], X_moons[:, 1], c=hierarchical_labels, cmap="plasma", alpha=0.7
    )
    plt.title("Hierarchical Clustering (Moons Dataset)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # DBSCAN results
    plt.subplot(1, 3, 3)
    plt.scatter(
        X_circles[:, 0], X_circles[:, 1], c=dbscan_labels, cmap="cividis", alpha=0.7
    )
    plt.title("DBSCAN Clustering (Circles Dataset)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig("clustering_results.png", dpi=300, bbox_inches="tight")
    print("✅ Clustering visualizations saved as 'clustering_results.png'")
    plt.close()


def demonstrate_dimensionality_reduction():
    """Demonstrate dimensionality reduction techniques."""
    print("Dimensionality Reduction:")
    print("-" * 40)

    # Create high-dimensional dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X_high_dim = np.random.randn(n_samples, n_features)
    X_high_dim[:, :5] += np.random.randn(n_samples, 5) * 0.5  # Add some structure

    print(f"Working with high-dimensional dataset: {X_high_dim.shape}")
    print()

    # 1. Principal Component Analysis (PCA)
    print("1. PRINCIPAL COMPONENT ANALYSIS (PCA):")
    print("-" * 45)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_high_dim)

    print(f"✅ PCA applied with 95% variance threshold")
    print(f"Original dimensions: {X_high_dim.shape[1]}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    print()

    # Show variance explained by each component
    print("Variance explained by top 10 components:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"  Component {i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    print()

    # 2. t-SNE for Visualization
    print("2. T-SNE FOR VISUALIZATION:")
    print("-" * 30)

    # Apply t-SNE to first 500 samples for computational efficiency
    n_samples_tsne = min(500, X_high_dim.shape[0])
    X_sample = X_high_dim[:n_samples_tsne]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sample)

    print(f"✅ t-SNE applied to {n_samples_tsne} samples")
    print(f"Perplexity: 30")
    print(f"Output dimensions: 2")
    print()

    # 3. Visualization of Results
    print("3. VISUALIZATION OF RESULTS:")
    print("-" * 30)

    plt.figure(figsize=(15, 5))

    # PCA variance explained
    plt.subplot(1, 3, 1)
    n_components = min(20, len(pca.explained_variance_ratio_))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components])
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.xticks(range(1, n_components + 1, 2))

    # First two PCA components
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, color="green")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("First Two PCA Components")

    # t-SNE visualization
    plt.subplot(1, 3, 3)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, color="purple")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization")

    plt.tight_layout()
    plt.savefig("dimensionality_reduction.png", dpi=300, bbox_inches="tight")
    print(
        "✅ Dimensionality reduction visualizations saved as 'dimensionality_reduction.png'"
    )
    plt.close()

    print("Unsupervised Learning Summary:")
    print("✅ Applied clustering algorithms to synthetic datasets")
    print("✅ Implemented dimensionality reduction with PCA and t-SNE")
    print("✅ Evaluated clustering quality using multiple metrics")
    print("✅ Visualized patterns and relationships in unlabeled data")


if __name__ == "__main__":
    main()
