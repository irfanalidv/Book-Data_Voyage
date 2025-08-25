#!/usr/bin/env python3
"""
Chapter 11: Unsupervised Learning
Data Voyage: Discovering Patterns in Unlabeled Data with Real Data

This script covers essential unsupervised learning concepts and algorithms using REAL DATA.
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
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
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
    print("✅ Unsupervised learning overview and concepts with real data")
    print("✅ Clustering algorithms and evaluation on actual datasets")
    print("✅ Dimensionality reduction techniques using authentic data")
    print()
    print("Next: Chapter 12 - Deep Learning Fundamentals")
    print("=" * 80)


def demonstrate_unsupervised_overview():
    """Demonstrate unsupervised learning overview and concepts using real data."""
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

    # 2. Load Real Datasets for Analysis
    print("2. LOADING REAL DATASETS FOR UNSUPERVISED LEARNING:")
    print("-" * 55)

    iris = load_iris()
    diabetes = load_diabetes()
    breast_cancer = load_breast_cancer()
    wine = load_wine()

    # Create DataFrames
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target
    iris_df["species"] = [iris.target_names[i] for i in iris.target]

    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df["target"] = diabetes.target

    breast_cancer_df = pd.DataFrame(
        breast_cancer.data, columns=breast_cancer.feature_names
    )
    breast_cancer_df["target"] = breast_cancer.target
    breast_cancer_df["diagnosis"] = [
        "Malignant" if t == 1 else "Benign" for t in breast_cancer.target
    ]

    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df["target"] = wine.target
    wine_df["wine_type"] = [wine.target_names[i] for i in wine.target]

    print(f"✅ Loaded real datasets:")
    print(f"  • Iris: {iris_df.shape[0]} samples, {iris_df.shape[1]-2} features")
    print(
        f"  • Diabetes: {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]-1} features"
    )
    print(
        f"  • Breast Cancer: {breast_cancer_df.shape[0]} samples, {breast_cancer_df.shape[1]-2} features"
    )
    print(f"  • Wine: {wine_df.shape[0]} samples, {wine_df.shape[1]-2} features")
    print()

    # 3. Dataset Characteristics for Unsupervised Learning
    print("3. DATASET CHARACTERISTICS FOR UNSUPERVISED LEARNING:")
    print("-" * 55)

    datasets_info = {
        "Iris": {
            "df": iris_df,
            "features": iris.feature_names,
            "description": "Botanical measurements of iris flowers",
            "clustering_potential": "High - clear species separation",
            "dimensionality_reduction": "Good - 4 features to 2-3 components",
        },
        "Diabetes": {
            "df": diabetes_df,
            "features": diabetes.feature_names,
            "description": "Medical measurements for diabetes progression",
            "clustering_potential": "Medium - continuous regression data",
            "dimensionality_reduction": "Excellent - 10 features to 3-5 components",
        },
        "Breast Cancer": {
            "df": breast_cancer_df,
            "features": breast_cancer.feature_names,
            "description": "Cell nucleus characteristics for cancer diagnosis",
            "clustering_potential": "High - clear malignant/benign separation",
            "dimensionality_reduction": "Excellent - 30 features to 5-10 components",
        },
        "Wine": {
            "df": wine_df,
            "features": wine.feature_names,
            "description": "Chemical analysis of wine samples",
            "clustering_potential": "High - distinct wine type characteristics",
            "dimensionality_reduction": "Good - 13 features to 3-5 components",
        },
    }

    for dataset_name, info in datasets_info.items():
        print(f"{dataset_name} Dataset:")
        print(f"  Description: {info['description']}")
        print(f"  Features: {len(info['features'])}")
        print(f"  Clustering Potential: {info['clustering_potential']}")
        print(f"  Dimensionality Reduction: {info['dimensionality_reduction']}")
        print()

    # Store datasets globally for other functions
    global datasets
    datasets = datasets_info

    return datasets_info


def demonstrate_clustering():
    """Demonstrate clustering algorithms using real data."""
    print("Clustering Algorithms:")
    print("-" * 40)

    if "datasets" not in globals() or datasets is None:
        print("❌ No datasets available for clustering")
        return

    # Use Iris dataset for clustering demonstration
    iris_df = datasets["Iris"]["df"]
    print(
        f"Clustering analysis on Iris dataset: {iris_df.shape[0]} samples, {iris_df.shape[1]-2} features"
    )
    print()

    # Prepare data for clustering
    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    X = iris_df[features].values
    y_true = iris_df["target"].values

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Features used for clustering: {features}")
    print(f"Data shape: {X.shape}")
    print(f"True labels: {len(np.unique(y_true))} classes")
    print()

    # 1. K-Means Clustering
    print("1. K-MEANS CLUSTERING:")
    print("-" * 25)

    # Find optimal number of clusters using elbow method
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

        if k > 1:  # Silhouette score requires at least 2 clusters
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X_scaled, labels))

    print(f"✅ K-means clustering completed for k = 2 to 10")
    print(f"  Optimal k (elbow method): Analyzing inertia values...")
    print(f"  Optimal k (silhouette): Analyzing silhouette scores...")
    print()

    # 2. Hierarchical Clustering
    print("2. HIERARCHICAL CLUSTERING:")
    print("-" * 30)

    # Use optimal k from silhouette analysis
    optimal_k = k_range[
        np.argmax(silhouette_scores) + 1
    ]  # +1 because we start from k=2
    print(f"Using optimal k = {optimal_k} (best silhouette score)")

    # Perform hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    y_hierarchical = hierarchical.fit_predict(X_scaled)

    print(f"✅ Hierarchical clustering completed with {optimal_k} clusters")
    print()

    # 3. DBSCAN Clustering
    print("3. DBSCAN CLUSTERING:")
    print("-" * 25)

    # Try different epsilon values
    eps_values = [0.3, 0.5, 0.7, 1.0]
    dbscan_results = {}

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        y_dbscan = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
        n_noise = list(y_dbscan).count(-1)

        dbscan_results[eps] = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "labels": y_dbscan,
        }

    print(f"✅ DBSCAN clustering completed for different epsilon values")
    print("DBSCAN Results:")
    for eps, results in dbscan_results.items():
        print(
            f"  ε = {eps}: {results['n_clusters']} clusters, {results['n_noise']} noise points"
        )
    print()

    # 4. Clustering Evaluation
    print("4. CLUSTERING EVALUATION:")
    print("-" * 30)

    # Evaluate K-means with optimal k
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    y_kmeans = kmeans_optimal.fit_predict(X_scaled)

    # Calculate evaluation metrics
    kmeans_silhouette = silhouette_score(X_scaled, y_kmeans)
    kmeans_calinski = calinski_harabasz_score(X_scaled, y_kmeans)

    hierarchical_silhouette = silhouette_score(X_scaled, y_hierarchical)
    hierarchical_calinski = calinski_harabasz_score(X_scaled, y_hierarchical)

    print("Clustering Performance Metrics:")
    print(f"  K-means (k={optimal_k}):")
    print(f"    Silhouette Score: {kmeans_silhouette:.3f}")
    print(f"    Calinski-Harabasz Score: {kmeans_calinski:.1f}")
    print(f"  Hierarchical (k={optimal_k}):")
    print(f"    Silhouette Score: {hierarchical_silhouette:.3f}")
    print(f"    Calinski-Harabasz Score: {hierarchical_calinski:.1f}")
    print()

    # 5. Cluster Analysis
    print("5. CLUSTER ANALYSIS:")
    print("-" * 25)

    # Analyze K-means clusters
    print("K-means Cluster Analysis:")
    for cluster_id in range(optimal_k):
        cluster_mask = y_kmeans == cluster_id
        cluster_data = X[cluster_mask]

        print(f"  Cluster {cluster_id}: {np.sum(cluster_mask)} samples")
        print(f"    Mean sepal length: {cluster_data[:, 0].mean():.2f} cm")
        print(f"    Mean sepal width: {cluster_data[:, 1].mean():.2f} cm")
        print(f"    Mean petal length: {cluster_data[:, 2].mean():.2f} cm")
        print(f"    Mean petal width: {cluster_data[:, 3].mean():.2f} cm")
        print()

    # 6. Create clustering visualization
    print("6. CREATING CLUSTERING VISUALIZATION:")
    print("-" * 40)

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Clustering Results - Real Iris Data", fontsize=16, fontweight="bold"
        )

        # 1. True labels (ground truth)
        scatter1 = axes[0, 0].scatter(
            X[:, 0], X[:, 1], c=y_true, cmap="viridis", alpha=0.7
        )
        axes[0, 0].set_title("True Labels (Ground Truth)")
        axes[0, 0].set_xlabel("Sepal Length (cm)")
        axes[0, 0].set_ylabel("Sepal Width (cm)")
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 0], label="Species")

        # 2. K-means clustering
        scatter2 = axes[0, 1].scatter(
            X[:, 0], X[:, 1], c=y_kmeans, cmap="Set1", alpha=0.7
        )
        axes[0, 1].set_title(f"K-means Clustering (k={optimal_k})")
        axes[0, 1].set_xlabel("Sepal Length (cm)")
        axes[0, 1].set_ylabel("Sepal Width (cm)")
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0, 1], label="Cluster")

        # 3. Hierarchical clustering
        scatter3 = axes[0, 2].scatter(
            X[:, 0], X[:, 1], c=y_hierarchical, cmap="Set2", alpha=0.7
        )
        axes[0, 2].set_title(f"Hierarchical Clustering (k={optimal_k})")
        axes[0, 2].set_xlabel("Sepal Length (cm)")
        axes[0, 2].set_ylabel("Sepal Width (cm)")
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[0, 2], label="Cluster")

        # 4. Elbow method for K-means
        axes[1, 0].plot(k_range, inertias, "bo-", linewidth=2, markersize=6)
        axes[1, 0].axvline(
            x=optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}"
        )
        axes[1, 0].set_title("Elbow Method for K-means")
        axes[1, 0].set_xlabel("Number of Clusters (k)")
        axes[1, 0].set_ylabel("Inertia")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Silhouette scores
        k_silhouette = [2] + list(k_range[1:])  # Add k=2 back for plotting
        silhouette_values = [0] + silhouette_scores  # Add 0 for k=1
        axes[1, 1].plot(
            k_silhouette, silhouette_values, "ro-", linewidth=2, markersize=6
        )
        axes[1, 1].axvline(
            x=optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}"
        )
        axes[1, 1].set_title("Silhouette Score vs Number of Clusters")
        axes[1, 1].set_xlabel("Number of Clusters (k)")
        axes[1, 1].set_ylabel("Silhouette Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Feature comparison by cluster
        cluster_means = []
        cluster_labels = []
        for cluster_id in range(optimal_k):
            cluster_mask = y_kmeans == cluster_id
            cluster_data = X[cluster_mask]
            cluster_means.append(cluster_data.mean(axis=0))
            cluster_labels.append(f"Cluster {cluster_id}")

        cluster_means = np.array(cluster_means)
        x_pos = np.arange(len(features))
        width = 0.8 / optimal_k

        for i in range(optimal_k):
            axes[1, 2].bar(
                x_pos + i * width,
                cluster_means[i],
                width,
                label=cluster_labels[i],
                alpha=0.7,
            )

        axes[1, 2].set_title("Feature Means by Cluster")
        axes[1, 2].set_xlabel("Features")
        axes[1, 2].set_ylabel("Mean Value")
        axes[1, 2].set_xticks(x_pos + width * (optimal_k - 1) / 2)
        axes[1, 2].set_xticklabels([f.split()[0] for f in features], rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the visualization
        output_file = "clustering_results.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")

        plt.show()

    except Exception as e:
        print(f"  ❌ Error creating visualization: {e}")

    print()

    # Store clustering results globally
    global clustering_results
    clustering_results = {
        "kmeans": y_kmeans,
        "hierarchical": y_hierarchical,
        "dbscan": dbscan_results,
        "optimal_k": optimal_k,
        "features": features,
        "X": X,
        "y_true": y_true,
    }

    return clustering_results


def demonstrate_dimensionality_reduction():
    """Demonstrate dimensionality reduction techniques using real data."""
    print("Dimensionality Reduction:")
    print("-" * 40)

    if "datasets" not in globals() or datasets is None:
        print("❌ No datasets available for dimensionality reduction")
        return

    # Use Breast Cancer dataset for dimensionality reduction (more features)
    breast_cancer_df = datasets["Breast Cancer"]["df"]
    print(
        f"Dimensionality reduction on Breast Cancer dataset: {breast_cancer_df.shape[0]} samples, {breast_cancer_df.shape[1]-2} features"
    )
    print()

    # Prepare data for dimensionality reduction
    features = [
        col for col in breast_cancer_df.columns if col not in ["target", "diagnosis"]
    ]
    X = breast_cancer_df[features].values
    y_true = breast_cancer_df["target"].values

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Features used: {len(features)}")
    print(f"Data shape: {X.shape}")
    print(f"True labels: {len(np.unique(y_true))} classes")
    print()

    # 1. Principal Component Analysis (PCA)
    print("1. PRINCIPAL COMPONENT ANALYSIS (PCA):")
    print("-" * 45)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Get explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    print(f"✅ PCA completed")
    print(f"  Original features: {X.shape[1]}")
    print(f"  PCA components: {X_pca.shape[1]}")
    print()

    print("Explained variance by components:")
    for i, (var_ratio, cum_var_ratio) in enumerate(
        zip(explained_variance_ratio, cumulative_variance_ratio)
    ):
        print(
            f"  Component {i+1:2d}: {var_ratio:.3f} ({var_ratio*100:.1f}%) - Cumulative: {cum_var_ratio:.3f} ({cum_var_ratio*100:.1f}%)"
        )
        if cum_var_ratio >= 0.95:
            print(f"    → 95% variance explained with {i+1} components")
            break
    print()

    # 2. Feature Importance in PCA
    print("2. FEATURE IMPORTANCE IN PCA:")
    print("-" * 35)

    # Get feature importance (absolute loadings) for first 3 components
    print("Top 5 features contributing to first 3 principal components:")

    for comp_idx in range(min(3, len(pca.components_))):
        print(f"\nPrincipal Component {comp_idx + 1}:")

        # Get feature loadings for this component
        loadings = np.abs(pca.components_[comp_idx])
        feature_importance = pd.DataFrame(
            {"Feature": features, "Loading": loadings}
        ).sort_values("Loading", ascending=False)

        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"  {i+1:2d}. {row['Feature']:25s}: {row['Loading']:.3f}")
    print()

    # 3. t-SNE for Visualization
    print("3. T-SNE FOR VISUALIZATION:")
    print("-" * 30)

    # Apply t-SNE for 2D visualization
    print("Applying t-SNE for 2D visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    print(f"✅ t-SNE completed")
    print(f"  Original dimensions: {X.shape[1]}")
    print(f"  t-SNE dimensions: {X_tsne.shape[1]}")
    print()

    # 4. Dimensionality Reduction Comparison
    print("4. DIMENSIONALITY REDUCTION COMPARISON:")
    print("-" * 45)

    # Compare different numbers of PCA components
    n_components_list = [2, 3, 5, 10, 15, 20]

    print("Variance explained with different numbers of PCA components:")
    for n_comp in n_components_list:
        if n_comp <= len(explained_variance_ratio):
            var_explained = cumulative_variance_ratio[n_comp - 1]
            print(
                f"  {n_comp:2d} components: {var_explained:.3f} ({var_explained*100:.1f}%)"
            )
    print()

    # 5. Create dimensionality reduction visualization
    print("5. CREATING DIMENSIONALITY REDUCTION VISUALIZATION:")
    print("-" * 50)

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Dimensionality Reduction Results - Real Breast Cancer Data",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Explained variance plot
        n_components = min(20, len(explained_variance_ratio))
        axes[0, 0].plot(
            range(1, n_components + 1),
            explained_variance_ratio[:n_components],
            "bo-",
            linewidth=2,
            markersize=6,
        )
        axes[0, 0].set_title("Explained Variance by Component")
        axes[0, 0].set_xlabel("Principal Component")
        axes[0, 0].set_ylabel("Explained Variance Ratio")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cumulative explained variance
        axes[0, 1].plot(
            range(1, n_components + 1),
            cumulative_variance_ratio[:n_components],
            "ro-",
            linewidth=2,
            markersize=6,
        )
        axes[0, 1].axhline(y=0.95, color="green", linestyle="--", label="95% Threshold")
        axes[0, 1].set_title("Cumulative Explained Variance")
        axes[0, 1].set_xlabel("Number of Components")
        axes[0, 1].set_ylabel("Cumulative Explained Variance Ratio")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. First two principal components scatter plot
        scatter1 = axes[0, 2].scatter(
            X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="viridis", alpha=0.7
        )
        axes[0, 2].set_title("First Two Principal Components")
        axes[0, 2].set_xlabel("Principal Component 1")
        axes[0, 2].set_ylabel("Principal Component 2")
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 2], label="Diagnosis")

        # 4. t-SNE visualization
        scatter2 = axes[1, 0].scatter(
            X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap="viridis", alpha=0.7
        )
        axes[1, 0].set_title("t-SNE 2D Visualization")
        axes[1, 0].set_xlabel("t-SNE Component 1")
        axes[1, 0].set_ylabel("t-SNE Component 2")
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1, 0], label="Diagnosis")

        # 5. Feature importance heatmap (first 5 components)
        n_comp_heatmap = min(5, len(pca.components_))
        if n_comp_heatmap > 0:
            # Get top 10 features for heatmap
            top_features_idx = np.argsort(
                np.sum(np.abs(pca.components_[:n_comp_heatmap]), axis=0)
            )[-10:]
            top_features = [features[i] for i in top_features_idx]

            # Create heatmap data
            heatmap_data = pca.components_[:n_comp_heatmap, top_features_idx]

            sns.heatmap(
                heatmap_data,
                xticklabels=top_features,
                yticklabels=[f"PC{i+1}" for i in range(n_comp_heatmap)],
                cmap="coolwarm",
                center=0,
                ax=axes[1, 1],
                cbar_kws={"label": "Component Loading"},
            )
            axes[1, 1].set_title("Feature Loadings in Top 5 Components")
            axes[1, 1].tick_params(axis="x", rotation=45)

        # 6. 3D PCA visualization
        if X_pca.shape[1] >= 3:
            ax3d = fig.add_subplot(2, 3, 6, projection="3d")
            scatter3d = ax3d.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                X_pca[:, 2],
                c=y_true,
                cmap="viridis",
                alpha=0.7,
            )
            ax3d.set_title("First Three Principal Components (3D)")
            ax3d.set_xlabel("PC1")
            ax3d.set_ylabel("PC2")
            ax3d.set_zlabel("PC3")
            plt.colorbar(scatter3d, ax=ax3d, label="Diagnosis")

        plt.tight_layout()

        # Save the visualization
        output_file = "dimensionality_reduction.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")

        plt.show()

    except Exception as e:
        print(f"  ❌ Error creating visualization: {e}")

    print()

    # 6. Summary
    print("6. DIMENSIONALITY REDUCTION SUMMARY:")
    print("-" * 40)

    print(
        f"✅ PCA successfully reduced {X.shape[1]} features to {X_pca.shape[1]} components"
    )
    print(
        f"✅ 95% variance explained with {np.argmax(cumulative_variance_ratio >= 0.95) + 1} components"
    )
    print(f"✅ t-SNE provided 2D visualization for data exploration")
    print(f"✅ Top contributing features identified for each principal component")
    print(
        f"✅ Visualization created showing explained variance and component relationships"
    )
    print()

    # Store dimensionality reduction results globally
    global dr_results
    dr_results = {
        "pca": pca,
        "X_pca": X_pca,
        "tsne": tsne,
        "X_tsne": X_tsne,
        "explained_variance": explained_variance_ratio,
        "cumulative_variance": cumulative_variance_ratio,
        "features": features,
        "X": X,
        "y_true": y_true,
    }

    return dr_results


if __name__ == "__main__":
    main()
