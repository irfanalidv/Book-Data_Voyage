#!/usr/bin/env python3
"""
Chapter 7: Exploratory Data Analysis (EDA)
Data Voyage: Discovering Insights and Patterns in Data

This script covers essential EDA concepts using REAL datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import requests
import os

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def main():
    print("=" * 80)
    print("CHAPTER 7: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    print()

    # Section 7.1: Data Overview and Summary Statistics
    print("7.1 DATA OVERVIEW AND SUMMARY STATISTICS")
    print("-" * 50)
    demonstrate_data_overview()

    # Section 7.2: Univariate Analysis
    print("\n7.2 UNIVARIATE ANALYSIS")
    print("-" * 40)
    demonstrate_univariate_analysis()

    # Section 7.3: Bivariate Analysis
    print("\n7.3 BIVARIATE ANALYSIS")
    print("-" * 40)
    demonstrate_bivariate_analysis()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Data overview and summary statistics - Understanding real data structure")
    print("✅ Univariate analysis - Individual variable exploration with real data")
    print("✅ Bivariate analysis - Relationship between variables in real datasets")
    print()
    print("Next: Chapter 8 - Statistical Inference and Hypothesis Testing")
    print("=" * 80)


def demonstrate_data_overview():
    """Demonstrate data overview and summary statistics using REAL datasets."""
    print("Data Overview and Summary Statistics:")
    print("-" * 40)

    # Load multiple real datasets for comprehensive EDA
    print("Loading real datasets for EDA...")

    # 1. Load sklearn built-in datasets
    try:
        from sklearn.datasets import (
            load_iris,
            load_diabetes,
            load_breast_cancer,
            load_wine,
        )

        iris = load_iris()
        diabetes = load_diabetes()
        breast_cancer = load_breast_cancer()
        wine = load_wine()

        print("✅ Loaded sklearn datasets:")
        print(f"  • Iris: {iris.data.shape[0]} samples, {iris.data.shape[1]} features")
        print(
            f"  • Diabetes: {diabetes.data.shape[0]} samples, {diabetes.data.shape[1]} features"
        )
        print(
            f"  • Breast Cancer: {breast_cancer.data.shape[0]} samples, {breast_cancer.data.shape[1]} features"
        )
        print(f"  • Wine: {wine.data.shape[0]} samples, {wine.data.shape[1]} features")

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

    except ImportError:
        print("❌ sklearn not available, using simulated data")
        iris_df = None
        diabetes_df = None
        breast_cancer_df = None
        wine_df = None

    # 2. Load real-world dataset from GitHub
    try:
        # Download a real dataset from GitHub
        print("\n📥 Downloading real dataset from GitHub...")

        # Try to download a popular dataset
        dataset_url = (
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/iris.csv"
        )
        response = requests.get(dataset_url, timeout=10)

        if response.status_code == 200:
            # Save to local file
            os.makedirs("real_datasets", exist_ok=True)
            with open("real_datasets/iris_github.csv", "w") as f:
                f.write(response.text)

            # Load the downloaded data
            iris_github_df = pd.read_csv("real_datasets/iris_github.csv")
            print(
                f"✅ Downloaded iris dataset: {iris_github_df.shape[0]} samples, {iris_github_df.shape[1]} features"
            )
            print(f"  Columns: {list(iris_github_df.columns)}")
            print(f"  Species: {iris_github_df['Species'].unique()}")
        else:
            print(f"❌ Failed to download dataset: {response.status_code}")
            iris_github_df = None

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        iris_github_df = None

    # 3. Create a comprehensive dataset combining multiple sources
    print("\n🔗 Creating comprehensive dataset for analysis...")

    if iris_df is not None:
        # Use the sklearn iris dataset as our main dataset
        df = iris_df.copy()

        # Add some derived features for analysis
        df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]
        df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]
        df["petal_to_sepal_ratio"] = df["petal_area"] / df["sepal_area"]

        # Add size categories
        df["size_category"] = pd.cut(
            df["petal_area"],
            bins=[0, 2, 5, 10, float("inf")],
            labels=["Small", "Medium", "Large", "Extra Large"],
        )

        print(
            f"✅ Created comprehensive iris dataset: {df.shape[0]} samples, {df.shape[1]} features"
        )
        print(f"  Original features: {list(iris.feature_names)}")
        print(
            f"  Derived features: petal_area, sepal_area, petal_to_sepal_ratio, size_category"
        )
        print(f"  Target variable: species ({df['species'].nunique()} unique values)")

        # Display basic information
        print(f"\n📊 Dataset Information:")
        print(f"  Shape: {df.shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print(f"  Data types: {df.dtypes.value_counts().to_dict()}")

        # Display summary statistics
        print(f"\n📈 Summary Statistics:")
        print(df.describe().round(3))

        # Display target distribution
        print(f"\n🎯 Target Distribution:")
        target_counts = df["species"].value_counts()
        for species, count in target_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {species}: {count} samples ({percentage:.1f}%)")

        # Store the main dataset globally
        global main_dataset
        main_dataset = df

    else:
        print("❌ Could not create comprehensive dataset")
        main_dataset = None

    print()


def demonstrate_univariate_analysis():
    """Demonstrate univariate analysis using REAL data."""
    print("Univariate Analysis:")
    print("-" * 40)

    if "main_dataset" not in globals() or main_dataset is None:
        print("❌ No dataset available for analysis")
        return

    df = main_dataset

    print("Analyzing individual variables in the iris dataset...")

    # 1. Numerical Variables Analysis
    print("\n1. NUMERICAL VARIABLES ANALYSIS:")
    print("-" * 30)

    numerical_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    for col in numerical_cols:
        print(f"\n📊 {col}:")
        print(f"  Mean: {df[col].mean():.3f}")
        print(f"  Median: {df[col].median():.3f}")
        print(f"  Std Dev: {df[col].std():.3f}")
        print(f"  Min: {df[col].min():.3f}")
        print(f"  Max: {df[col].max():.3f}")
        print(f"  Skewness: {df[col].skew():.3f}")
        print(f"  Kurtosis: {df[col].kurtosis():.3f}")

        # Check for outliers using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"  Outliers: {len(outliers)} samples ({len(outliers)/len(df)*100:.1f}%)")

    # 2. Categorical Variables Analysis
    print("\n2. CATEGORICAL VARIABLES ANALYSIS:")
    print("-" * 30)

    categorical_cols = ["species", "size_category"]

    for col in categorical_cols:
        if col in df.columns:
            print(f"\n📊 {col}:")
            value_counts = df[col].value_counts()
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Distribution:")
            for value, count in value_counts.items():
                percentage = (count / len(df)) * 100
                print(f"    {value}: {count} ({percentage:.1f}%)")

    # 3. Create univariate visualizations
    print("\n3. CREATING UNIVARIATE VISUALIZATIONS:")
    print("-" * 30)

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Univariate Analysis of Iris Dataset", fontsize=16, fontweight="bold"
        )

        # Histograms for numerical variables
        for i, col in enumerate(numerical_cols):
            row = i // 2
            col_idx = i % 2
            axes[row, col_idx].hist(df[col], bins=20, alpha=0.7, edgecolor="black")
            axes[row, col_idx].set_title(f"{col} Distribution")
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel("Frequency")
            axes[row, col_idx].grid(True, alpha=0.3)

        # Box plots for numerical variables
        axes[0, 2].boxplot(
            [df[col] for col in numerical_cols],
            labels=[col.split()[0] for col in numerical_cols],
        )
        axes[0, 2].set_title("Box Plots of All Features")
        axes[0, 2].set_ylabel("Measurement (cm)")
        axes[0, 2].grid(True, alpha=0.3)

        # Species distribution
        species_counts = df["species"].value_counts()
        axes[1, 2].pie(
            species_counts.values,
            labels=species_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1, 2].set_title("Species Distribution")

        # Size category distribution
        if "size_category" in df.columns:
            size_counts = df["size_category"].value_counts()
            axes[1, 1].bar(
                size_counts.index,
                size_counts.values,
                color="skyblue",
                edgecolor="black",
            )
            axes[1, 1].set_title("Size Category Distribution")
            axes[1, 1].set_xlabel("Size Category")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the visualization
        output_file = "univariate_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")

        plt.show()

    except Exception as e:
        print(f"  ❌ Error creating visualizations: {e}")

    print()


def demonstrate_bivariate_analysis():
    """Demonstrate bivariate analysis using REAL data."""
    print("Bivariate Analysis:")
    print("-" * 40)

    if "main_dataset" not in globals() or main_dataset is None:
        print("❌ No dataset available for analysis")
        return

    df = main_dataset

    print("Analyzing relationships between variables in the iris dataset...")

    # 1. Correlation Analysis
    print("\n1. CORRELATION ANALYSIS:")
    print("-" * 25)

    numerical_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))

    # Find strongest correlations
    print("\nStrongest Correlations:")
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            corr = correlation_matrix.iloc[i, j]
            print(f"  {numerical_cols[i]} vs {numerical_cols[j]}: {corr:.3f}")

    # 2. Group Analysis by Species
    print("\n2. GROUP ANALYSIS BY SPECIES:")
    print("-" * 30)

    for col in numerical_cols:
        print(f"\n📊 {col} by Species:")
        species_stats = (
            df.groupby("species")[col].agg(["mean", "std", "min", "max"]).round(3)
        )
        print(species_stats)

        # Statistical significance test (ANOVA)
        groups = [
            df[df["species"] == species][col].values
            for species in df["species"].unique()
        ]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"  ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.6f}")
        print(
            f"  {'Significant' if p_value < 0.05 else 'Not significant'} difference between species"
        )

    # 3. Create bivariate visualizations
    print("\n3. CREATING BIVARIATE VISUALIZATIONS:")
    print("-" * 30)

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Bivariate Analysis of Iris Dataset", fontsize=16, fontweight="bold"
        )

        # Scatter plots for key relationships
        # Petal length vs Petal width
        axes[0, 0].scatter(
            df["petal length (cm)"],
            df["petal width (cm)"],
            c=df["target"],
            cmap="viridis",
            alpha=0.7,
        )
        axes[0, 0].set_xlabel("Petal Length (cm)")
        axes[0, 0].set_ylabel("Petal Width (cm)")
        axes[0, 0].set_title("Petal Length vs Width")
        axes[0, 0].grid(True, alpha=0.3)

        # Sepal length vs Sepal width
        axes[0, 1].scatter(
            df["sepal length (cm)"],
            df["sepal width (cm)"],
            c=df["target"],
            cmap="viridis",
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("Sepal Length (cm)")
        axes[0, 1].set_ylabel("Sepal Width (cm)")
        axes[0, 1].set_title("Sepal Length vs Width")
        axes[0, 1].grid(True, alpha=0.3)

        # Petal length vs Sepal length
        axes[0, 2].scatter(
            df["petal length (cm)"],
            df["sepal length (cm)"],
            c=df["target"],
            cmap="viridis",
            alpha=0.7,
        )
        axes[0, 2].set_xlabel("Petal Length (cm)")
        axes[0, 2].set_ylabel("Sepal Length (cm)")
        axes[0, 2].set_title("Petal vs Sepal Length")
        axes[0, 2].grid(True, alpha=0.3)

        # Correlation heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            ax=axes[1, 0],
            cbar_kws={"label": "Correlation Coefficient"},
        )
        axes[1, 0].set_title("Feature Correlation Heatmap")

        # Box plots by species for petal length
        df.boxplot(column="petal length (cm)", by="species", ax=axes[1, 1])
        axes[1, 1].set_title("Petal Length by Species")
        axes[1, 1].set_xlabel("Species")
        axes[1, 1].set_ylabel("Petal Length (cm)")
        axes[1, 1].grid(True, alpha=0.3)

        # Violin plot for sepal width by species
        sns.violinplot(data=df, x="species", y="sepal width (cm)", ax=axes[1, 2])
        axes[1, 2].set_title("Sepal Width Distribution by Species")
        axes[1, 2].set_xlabel("Species")
        axes[1, 2].set_ylabel("Sepal Width (cm)")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the visualization
        output_file = "bivariate_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")

        plt.show()

    except Exception as e:
        print(f"  ❌ Error creating visualizations: {e}")

    print()


if __name__ == "__main__":
    main()
