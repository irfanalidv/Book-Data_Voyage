#!/usr/bin/env python3
"""
Chapter 7: Exploratory Data Analysis (EDA)
Data Voyage: Discovering Insights and Patterns in Data

This script covers essential EDA concepts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

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
    print("✅ Data overview and summary statistics - Understanding data structure")
    print("✅ Univariate analysis - Individual variable exploration")
    print("✅ Bivariate analysis - Relationship between two variables")
    print()
    print("Next: Chapter 8 - Statistical Inference and Hypothesis Testing")
    print("=" * 80)


def demonstrate_data_overview():
    """Demonstrate data overview and summary statistics."""
    print("Data Overview and Summary Statistics:")
    print("-" * 40)

    # Create a comprehensive dataset for EDA
    print("Creating comprehensive dataset for EDA...")

    np.random.seed(42)
    n_records = 500

    # Generate realistic data
    ages = np.random.normal(35, 12, n_records)
    ages = np.clip(ages, 18, 80)

    incomes = np.random.lognormal(10.5, 0.6, n_records)
    incomes = np.clip(incomes, 20000, 200000)

    education_years = np.random.poisson(16, n_records)
    education_years = np.clip(education_years, 8, 25)

    # Create correlated features
    credit_scores = (
        300 + (ages * 2) + (incomes / 1000) + np.random.normal(0, 50, n_records)
    )
    credit_scores = np.clip(credit_scores, 300, 850)

    # Create categorical variables
    cities = np.random.choice(
        ["NYC", "LA", "Chicago", "Boston", "Seattle"],
        size=n_records,
        p=[0.3, 0.25, 0.2, 0.15, 0.1],
    )

    employment_status = np.random.choice(
        ["Full-time", "Part-time", "Self-employed"],
        size=n_records,
        p=[0.75, 0.15, 0.10],
    )

    # Create the dataset
    data = {
        "customer_id": range(1, n_records + 1),
        "age": ages,
        "income": incomes,
        "education_years": education_years,
        "credit_score": credit_scores,
        "city": cities,
        "employment_status": employment_status,
        "total_purchases": np.random.poisson(5, n_records),
        "avg_purchase_amount": np.random.exponential(100, n_records),
    }

    df = pd.DataFrame(data)

    # Add derived features
    df["income_category"] = pd.cut(
        df["income"],
        bins=[0, 50000, 100000, 150000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
    )

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"],
    )

    df["customer_lifetime_value"] = df["total_purchases"] * df["avg_purchase_amount"]

    print(f"✅ Created dataset with {len(df)} records and {len(df.columns)} features")
    print(f"Dataset shape: {df.shape}")
    print()

    # Basic Dataset Information
    print("1. BASIC DATASET INFORMATION:")
    print("-" * 30)

    print("Dataset Info:")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Data types: {df.dtypes.nunique()} unique types")
    print()

    # Display data types
    print("Data Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} features")
    print()

    # Summary Statistics
    print("2. SUMMARY STATISTICS:")
    print("-" * 25)

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("Numeric Features Summary:")
    print(df[numeric_cols].describe().round(2))
    print()

    # Categorical summary
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    print("Categorical Features Summary:")
    for col in categorical_cols:
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most common: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
        print(
            f"  Least common: {value_counts.index[-1]} ({value_counts.iloc[-1]} times)"
        )
    print()

    # Store dataset for later use
    global eda_df
    eda_df = df.copy()


def demonstrate_univariate_analysis():
    """Demonstrate univariate analysis techniques."""
    print("Univariate Analysis and Distribution Exploration:")
    print("-" * 40)

    # Start with the EDA dataset
    df = eda_df.copy()
    print(f"Starting univariate analysis with dataset: {df.shape}")
    print()

    # Numeric Variable Analysis
    print("1. NUMERIC VARIABLE ANALYSIS:")
    print("-" * 30)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col != "customer_id":
            print(f"\n{col.upper()} Analysis:")
            print("-" * 20)

            # Basic statistics
            stats_data = df[col].describe()
            print(f"  Mean: {stats_data['mean']:.2f}")
            print(f"  Median: {stats_data['50%']:.2f}")
            print(f"  Std: {stats_data['std']:.2f}")
            print(f"  Min: {stats_data['min']:.2f}")
            print(f"  Max: {stats_data['max']:.2f}")
            print(f"  Range: {stats_data['max'] - stats_data['min']:.2f}")
            print(f"  IQR: {stats_data['75%'] - stats_data['25%']:.2f}")

            # Skewness and kurtosis
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            print(f"  Skewness: {skewness:.3f}")
            print(f"  Kurtosis: {kurtosis:.3f}")

            # Outlier detection
            Q1 = stats_data["25%"]
            Q3 = stats_data["75%"]
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    print()

    # Categorical Variable Analysis
    print("2. CATEGORICAL VARIABLE ANALYSIS:")
    print("-" * 35)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        print(f"\n{col.upper()} Analysis:")
        print("-" * 20)

        # Value counts
        value_counts = df[col].value_counts()
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most common: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
        print(
            f"  Least common: {value_counts.index[-1]} ({value_counts.iloc[-1]} times)"
        )

        # Mode
        mode_value = df[col].mode()[0]
        mode_count = (df[col] == mode_value).sum()
        print(f"  Mode: {mode_value} ({mode_count} times)")
    print()

    # Create univariate plots
    plt.figure(figsize=(15, 10))

    # Numeric distributions
    for i, col in enumerate(numeric_cols[:4]):  # First 4 numeric columns
        if col != "customer_id":
            plt.subplot(2, 3, i + 1)
            plt.hist(df[col], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
            plt.title(f'{col.replace("_", " ").title()} Distribution')
            plt.xlabel(col.replace("_", " ").title())
            plt.ylabel("Frequency")

    # Categorical distributions
    for i, col in enumerate(categorical_cols[:2]):  # First 2 categorical columns
        plt.subplot(2, 3, 5 + i)
        df[col].value_counts().plot(kind="bar", color="lightgreen")
        plt.title(f'{col.replace("_", " ").title()} Distribution')
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel("Count")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("univariate_analysis.png", dpi=300, bbox_inches="tight")
    print("✅ Univariate analysis visualization saved as 'univariate_analysis.png'")
    plt.close()


def demonstrate_bivariate_analysis():
    """Demonstrate bivariate analysis techniques."""
    print("Bivariate Analysis and Relationship Exploration:")
    print("-" * 40)

    # Start with the EDA dataset
    df = eda_df.copy()
    print(f"Starting bivariate analysis with dataset: {df.shape}")
    print()

    # Numeric-Numeric Relationships
    print("1. NUMERIC-NUMERIC RELATIONSHIPS:")
    print("-" * 35)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "customer_id"]

    # Correlation analysis
    correlation_matrix = df[numeric_cols].corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    print()

    # Find strongest correlations
    print("Strongest Correlations (|r| > 0.3):")
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.3:
                print(f"  {numeric_cols[i]} vs {numeric_cols[j]}: {corr_value:.3f}")
    print()

    # Categorical-Numeric Relationships
    print("2. CATEGORICAL-NUMERIC RELATIONSHIPS:")
    print("-" * 40)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for cat_col in categorical_cols[:2]:  # Analyze first 2 categorical variables
        print(f"\n{cat_col.upper()} vs Numeric Variables:")
        print("-" * 30)

        for num_col in numeric_cols[:3]:  # Analyze first 3 numeric variables
            print(f"\n  {num_col} by {cat_col}:")

            # Group statistics
            group_stats = (
                df.groupby(cat_col)[num_col].agg(["mean", "std", "count"]).round(2)
            )
            print(f"    Group Statistics:")
            print(group_stats)
    print()

    # Create bivariate plots
    plt.figure(figsize=(15, 10))

    # Scatter plots for numeric relationships
    plt.subplot(2, 3, 1)
    plt.scatter(df["age"], df["income"], alpha=0.6, color="blue")
    plt.xlabel("Age")
    plt.ylabel("Income")
    plt.title("Age vs Income")

    # Add trend line
    z = np.polyfit(df["age"], df["income"], 1)
    p = np.poly1d(z)
    plt.plot(df["age"], p(df["age"]), "r--", alpha=0.8)

    plt.subplot(2, 3, 2)
    plt.scatter(df["education_years"], df["income"], alpha=0.6, color="green")
    plt.xlabel("Education Years")
    plt.ylabel("Income")
    plt.title("Education vs Income")

    # Add trend line
    z = np.polyfit(df["education_years"], df["income"], 1)
    p = np.poly1d(z)
    plt.plot(df["education_years"], p(df["education_years"]), "r--", alpha=0.8)

    # Box plots for categorical-numeric
    plt.subplot(2, 3, 3)
    df.boxplot(column="income", by="income_category", ax=plt.gca())
    plt.title("Income by Income Category")
    plt.suptitle("")

    # Heatmap for correlations
    plt.subplot(2, 3, 4)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")

    # Violin plots
    plt.subplot(2, 3, 5)
    sns.violinplot(data=df, x="age_group", y="credit_score", palette="Set2")
    plt.title("Credit Score by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Credit Score")
    plt.xticks(rotation=45)

    # Scatter with color by category
    plt.subplot(2, 3, 6)
    for status in df["employment_status"].unique():
        subset = df[df["employment_status"] == status]
        plt.scatter(subset["age"], subset["income"], alpha=0.6, label=status)
    plt.xlabel("Age")
    plt.ylabel("Income")
    plt.title("Age vs Income by Employment Status")
    plt.legend()

    plt.tight_layout()
    plt.savefig("bivariate_analysis.png", dpi=300, bbox_inches="tight")
    print("✅ Bivariate analysis visualization saved as 'bivariate_analysis.png'")
    plt.close()

    # Final summary
    print("\nBIVARIATE ANALYSIS SUMMARY:")
    print("-" * 30)
    print(
        f"Strong correlations found: {len([c for c in correlation_matrix.values.flatten() if abs(c) > 0.3 and c != 1.0])}"
    )
    print(
        f"Features analyzed: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical"
    )
    print("\nExploratory Data Analysis complete!")
    print("Key insights and patterns have been identified and visualized.")


if __name__ == "__main__":
    main()
