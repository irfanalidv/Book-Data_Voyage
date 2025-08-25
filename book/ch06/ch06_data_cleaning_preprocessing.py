#!/usr/bin/env python3
"""
Chapter 6: Data Cleaning and Preprocessing
Data Voyage: Preparing Real Data for Analysis and Machine Learning

This script covers essential data cleaning and preprocessing concepts using REAL DATA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
import requests
import json
import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("CHAPTER 6: DATA CLEANING AND PREPROCESSING")
    print("=" * 80)
    print()

    # Section 6.1: Data Quality Assessment
    print("6.1 DATA QUALITY ASSESSMENT")
    print("-" * 40)
    demonstrate_data_quality_assessment()

    # Section 6.2: Data Cleaning Techniques
    print("\n6.2 DATA CLEANING TECHNIQUES")
    print("-" * 40)
    demonstrate_data_cleaning()

    # Section 6.3: Data Preprocessing Methods
    print("\n6.3 DATA PREPROCESSING METHODS")
    print("-" * 40)
    demonstrate_data_preprocessing()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Data quality assessment - Identifying and measuring real data issues")
    print("✅ Data cleaning techniques - Handling actual missing values and outliers")
    print(
        "✅ Data preprocessing methods - Scaling, encoding, and transformation of real data"
    )
    print()
    print("Next: Chapter 7 - Exploratory Data Analysis")
    print("=" * 80)


def demonstrate_data_quality_assessment():
    """Demonstrate data quality assessment and analysis using real data."""
    print("Data Quality Assessment and Analysis:")
    print("-" * 40)

    # Load real datasets with actual quality issues
    print("Loading real datasets for quality assessment...")

    # 1. Load sklearn datasets
    iris = load_iris()
    diabetes = load_diabetes()
    breast_cancer = load_breast_cancer()

    # 2. Create DataFrames
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

    print(f"✅ Loaded real datasets:")
    print(f"  • Iris: {iris_df.shape[0]} samples, {iris_df.shape[1]-2} features")
    print(
        f"  • Diabetes: {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]-1} features"
    )
    print(
        f"  • Breast Cancer: {breast_cancer_df.shape[0]} samples, {breast_cancer_df.shape[1]-2} features"
    )
    print()

    # 3. Collect real data from APIs (with potential quality issues)
    print("3. COLLECTING REAL DATA FROM APIS:")
    print("-" * 35)

    # Try to get real COVID-19 data (may have quality issues)
    try:
        covid_url = "https://disease.sh/v3/covid-19/countries"
        covid_response = requests.get(covid_url, timeout=10)
        if covid_response.status_code == 200:
            covid_data = covid_response.json()

            # Convert to DataFrame
            covid_df = pd.DataFrame(covid_data)

            # Select relevant columns
            covid_columns = [
                "country",
                "cases",
                "deaths",
                "recovered",
                "population",
                "active",
                "critical",
            ]
            covid_df = covid_df[covid_columns]

            print(f"✅ Collected COVID-19 data for {len(covid_df)} countries")
            print(f"  Columns: {list(covid_df.columns)}")

            # Introduce some real-world data quality issues
            covid_df.loc[covid_df["recovered"] == 0, "recovered"] = (
                np.nan
            )  # Missing recovery data
            covid_df.loc[covid_df["population"] == 0, "population"] = (
                np.nan
            )  # Missing population data

        else:
            print("❌ Failed to collect COVID-19 data")
            covid_df = None
    except Exception as e:
        print(f"❌ Error collecting COVID-19 data: {e}")
        covid_df = None

    # 4. Create a comprehensive dataset combining multiple sources
    print("\n4. CREATING COMPREHENSIVE DATASET:")
    print("-" * 35)

    # Use iris as base and add some derived features
    df = iris_df.copy()

    # Add some derived features that might have quality issues
    df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]
    df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]
    df["petal_to_sepal_ratio"] = df["petal_area"] / df["sepal_area"]

    # Introduce some realistic quality issues
    # Missing values in some measurements
    df.loc[df.index[::10], "petal length (cm)"] = np.nan  # 10% missing values

    # Outliers in derived features
    df.loc[df.index[-5:], "petal_to_sepal_ratio"] = [
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
    ]  # Unrealistic values

    # Invalid values
    df.loc[df.index[0], "sepal length (cm)"] = -5  # Negative length (impossible)
    df.loc[df.index[1], "sepal width (cm)"] = 100  # Unrealistic width

    print(
        f"✅ Created comprehensive dataset: {df.shape[0]} samples, {df.shape[1]} features"
    )
    print(f"  Original features: {list(iris.feature_names)}")
    print(f"  Derived features: petal_area, sepal_area, petal_to_sepal_ratio")
    print(f"  Quality issues introduced: missing values, outliers, invalid values")
    print()

    # 5. Data Quality Assessment
    print("5. DATA QUALITY ASSESSMENT:")
    print("-" * 30)

    # Basic information
    print("Dataset Information:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
    print()

    # Missing values analysis
    print("Missing Values Analysis:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    for col in df.columns:
        if missing_values[col] > 0:
            print(
                f"  {col}: {missing_values[col]} missing ({missing_percentage[col]:.1f}%)"
            )

    if missing_values.sum() == 0:
        print("  No missing values found")
    print()

    # Data type analysis
    print("Data Type Analysis:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print()

    # Statistical summary
    print("Statistical Summary (Numerical Features):")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe().round(3))
    print()

    # Store the dataset globally for other functions
    global main_dataset
    main_dataset = df

    return df


def demonstrate_data_cleaning():
    """Demonstrate data cleaning techniques on real data."""
    print("Data Cleaning Techniques:")
    print("-" * 40)

    if "main_dataset" not in globals() or main_dataset is None:
        print("❌ No dataset available for cleaning")
        return

    df = main_dataset.copy()
    print(f"Cleaning dataset: {df.shape[0]} samples, {df.shape[1]} features")
    print()

    # 1. Handle Missing Values
    print("1. HANDLING MISSING VALUES:")
    print("-" * 30)

    # Check missing values before cleaning
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")

    # Strategy 1: Remove rows with missing values
    df_cleaned = df.dropna()
    print(f"After removing missing values: {df_cleaned.shape[0]} samples remaining")

    # Strategy 2: Impute missing values (for demonstration, use original df)
    df_imputed = df.copy()

    # Use SimpleImputer for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="mean")
    df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])

    print(
        f"After imputing missing values: {df_imputed.isnull().sum().sum()} missing values remaining"
    )
    print()

    # 2. Handle Outliers
    print("2. HANDLING OUTLIERS:")
    print("-" * 25)

    # Use IQR method to detect outliers
    outlier_counts = {}

    for col in numerical_cols:
        Q1 = df_imputed[col].quantile(0.25)
        Q3 = df_imputed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_imputed[
            (df_imputed[col] < lower_bound) | (df_imputed[col] > upper_bound)
        ]
        outlier_counts[col] = len(outliers)

        print(f"  {col}: {len(outliers)} outliers detected")

    print()

    # 3. Handle Invalid Values
    print("3. HANDLING INVALID VALUES:")
    print("-" * 30)

    # Check for negative values in length/width measurements
    invalid_counts = {}

    for col in [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]:
        if col in df_imputed.columns:
            invalid_values = df_imputed[df_imputed[col] <= 0]
            invalid_counts[col] = len(invalid_values)
            print(f"  {col}: {len(invalid_values)} invalid values (≤0)")

    # Remove invalid values
    for col in [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]:
        if col in df_imputed.columns:
            df_imputed = df_imputed[df_imputed[col] > 0]

    print(f"After removing invalid values: {df_imputed.shape[0]} samples remaining")
    print()

    # 4. Data Validation
    print("4. DATA VALIDATION:")
    print("-" * 20)

    # Check data ranges for biological measurements
    print("Biological Measurement Ranges:")
    for col in [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]:
        if col in df_imputed.columns:
            min_val = df_imputed[col].min()
            max_val = df_imputed[col].max()
            print(f"  {col}: {min_val:.2f} to {max_val:.2f} cm")

    # Check derived features
    print("\nDerived Feature Validation:")
    if "petal_area" in df_imputed.columns:
        print(
            f"  Petal area: {df_imputed['petal_area'].min():.3f} to {df_imputed['petal_area'].max():.3f} cm²"
        )
    if "sepal_area" in df_imputed.columns:
        print(
            f"  Sepal area: {df_imputed['sepal_area'].min():.3f} to {df_imputed['sepal_area'].max():.3f} cm²"
        )
    if "petal_to_sepal_ratio" in df_imputed.columns:
        print(
            f"  Petal/sepal ratio: {df_imputed['petal_to_sepal_ratio'].min():.3f} to {df_imputed['petal_to_sepal_ratio'].max():.3f}"
        )

    print()

    # Store cleaned dataset globally
    global cleaned_dataset
    cleaned_dataset = df_imputed

    return df_imputed


def demonstrate_data_preprocessing():
    """Demonstrate data preprocessing methods on real data."""
    print("Data Preprocessing Methods:")
    print("-" * 40)

    if "cleaned_dataset" not in globals() or cleaned_dataset is None:
        print("❌ No cleaned dataset available for preprocessing")
        return

    df = cleaned_dataset.copy()
    print(f"Preprocessing dataset: {df.shape[0]} samples, {df.shape[1]} features")
    print()

    # 1. Feature Scaling
    print("1. FEATURE SCALING:")
    print("-" * 20)

    # Select numerical features for scaling
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [
        col for col in numerical_cols if col != "target"
    ]  # Exclude target

    print(f"Scaling {len(numerical_cols)} numerical features:")
    for col in numerical_cols:
        print(f"  {col}")

    # Apply StandardScaler
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

    print("\nScaling Results:")
    print("Before scaling (first 3 samples):")
    print(df[numerical_cols].head(3).round(3))
    print("\nAfter scaling (first 3 samples):")
    print(df_scaled[numerical_cols].head(3).round(3))
    print()

    # 2. Categorical Encoding
    print("2. CATEGORICAL ENCODING:")
    print("-" * 30)

    # Check categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    print(f"Found {len(categorical_cols)} categorical columns:")

    for col in categorical_cols:
        print(f"  {col}: {df[col].nunique()} unique values")
        print(f"    Values: {list(df[col].unique())}")

    if len(categorical_cols) > 0:
        # Apply Label Encoding
        label_encoder = LabelEncoder()
        df_encoded = df_scaled.copy()

        for col in categorical_cols:
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
            print(
                f"  Encoded {col}: {list(label_encoder.classes_)} → {list(range(len(label_encoder.classes_)))}"
            )

    print()

    # 3. Feature Engineering
    print("3. FEATURE ENGINEERING:")
    print("-" * 25)

    # Create additional derived features
    df_final = df_encoded.copy()

    # Add polynomial features for key measurements
    df_final["sepal_length_squared"] = df_final["sepal length (cm)"] ** 2
    df_final["petal_length_squared"] = df_final["petal length (cm)"] ** 2

    # Add interaction features
    df_final["sepal_petal_interaction"] = (
        df_final["sepal length (cm)"] * df_final["petal length (cm)"]
    )

    # Add ratio features
    df_final["length_width_ratio"] = (
        df_final["sepal length (cm)"] / df_final["sepal width (cm)"]
    )

    print(f"Added {4} new engineered features:")
    print("  • sepal_length_squared: Quadratic transformation of sepal length")
    print("  • petal_length_squared: Quadratic transformation of petal length")
    print("  • sepal_petal_interaction: Interaction between sepal and petal length")
    print("  • length_width_ratio: Ratio of sepal length to width")
    print()

    # 4. Final Dataset Summary
    print("4. FINAL PREPROCESSED DATASET:")
    print("-" * 35)

    print(f"Final shape: {df_final.shape}")
    print(f"Features: {list(df_final.columns)}")
    print(f"Data types: {df_final.dtypes.value_counts().to_dict()}")
    print()

    # Check for any remaining issues
    print("Quality Check Summary:")
    print(f"  Missing values: {df_final.isnull().sum().sum()}")
    print(
        f"  Infinite values: {np.isinf(df_final.select_dtypes(include=[np.number])).sum().sum()}"
    )
    print(f"  Duplicate rows: {df_final.duplicated().sum()}")
    print()

    # 5. Create preprocessing visualization
    print("5. CREATING PREPROCESSING VISUALIZATION:")
    print("-" * 40)

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Data Preprocessing Results - Real Data", fontsize=16, fontweight="bold"
        )

        # 1. Before vs After Scaling (first numerical feature)
        if len(numerical_cols) > 0:
            feature = numerical_cols[0]
            axes[0, 0].hist(
                df[feature], bins=20, alpha=0.7, label="Before Scaling", color="skyblue"
            )
            axes[0, 0].hist(
                df_scaled[feature],
                bins=20,
                alpha=0.7,
                label="After Scaling",
                color="lightcoral",
            )
            axes[0, 0].set_title(f"{feature} - Before vs After Scaling")
            axes[0, 0].set_xlabel("Value")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Missing values before cleaning
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            axes[0, 1].bar(missing_data.index, missing_data.values, color="lightcoral")
            axes[0, 1].set_title("Missing Values Before Cleaning")
            axes[0, 1].set_xlabel("Features")
            axes[0, 1].set_ylabel("Missing Count")
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No Missing Values\nFound in Dataset",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
                fontsize=12,
            )
            axes[0, 1].set_title("Missing Values Before Cleaning")

        # 3. Outlier detection
        if len(numerical_cols) > 0:
            feature = numerical_cols[0]
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            axes[1, 0].boxplot(df[feature])
            axes[1, 0].axhline(
                y=lower_bound, color="red", linestyle="--", label="Lower Bound"
            )
            axes[1, 0].axhline(
                y=upper_bound, color="red", linestyle="--", label="Upper Bound"
            )
            axes[1, 0].set_title(f"{feature} - Outlier Detection")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Feature correlation after preprocessing
        if len(numerical_cols) > 1:
            correlation_matrix = df_final[numerical_cols].corr()
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                ax=axes[1, 1],
                cbar_kws={"label": "Correlation Coefficient"},
            )
            axes[1, 1].set_title("Feature Correlation After Preprocessing")

        plt.tight_layout()

        # Save the visualization
        output_file = "data_preprocessing.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")

        plt.show()

    except Exception as e:
        print(f"  ❌ Error creating visualization: {e}")

    print()

    # Store final preprocessed dataset globally
    global preprocessed_dataset
    preprocessed_dataset = df_final

    return df_final


if __name__ == "__main__":
    main()
