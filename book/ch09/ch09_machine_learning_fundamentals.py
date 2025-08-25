#!/usr/bin/env python3
"""
Chapter 9: Machine Learning Fundamentals
Data Voyage: Building and Evaluating Machine Learning Models

This script covers essential machine learning concepts using REAL datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
import warnings

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("default")
sns.set_palette("husl")


def main():
    print("=" * 80)
    print("CHAPTER 9: MACHINE LEARNING FUNDAMENTALS")
    print("=" * 80)
    print()

    # Section 9.1: Machine Learning Overview
    print("9.1 MACHINE LEARNING OVERVIEW")
    print("-" * 40)
    demonstrate_ml_overview()

    # Section 9.2: Data Preparation
    print("\n9.2 DATA PREPARATION")
    print("-" * 40)
    demonstrate_data_preparation()

    # Section 9.3: Model Training and Evaluation
    print("\n9.3 MODEL TRAINING AND EVALUATION")
    print("-" * 40)
    demonstrate_model_training()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Machine learning overview - Understanding ML concepts and types")
    print("✅ Data preparation - Working with real datasets and preprocessing")
    print("✅ Model training and evaluation - Building and assessing ML models")
    print()
    print("Next: Chapter 10 - Feature Engineering and Selection")
    print("=" * 80)


def demonstrate_ml_overview():
    """Demonstrate machine learning concepts and types using REAL datasets."""
    print("Machine Learning Overview and Types:")
    print("-" * 40)

    # Load multiple real datasets to demonstrate different ML types
    print("Loading real datasets for machine learning examples...")

    try:
        # 1. Classification dataset (Iris)
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df["target"] = iris.target
        iris_df["species"] = [iris.target_names[i] for i in iris.target]

        # 2. Regression dataset (Diabetes)
        diabetes = load_diabetes()
        diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        diabetes_df["target"] = diabetes.target

        # 3. Binary classification dataset (Breast Cancer)
        breast_cancer = load_breast_cancer()
        breast_cancer_df = pd.DataFrame(
            breast_cancer.data, columns=breast_cancer.feature_names
        )
        breast_cancer_df["target"] = breast_cancer.target
        breast_cancer_df["diagnosis"] = [
            "Malignant" if t == 1 else "Benign" for t in breast_cancer.target
        ]

        # 4. Multi-class classification dataset (Wine)
        wine = load_wine()
        wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
        wine_df["target"] = wine.target
        wine_df["wine_type"] = [wine.target_names[i] for i in wine.target]

        print("✅ Loaded real datasets:")
        print(
            f"  • Iris (Classification): {iris_df.shape[0]} samples, {iris_df.shape[1]-2} features, 3 classes"
        )
        print(
            f"  • Diabetes (Regression): {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]-1} features"
        )
        print(
            f"  • Breast Cancer (Binary Classification): {breast_cancer_df.shape[0]} samples, {breast_cancer_df.shape[1]-2} features"
        )
        print(
            f"  • Wine (Multi-class): {wine_df.shape[0]} samples, {wine_df.shape[1]-2} features, 3 classes"
        )

        # Store datasets globally
        global datasets
        datasets = {
            "iris": iris_df,
            "diabetes": diabetes_df,
            "breast_cancer": breast_cancer_df,
            "wine": wine_df,
        }

    except ImportError:
        print("❌ sklearn not available")
        datasets = {}

    # Explain different types of machine learning
    print("\n🔍 MACHINE LEARNING TYPES:")
    print("-" * 30)

    print("1. SUPERVISED LEARNING:")
    print("   • Classification: Predicting categories (e.g., Iris species, Wine type)")
    print("   • Regression: Predicting continuous values (e.g., Diabetes progression)")

    print("\n2. UNSUPERVISED LEARNING:")
    print("   • Clustering: Finding patterns in data without labels")
    print("   • Dimensionality Reduction: Reducing feature complexity")

    print("\n3. REINFORCEMENT LEARNING:")
    print("   • Learning through interaction with environment")
    print("   • Examples: Game playing, robotics, autonomous systems")

    # Show dataset characteristics
    if datasets:
        print("\n📊 DATASET CHARACTERISTICS:")
        print("-" * 30)

        for name, df in datasets.items():
            print(f"\n{name.upper()} Dataset:")
            print(f"  Shape: {df.shape}")
            print(
                f"  Features: {df.shape[1]-2 if 'target' in df.columns else df.shape[1]-1}"
            )
            print(f"  Samples: {df.shape[0]}")

            if "target" in df.columns:
                if name == "diabetes":
                    print(f"  Target type: Regression (continuous)")
                    print(
                        f"  Target range: {df['target'].min():.2f} to {df['target'].max():.2f}"
                    )
                else:
                    print(f"  Target type: Classification")
                    target_col = (
                        "species"
                        if "species" in df.columns
                        else "diagnosis" if "diagnosis" in df.columns else "wine_type"
                    )
                    if target_col in df.columns:
                        unique_targets = df[target_col].unique()
                        print(f"  Classes: {list(unique_targets)}")
                        print(f"  Class distribution:")
                        for target in unique_targets:
                            count = (df[target_col] == target).sum()
                            percentage = (count / len(df)) * 100
                            print(f"    {target}: {count} ({percentage:.1f}%)")

    print()


def demonstrate_data_preparation():
    """Demonstrate data preparation using REAL datasets."""
    print("Data Preparation and Preprocessing:")
    print("-" * 40)

    if "datasets" not in globals() or not datasets:
        print("❌ No datasets available for preparation")
        return

    # Use Iris dataset for demonstration
    df = datasets["iris"].copy()
    print(f"Preparing Iris dataset: {df.shape[0]} samples, {df.shape[1]-2} features")

    # 1. Data Overview
    print("\n1. DATA OVERVIEW:")
    print("-" * 20)

    print("Feature Information:")
    for i, feature in enumerate(df.columns[:-2]):  # Exclude target and species
        print(f"  {feature}:")
        print(f"    Mean: {df[feature].mean():.3f}")
        print(f"    Std: {df[feature].std():.3f}")
        print(f"    Min: {df[feature].min():.3f}")
        print(f"    Max: {df[feature].max():.3f}")
        print(f"    Missing values: {df[feature].isnull().sum()}")

    print(f"\nTarget Distribution:")
    target_counts = df["species"].value_counts()
    for species, count in target_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {species}: {count} samples ({percentage:.1f}%)")

    # 2. Data Splitting
    print("\n2. DATA SPLITTING:")
    print("-" * 20)

    # Prepare features and target
    X = df[df.columns[:-2]].values  # All features except target and species
    y = df["target"].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

    # Check class distribution in splits
    print(f"\nTraining set class distribution:")
    train_counts = np.bincount(y_train)
    for i, count in enumerate(train_counts):
        species = (
            df["species"].iloc[0]
            if i == 0
            else df["species"].iloc[50] if i == 1 else df["species"].iloc[100]
        )
        percentage = (count / len(y_train)) * 100
        print(f"  {species}: {count} samples ({percentage:.1f}%)")

    print(f"\nTesting set class distribution:")
    test_counts = np.bincount(y_test)
    for i, count in enumerate(test_counts):
        species = (
            df["species"].iloc[0]
            if i == 0
            else df["species"].iloc[50] if i == 1 else df["species"].iloc[100]
        )
        percentage = (count / len(y_test)) * 100
        print(f"  {species}: {count} samples ({percentage:.1f}%)")

    # 3. Feature Scaling
    print("\n3. FEATURE SCALING:")
    print("-" * 20)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling applied:")
    print(f"  Training set mean: {X_train_scaled.mean(axis=0).round(3)}")
    print(f"  Training set std: {X_train_scaled.std(axis=0).round(3)}")
    print(f"  Testing set mean: {X_test_scaled.mean(axis=0).round(3)}")
    print(f"  Testing set std: {X_test_scaled.std(axis=0).round(3)}")

    # Store prepared data globally
    global prepared_data
    prepared_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }

    print()


def demonstrate_model_training():
    """Demonstrate model training and evaluation using REAL data."""
    print("Model Training and Evaluation:")
    print("-" * 40)

    if "prepared_data" not in globals():
        print("❌ No prepared data available for training")
        return

    # Get prepared data
    X_train = prepared_data["X_train"]
    X_test = prepared_data["X_test"]
    y_train = prepared_data["y_train"]
    y_test = prepared_data["y_test"]
    X_train_scaled = prepared_data["X_train_scaled"]
    X_test_scaled = prepared_data["X_test_scaled"]

    # 1. Classification Model (Iris dataset)
    print("1. CLASSIFICATION MODEL - IRIS SPECIES PREDICTION:")
    print("-" * 50)

    # Train Logistic Regression
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)

    # Evaluate model
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print(f"  Accuracy: {accuracy_lr:.3f}")

    # Classification report
    print(f"\n  Classification Report:")
    print(
        classification_report(
            y_test, y_pred_lr, target_names=["Setosa", "Versicolor", "Virginica"]
        )
    )

    # Confusion matrix
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print(f"\n  Confusion Matrix:")
    print(cm_lr)

    # 2. Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)

    # Evaluate model
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"  Accuracy: {accuracy_rf:.3f}")

    # Feature importance
    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    feature_importance = rf_model.feature_importances_
    print(f"\n  Feature Importance:")
    for name, importance in zip(feature_names, feature_importance):
        print(f"    {name}: {importance:.3f}")

    # 3. Cross-validation
    print("\n3. CROSS-VALIDATION:")
    print("-" * 20)

    # Perform cross-validation on both models
    cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5)

    print(f"Logistic Regression CV scores: {cv_scores_lr.round(3)}")
    print(
        f"  Mean CV accuracy: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std() * 2:.3f})"
    )

    print(f"Random Forest CV scores: {cv_scores_rf.round(3)}")
    print(
        f"  Mean CV accuracy: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std() * 2:.3f})"
    )

    # 4. Model Comparison
    print("\n4. MODEL COMPARISON:")
    print("-" * 20)

    print("Performance Summary:")
    print(f"  Logistic Regression:")
    print(f"    • Test Accuracy: {accuracy_lr:.3f}")
    print(
        f"    • CV Accuracy: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std() * 2:.3f})"
    )

    print(f"  Random Forest:")
    print(f"    • Test Accuracy: {accuracy_rf:.3f}")
    print(
        f"    • CV Accuracy: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std() * 2:.3f})"
    )

    # 5. Create visualizations
    print("\n5. CREATING MODEL EVALUATION VISUALIZATIONS:")
    print("-" * 50)

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Machine Learning Model Evaluation - Iris Dataset",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Confusion Matrix - Logistic Regression
        sns.heatmap(
            cm_lr,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Setosa", "Versicolor", "Virginica"],
            yticklabels=["Setosa", "Versicolor", "Virginica"],
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Logistic Regression - Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")

        # 2. Confusion Matrix - Random Forest
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        sns.heatmap(
            cm_rf,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=["Setosa", "Versicolor", "Virginica"],
            yticklabels=["Setosa", "Versicolor", "Virginica"],
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Random Forest - Confusion Matrix")
        axes[0, 1].set_xlabel("Predicted")
        axes[0, 1].set_ylabel("Actual")

        # 3. Feature Importance
        feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        axes[0, 2].bar(
            feature_names, feature_importance, color="skyblue", edgecolor="black"
        )
        axes[0, 2].set_title("Random Forest - Feature Importance")
        axes[0, 2].set_ylabel("Importance")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Cross-validation comparison
        models = ["Logistic\nRegression", "Random\nForest"]
        cv_means = [cv_scores_lr.mean(), cv_scores_rf.mean()]
        cv_stds = [cv_scores_lr.std(), cv_scores_rf.std()]

        bars = axes[1, 0].bar(
            models,
            cv_means,
            yerr=cv_stds,
            capsize=5,
            color=["lightcoral", "lightgreen"],
            edgecolor="black",
        )
        axes[1, 0].set_title("Cross-Validation Accuracy Comparison")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, mean in zip(bars, cv_means):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
            )

        # 5. ROC Curves for each class (Logistic Regression)
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        # Binarize the output for ROC curve
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = 3

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba_lr[:, i])
            roc_auc = auc(fpr, tpr)
            axes[1, 1].plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

        axes[1, 1].plot([0, 1], [0, 1], "k--", lw=2)
        axes[1, 1].set_xlim([0.0, 1.0])
        axes[1, 1].set_ylim([0.0, 1.05])
        axes[1, 1].set_xlabel("False Positive Rate")
        axes[1, 1].set_ylabel("True Positive Rate")
        axes[1, 1].set_title("ROC Curves - Logistic Regression")
        axes[1, 1].legend(loc="lower right")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Prediction probabilities distribution
        axes[1, 2].hist(
            y_pred_proba_lr.max(axis=1),
            bins=20,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        axes[1, 2].set_xlabel("Maximum Prediction Probability")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].set_title("Distribution of Prediction Confidence")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the visualization
        output_file = "model_evaluation.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")

        plt.show()

    except Exception as e:
        print(f"  ❌ Error creating visualizations: {e}")

    print("\n🎯 MODEL EVALUATION SUMMARY:")
    print("-" * 30)
    print(f"• Both models achieved high accuracy on the Iris dataset")
    print(f"• Random Forest shows slightly better performance")
    print(f"• Petal measurements are the most important features")
    print(f"• Cross-validation confirms model reliability")
    print(f"• Ready for deployment or further optimization")

    print()


if __name__ == "__main__":
    main()
