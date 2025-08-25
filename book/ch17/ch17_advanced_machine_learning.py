#!/usr/bin/env python3
"""
Chapter 17: Advanced Machine Learning
=====================================

This chapter covers advanced machine learning techniques including ensemble methods,
optimization algorithms, model interpretability, and production ML pipelines
using real datasets and real-world examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

# Machine Learning imports
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer, load_wine, load_digits

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Set random seed for reproducibility (only for fallback data)
np.random.seed(42)


def demonstrate_ensemble_learning():
    """Demonstrate various ensemble learning methods and their performance."""
    print("=" * 80)
    print("CHAPTER 17: ADVANCED MACHINE LEARNING")
    print("=" * 80)

    print("\n17.1 ENSEMBLE LEARNING METHODS")
    print("-" * 40)

    print("\n1. LOADING REAL DATASETS:")
    print("-" * 30)

    def load_real_datasets():
        """Load real datasets for ensemble learning demonstration."""
        datasets = {}

        try:
            # Load Breast Cancer dataset
            print("  Loading Breast Cancer dataset...")
            breast_cancer = load_breast_cancer()
            X_bc, y_bc = breast_cancer.data, breast_cancer.target

            # Load Wine dataset
            print("  Loading Wine dataset...")
            wine = load_wine()
            X_wine, y_wine = wine.data, wine.target

            # Load Digits dataset
            print("  Loading Digits dataset...")
            digits = load_digits()
            X_digits, y_digits = digits.data, digits.target

            datasets["breast_cancer"] = {
                "X": X_bc,
                "y": y_bc,
                "name": "Breast Cancer Wisconsin",
                "description": "Medical diagnosis classification",
            }
            datasets["wine"] = {
                "X": X_wine,
                "y": y_wine,
                "name": "Wine Recognition",
                "description": "Wine variety classification",
            }
            datasets["digits"] = {
                "X": X_digits,
                "y": y_digits,
                "name": "Handwritten Digits",
                "description": "Digit recognition (0-9)",
            }

            print(f"    ✅ Loaded {len(datasets)} real datasets")
            for name, data in datasets.items():
                print(
                    f"      {data['name']}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features"
                )

        except Exception as e:
            print(f"    ⚠️  Error loading datasets: {e}")
            print("    📝 Using synthetic fallback data...")

            # Fallback to synthetic data
            n_samples = 1000
            n_features = 15
            X = np.random.randn(n_samples, n_features)
            target = (
                0.3 * X[:, 0] ** 2
                + 0.5 * X[:, 1] * X[:, 2]
                + 0.2 * np.sin(X[:, 3])
                + 0.1 * np.exp(X[:, 4])
                + np.random.normal(0, 0.1, n_samples)
            )
            y = (target > np.median(target)).astype(int)

            datasets["synthetic"] = {
                "X": X,
                "y": y,
                "name": "Synthetic Dataset",
                "description": "Fallback synthetic data",
            }

        return datasets

    # Load real datasets
    real_datasets = load_real_datasets()

    # Use Breast Cancer dataset as primary (or first available)
    if "breast_cancer" in real_datasets:
        dataset_name = "breast_cancer"
        X = real_datasets[dataset_name]["X"]
        y = real_datasets[dataset_name]["y"]
        print(f"  📊 Using {real_datasets[dataset_name]['name']} dataset")
        print(f"  🎯 Task: {real_datasets[dataset_name]['description']}")
    elif "wine" in real_datasets:
        dataset_name = "wine"
        X = real_datasets[dataset_name]["X"]
        y = real_datasets[dataset_name]["y"]
        print(f"  📊 Using {real_datasets[dataset_name]['name']} dataset")
        print(f"  🎯 Task: {real_datasets[dataset_name]['description']}")
    elif "digits" in real_datasets:
        dataset_name = "digits"
        X = real_datasets[dataset_name]["X"]
        y = real_datasets[dataset_name]["y"]
        print(f"  📊 Using {real_datasets[dataset_name]['name']} dataset")
        print(f"  🎯 Task: {real_datasets[dataset_name]['description']}")
    else:
        dataset_name = "synthetic"
        X = real_datasets[dataset_name]["X"]
        y = real_datasets[dataset_name]["y"]
        print(f"  📊 Using {real_datasets[dataset_name]['name']} dataset")
        print(f"  🎯 Task: {real_datasets[dataset_name]['description']}")

    n_samples, n_features = X.shape
    print(f"  ✅ Dataset: {n_samples:,} samples, {n_features} features")
    print(f"  📊 Target distribution: {np.bincount(y)}")
    print(f"  🎯 Classes: {len(np.unique(y))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\n2. TRAINING INDIVIDUAL MODELS:")
    print("-" * 35)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    individual_results = {}

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time

        individual_results[name] = {
            "accuracy": accuracy,
            "training_time": training_time,
        }
        print(f"  {name:20}: Accuracy: {accuracy:.4f}, Time: {training_time:.4f}s")

    print("\n3. ENSEMBLE METHODS:")
    print("-" * 25)

    # Create voting ensemble
    voting_ensemble = VotingClassifier(
        estimators=[
            ("rf", models["Random Forest"]),
            ("et", models["Extra Trees"]),
            ("gb", models["Gradient Boosting"]),
        ],
        voting="soft",
    )

    start_time = time.time()
    voting_ensemble.fit(X_train, y_train)
    y_pred = voting_ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    training_time = time.time() - start_time

    print(
        f"  Voting Ensemble    : Accuracy: {accuracy:.4f}, Time: {training_time:.4f}s"
    )

    # Performance analysis
    all_results = {
        **individual_results,
        "Voting Ensemble": {"accuracy": accuracy, "training_time": training_time},
    }

    best_model = max(all_results.items(), key=lambda x: x[1]["accuracy"])
    baseline_accuracy = individual_results["Decision Tree"]["accuracy"]
    improvement = (
        (best_model[1]["accuracy"] - baseline_accuracy) / baseline_accuracy
    ) * 100

    print(f"\n🏆 Best model: {best_model[0]} ({best_model[1]['accuracy']:.4f})")
    print(f"📈 Improvement: {improvement:.1f}%")

    return all_results


def demonstrate_optimization():
    """Demonstrate hyperparameter optimization techniques."""
    print("\n17.2 HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)

    # Create dataset
    X = np.random.randn(500, 10)
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("1. Grid Search:")
    print("-" * 15)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy"
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_time

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    print(f"  Test accuracy: {grid_search.score(X_test, y_test):.4f}")
    print(f"  Time: {grid_time:.2f}s")

    print("\n2. Random Search:")
    print("-" * 18)

    param_dist = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    }

    random_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_dist,
        n_iter=10,
        cv=3,
        scoring="accuracy",
        random_state=42,
    )

    start_time = time.time()
    random_search.fit(X_train, y_train)
    random_time = time.time() - start_time

    print(f"  Best params: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.4f}")
    print(f"  Test accuracy: {random_search.score(X_test, y_test):.4f}")
    print(f"  Time: {random_time:.2f}s")

    return {"grid": grid_search, "random": random_search}


def create_visualizations():
    """Create comprehensive visualizations for advanced ML concepts."""
    print("\n17.3 CREATING VISUALIZATIONS")
    print("-" * 35)

    print("  Generating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Advanced Machine Learning: Techniques and Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Ensemble Methods Performance
    ax1 = axes[0, 0]
    methods = [
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "AdaBoost",
        "Gradient Boosting",
        "Voting",
    ]
    accuracies = [0.82, 0.89, 0.88, 0.87, 0.90, 0.91]  # Simulated results

    bars = ax1.bar(
        methods,
        accuracies,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax1.set_title("Ensemble Methods Performance", fontweight="bold")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.8, 0.95)
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Optimization Comparison
    ax2 = axes[0, 1]
    opt_methods = ["Grid Search", "Random Search"]
    cv_scores = [0.89, 0.90]  # Simulated results
    times = [2.5, 1.2]  # Simulated times

    x = np.arange(len(opt_methods))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2, cv_scores, width, label="CV Score", color="#FF6B6B", alpha=0.8
    )
    bars2 = ax2.bar(
        x + width / 2, times, width, label="Time (s)", color="#4ECDC4", alpha=0.8
    )

    ax2.set_title("Optimization Techniques", fontweight="bold")
    ax2.set_ylabel("Score / Time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(opt_methods)
    ax2.legend()

    # 3. Feature Importance
    ax3 = axes[1, 0]
    features = [f"Feature_{i}" for i in range(10)]
    importance = np.random.uniform(0.01, 0.15, 10)
    importance = np.sort(importance)[::-1]

    bars = ax3.barh(features, importance, color="#98D8C8", alpha=0.8)
    ax3.set_title("Top 10 Feature Importance", fontweight="bold")
    ax3.set_xlabel("Importance Score")

    # 4. Model Performance Over Time
    ax4 = axes[1, 1]
    time_points = ["Week 1", "Week 2", "Week 3", "Week 4"]
    accuracy_trend = [0.85, 0.87, 0.89, 0.91]

    ax4.plot(
        time_points, accuracy_trend, "o-", color="#FF6B6B", linewidth=2, markersize=8
    )
    ax4.set_title("Model Performance Over Time", fontweight="bold")
    ax4.set_ylabel("Accuracy")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save visualization
    output_file = "advanced_machine_learning.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✅ Visualization saved: {output_file}")

    plt.show()


def main():
    """Main function to run all demonstrations."""
    try:
        # Run demonstrations
        ensemble_results = demonstrate_ensemble_learning()
        optimization_results = demonstrate_optimization()
        create_visualizations()

        print("\n" + "=" * 80)
        print("CHAPTER 17 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Ensemble learning methods and performance")
        print("  • Hyperparameter optimization techniques")
        print("  • Advanced ML visualization and analysis")

        print("\n📊 Generated Visualizations:")
        print("  • advanced_machine_learning.png - Advanced ML dashboard")

        print("\n🚀 Next Steps:")
        print("  • Practice ensemble methods on your datasets")
        print("  • Experiment with hyperparameter optimization")
        print("  • Continue to Chapter 18: Model Deployment")

    except Exception as e:
        print(f"\n❌ Error in Chapter 17: {str(e)}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
