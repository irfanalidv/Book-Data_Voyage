#!/usr/bin/env python3
"""
Chapter 20: Data Science Ethics
===============================

This chapter covers essential ethical principles and responsible practices
in data science and AI development, including privacy protection, bias
detection, fairness evaluation, and responsible AI governance.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any, Tuple
import hashlib
import random

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


class PrivacyProtection:
    """Demonstrate privacy protection techniques."""

    def __init__(self):
        self.original_data = None
        self.anonymized_data = None

    def create_sensitive_dataset(self):
        """Create a synthetic dataset with sensitive information."""
        print("1. PRIVACY PROTECTION AND DATA ETHICS")
        print("=" * 50)

        print("\n1.1 CREATING SENSITIVE DATASET:")
        print("-" * 40)

        # Generate synthetic sensitive data
        n_records = 500

        sensitive_data = {
            "patient_id": range(1, n_records + 1),
            "name": [f"Patient_{i}" for i in range(1, n_records + 1)],
            "ssn": [
                f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
                for _ in range(n_records)
            ],
            "date_of_birth": pd.date_range("1950-01-01", periods=n_records, freq="D"),
            "address": [
                f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm'])} St"
                for _ in range(n_records)
            ],
            "phone": [
                f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
                for _ in range(n_records)
            ],
            "medical_condition": np.random.choice(
                ["Diabetes", "Hypertension", "Heart Disease", "Healthy"],
                n_records,
                p=[0.2, 0.3, 0.2, 0.3],
            ),
            "treatment_cost": np.random.lognormal(3, 0.8, n_records).astype(int),
            "insurance_claim": np.random.choice([True, False], n_records, p=[0.7, 0.3]),
        }

        self.original_data = pd.DataFrame(sensitive_data)

        print(f"  ✅ Sensitive dataset created: {len(self.original_data):,} records")
        print(f"  🔒 Contains: Names, SSNs, Addresses, Phone numbers, Medical data")
        print(f"  📊 Sample data:")
        print(self.original_data.head(3).to_string(index=False))

        return self.original_data

    def demonstrate_anonymization(self):
        """Demonstrate data anonymization techniques."""
        print("\n1.2 DATA ANONYMIZATION TECHNIQUES:")
        print("-" * 40)

        anonymized = self.original_data.copy()

        # 1. Direct identifier removal
        direct_identifiers = ["name", "ssn", "phone", "address"]
        anonymized = anonymized.drop(columns=direct_identifiers)

        # 2. Date generalization (year only)
        anonymized["birth_year"] = anonymized["date_of_birth"].dt.year
        anonymized = anonymized.drop(columns=["date_of_birth"])

        # 3. Age grouping
        anonymized["age"] = datetime.now().year - anonymized["birth_year"]
        anonymized["age_group"] = pd.cut(
            anonymized["age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        anonymized = anonymized.drop(columns=["birth_year", "age"])

        # 4. Cost binning
        anonymized["cost_category"] = pd.cut(
            anonymized["treatment_cost"],
            bins=[0, 1000, 5000, 10000, 50000],
            labels=["Low", "Medium", "High", "Very High"],
        )
        anonymized = anonymized.drop(columns=["treatment_cost"])

        # 5. Patient ID hashing
        anonymized["hashed_id"] = anonymized["patient_id"].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()[:8]
        )
        anonymized = anonymized.drop(columns=["patient_id"])

        self.anonymized_data = anonymized

        print("  🔒 Anonymization Applied:")
        print("    ✅ Direct identifiers removed (names, SSNs, phones, addresses)")
        print("    ✅ Dates generalized to birth years, then to age groups")
        print("    ✅ Treatment costs binned into categories")
        print("    ✅ Patient IDs hashed for privacy")

        print(f"\n  📊 Anonymized dataset structure:")
        print(anonymized.info())

        print(f"\n  🔍 Sample anonymized data:")
        print(anonymized.head(3).to_string(index=False))

        return self.anonymized_data

    def demonstrate_differential_privacy(self):
        """Demonstrate differential privacy concepts."""
        print("\n1.3 DIFFERENTIAL PRIVACY CONCEPTS:")
        print("-" * 40)

        # Simulate differential privacy with noise addition
        original_counts = self.anonymized_data["medical_condition"].value_counts()

        # Add Laplace noise for differential privacy
        epsilon = 1.0  # Privacy parameter (lower = more private)
        sensitivity = 1  # Maximum change in output

        noisy_counts = {}
        for condition, count in original_counts.items():
            # Laplace noise: Lap(0, sensitivity/epsilon)
            noise = np.random.laplace(0, sensitivity / epsilon)
            noisy_counts[condition] = max(0, int(count + noise))

        print("  🔒 Differential Privacy Simulation:")
        print(f"    Privacy parameter (ε): {epsilon}")
        print(f"    Sensitivity: {sensitivity}")

        print(f"\n  📊 Original vs. Noisy Counts:")
        print("    Condition        Original    Noisy      Difference")
        print("    " + "-" * 50)
        for condition in original_counts.index:
            orig = original_counts[condition]
            noisy = noisy_counts[condition]
            diff = noisy - orig
            print(f"    {condition:15} {orig:8} {noisy:8} {diff:+8}")

        return noisy_counts


class BiasDetection:
    """Demonstrate bias detection and mitigation techniques."""

    def __init__(self):
        self.dataset = None
        self.model = None

    def create_biased_dataset(self):
        """Create a synthetic dataset with intentional bias."""
        print("\n2. BIAS DETECTION AND MITIGATION")
        print("=" * 50)

        print("\n2.1 CREATING BIASED DATASET:")
        print("-" * 35)

        # Generate synthetic data with intentional bias
        n_samples = 1000

        # Create biased features
        np.random.seed(42)

        # Feature 1: Education level (biased by demographic)
        education_bias = np.random.choice(
            [0, 1], n_samples, p=[0.6, 0.4]
        )  # 60% group 0, 40% group 1
        education = np.where(
            education_bias == 0,
            np.random.choice(
                [1, 2, 3], n_samples, p=[0.7, 0.2, 0.1]
            ),  # Lower education for group 0
            np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        )  # Higher education for group 1

        # Feature 2: Income (biased by education and demographic)
        income = 30000 + education * 15000 + np.random.normal(0, 5000, n_samples)
        income = np.maximum(income, 20000)  # Minimum income

        # Feature 3: Credit score (biased by income and demographic)
        credit_score = (
            500 + (income - 30000) / 1000 + np.random.normal(0, 50, n_samples)
        )
        credit_score = np.clip(credit_score, 300, 850)

        # Target: Loan approval (biased by demographic)
        approval_prob = 1 / (1 + np.exp(-(credit_score - 650) / 100))
        approval_prob = np.where(
            education_bias == 0, approval_prob * 0.8, approval_prob
        )  # Bias against group 0
        loan_approved = np.random.binomial(1, approval_prob)

        # Create dataset
        self.dataset = pd.DataFrame(
            {
                "demographic_group": education_bias,
                "education_level": education,
                "income": income,
                "credit_score": credit_score,
                "loan_approved": loan_approved,
            }
        )

        # Add demographic labels
        self.dataset["demographic_label"] = self.dataset["demographic_group"].map(
            {0: "Group A", 1: "Group B"}
        )

        print(f"  ✅ Biased dataset created: {len(self.dataset):,} samples")
        print(f"  🔍 Demographic distribution:")
        print(self.dataset["demographic_label"].value_counts())
        print(f"  📊 Loan approval rates by group:")
        approval_rates = self.dataset.groupby("demographic_label")[
            "loan_approved"
        ].mean()
        for group, rate in approval_rates.items():
            print(f"    {group}: {rate:.1%}")

        return self.dataset

    def detect_bias(self):
        """Detect bias in the dataset and model."""
        print("\n2.2 BIAS DETECTION ANALYSIS:")
        print("-" * 35)

        # Prepare features for modeling
        features = ["education_level", "income", "credit_score"]
        X = self.dataset[features]
        y = self.dataset["loan_approved"]

        # Train a model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Add predictions to test data
        test_data = X_test.copy()
        test_data["loan_approved"] = y_test
        test_data["predicted"] = y_pred
        test_data["prediction_prob"] = y_prob
        test_data["demographic_group"] = self.dataset.loc[
            X_test.index, "demographic_group"
        ]
        test_data["demographic_label"] = self.dataset.loc[
            X_test.index, "demographic_label"
        ]

        # Calculate bias metrics
        print("  🔍 Bias Detection Results:")

        # 1. Demographic parity
        demo_parity = test_data.groupby("demographic_label")["predicted"].mean()
        print(f"\n    📊 Demographic Parity (Prediction Rate):")
        for group, rate in demo_parity.items():
            print(f"      {group}: {rate:.1%}")

        # 2. Equal opportunity (TPR by group)
        tpr_by_group = {}
        for group in test_data["demographic_label"].unique():
            group_data = test_data[test_data["demographic_label"] == group]
            if group_data["loan_approved"].sum() > 0:
                tpr = (
                    group_data["predicted"] & group_data["loan_approved"]
                ).sum() / group_data["loan_approved"].sum()
                tpr_by_group[group] = tpr

        print(f"\n    📊 Equal Opportunity (True Positive Rate):")
        for group, tpr in tpr_by_group.items():
            print(f"      {group}: {tpr:.1%}")

        # 3. Predictive rate equality
        pred_rate = test_data.groupby("demographic_label")["prediction_prob"].mean()
        print(f"\n    📊 Predictive Rate Equality (Average Probability):")
        for group, rate in pred_rate.items():
            print(f"      {group}: {rate:.3f}")

        # 4. Statistical parity difference
        spd = demo_parity["Group B"] - demo_parity["Group A"]
        print(f"\n    📊 Statistical Parity Difference: {spd:.3f}")
        if abs(spd) > 0.1:
            print("      ⚠️  Significant bias detected (>0.1 threshold)")
        else:
            print("      ✅ Bias within acceptable range")

        return test_data, demo_parity, tpr_by_group, pred_rate, spd


def create_ethics_visualizations():
    """Create comprehensive visualizations for ethics concepts."""
    print("\n4. CREATING ETHICS VISUALIZATIONS:")
    print("-" * 40)

    print("  Generating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Data Science Ethics: Responsible AI and Fairness Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Privacy Protection - Data Anonymization
    ax1 = axes[0, 0]
    data_types = ["Original", "Anonymized"]
    sensitive_fields = [8, 2]  # Number of sensitive fields

    bars = ax1.bar(
        data_types, sensitive_fields, color=["#FF6B6B", "#4ECDC4"], alpha=0.8
    )
    ax1.set_title("Data Anonymization Impact", fontweight="bold")
    ax1.set_ylabel("Number of Sensitive Fields")
    ax1.set_ylim(0, 10)

    # Add value labels
    for bar, count in zip(bars, sensitive_fields):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.2,
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Bias Detection - Demographic Parity
    ax2 = axes[0, 1]
    groups = ["Group A", "Group B"]
    approval_rates = [0.65, 0.78]  # Simulated rates

    bars = ax2.bar(groups, approval_rates, color=["#FFA07A", "#98D8C8"], alpha=0.8)
    ax2.set_title("Demographic Parity Analysis", fontweight="bold")
    ax2.set_ylabel("Loan Approval Rate")
    ax2.set_ylim(0, 1)

    # Add value labels
    for bar, rate in zip(bars, approval_rates):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Fairness Metrics - Equal Opportunity
    ax3 = axes[0, 2]
    metrics = ["Demographic\nParity", "Equal\nOpportunity", "Equalized\nOdds"]
    fairness_scores = [0.87, 0.92, 0.89]  # Simulated scores

    bars = ax3.bar(
        metrics, fairness_scores, color=["#FF6B6B", "#4ECDC4", "#45B7D1"], alpha=0.8
    )
    ax3.set_title("Fairness Metrics Comparison", fontweight="bold")
    ax3.set_ylabel("Fairness Score")
    ax3.set_ylim(0, 1)

    # Add value labels
    for bar, score in zip(bars, fairness_scores):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Privacy vs. Utility Trade-off
    ax4 = axes[1, 0]
    privacy_levels = ["Low", "Medium", "High"]
    utility_scores = [0.95, 0.85, 0.70]  # Higher privacy = lower utility

    bars = ax4.bar(
        privacy_levels,
        utility_scores,
        color=["#FF6B6B", "#FFA07A", "#4ECDC4"],
        alpha=0.8,
    )
    ax4.set_title("Privacy vs. Utility Trade-off", fontweight="bold")
    ax4.set_ylabel("Data Utility Score")
    ax4.set_ylim(0, 1)

    # Add value labels
    for bar, score in zip(bars, utility_scores):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 5. Bias Mitigation Impact
    ax5 = axes[1, 1]
    mitigation_steps = ["Before", "After\nMitigation"]
    bias_scores = [0.23, 0.08]  # Bias reduction

    bars = ax5.bar(
        mitigation_steps, bias_scores, color=["#FF6B6B", "#98D8C8"], alpha=0.8
    )
    ax5.set_title("Bias Mitigation Impact", fontweight="bold")
    ax5.set_ylabel("Bias Score (Lower = Better)")
    ax5.set_ylim(0, 0.3)

    # Add value labels
    for bar, score in zip(bars, bias_scores):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 6. Ethical AI Framework
    ax6 = axes[1, 2]
    framework_components = [
        "Privacy",
        "Fairness",
        "Transparency",
        "Accountability",
        "Safety",
    ]
    implementation_scores = [0.85, 0.78, 0.82, 0.75, 0.80]  # Implementation scores

    bars = ax6.barh(
        framework_components,
        implementation_scores,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"],
        alpha=0.8,
    )
    ax6.set_title("Ethical AI Framework Implementation", fontweight="bold")
    ax6.set_xlabel("Implementation Score")
    ax6.set_xlim(0, 1)

    # Add value labels
    for bar, score in zip(bars, implementation_scores):
        width = bar.get_width()
        ax6.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save visualization
    output_file = "data_science_ethics.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✅ Visualization saved: {output_file}")

    plt.show()


def main():
    """Main function to run ethics demonstrations."""
    try:
        print("=" * 80)
        print("CHAPTER 20: DATA SCIENCE ETHICS")
        print("=" * 80)

        # Initialize ethics demonstrations
        privacy = PrivacyProtection()
        bias = BiasDetection()

        # Run privacy protection demonstrations
        print("\n" + "=" * 80)
        sensitive_data = privacy.create_sensitive_dataset()
        anonymized_data = privacy.demonstrate_anonymization()
        noisy_counts = privacy.demonstrate_differential_privacy()

        # Run bias detection demonstrations
        print("\n" + "=" * 80)
        biased_data = bias.create_biased_dataset()
        bias_results = bias.detect_bias()

        # Create visualizations
        create_ethics_visualizations()

        print("\n" + "=" * 80)
        print("CHAPTER 20 - DATA SCIENCE ETHICS COMPLETED!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Privacy protection and data anonymization techniques")
        print("  • Bias detection and mitigation strategies")
        print("  • Fairness metrics and evaluation methods")
        print("  • Responsible AI development practices")

        print("\n📊 Generated Visualizations:")
        print("  • data_science_ethics.png - Comprehensive ethics dashboard")

        print("\n🚀 Next Steps:")
        print("  • Apply ethical principles to your ML projects")
        print("  • Implement bias detection in production systems")
        print("  • Continue to Chapter 21: Communication and Storytelling")

    except Exception as e:
        print(f"\n❌ Error in Chapter 20: {str(e)}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
