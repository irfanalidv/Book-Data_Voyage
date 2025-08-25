#!/usr/bin/env python3
"""
Chapter 21: Communication and Storytelling
=========================================

This chapter covers effective communication of data science insights
to diverse audiences, including storytelling techniques, visualization
design, and stakeholder communication strategies.
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

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import StandardScaler

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


class AudienceAnalysis:
    """Demonstrate audience analysis and communication strategy."""

    def __init__(self):
        self.audience_data = None
        self.communication_strategies = None
        self.real_datasets = None

    def load_real_datasets(self):
        """Load real datasets for communication and storytelling examples."""
        print("1. AUDIENCE ANALYSIS AND COMMUNICATION STRATEGY")
        print("=" * 60)

        print("\n1.1 LOADING REAL DATASETS FOR STORYTELLING:")
        print("-" * 50)

        datasets = {}

        try:
            # Load Breast Cancer dataset (medical story)
            print("  Loading Breast Cancer dataset (medical diagnosis story)...")
            breast_cancer = load_breast_cancer()
            X_bc, y_bc = breast_cancer.data, breast_cancer.target
            feature_names = breast_cancer.feature_names

            # Create medical dataset with patient context
            medical_data = pd.DataFrame(X_bc, columns=feature_names)
            medical_data["diagnosis"] = y_bc
            medical_data["patient_id"] = range(1, len(medical_data) + 1)
            medical_data["age_group"] = np.random.choice(
                ["25-35", "36-45", "46-55", "56-65", "65+"], len(medical_data)
            )
            medical_data["region"] = np.random.choice(
                ["Urban", "Suburban", "Rural"], len(medical_data)
            )

            datasets["breast_cancer"] = medical_data
            print(f"    ✅ {breast_cancer.DESCR.split('\\n')[1]}")
            print(f"    📊 Shape: {medical_data.shape}")
            print(
                f"    📖 Story: Medical diagnosis prediction for early cancer detection"
            )

            # Load Wine dataset (quality assessment story)
            print("  Loading Wine dataset (quality assessment story)...")
            wine = load_wine()
            X_wine, y_wine = wine.data, wine.target
            wine_data = pd.DataFrame(X_wine, columns=wine.feature_names)
            wine_data["quality"] = y_wine
            wine_data["region"] = np.random.choice(
                ["France", "Italy", "Spain"], len(wine_data)
            )
            wine_data["price_category"] = np.random.choice(
                ["Budget", "Mid-range", "Premium"], len(wine_data)
            )

            datasets["wine"] = wine_data
            print(f"    ✅ {wine.DESCR.split('\\n')[1]}")
            print(f"    📊 Shape: {wine_data.shape}")
            print(f"    📖 Story: Wine quality prediction for production optimization")

            # Load Digits dataset (image recognition story)
            print("  Loading Digits dataset (handwritten digit recognition story)...")
            digits = load_digits()
            X_digits, y_digits = digits.data, digits.target
            digits_data = pd.DataFrame(
                X_digits, columns=[f"pixel_{i}" for i in range(64)]
            )
            digits_data["digit"] = y_digits
            digits_data["image_id"] = range(1, len(digits_data) + 1)
            digits_data["source"] = np.random.choice(
                ["Handwritten", "Scanned", "Digital"], len(digits_data)
            )

            datasets["digits"] = digits_data
            print(f"    ✅ {digits.DESCR.split('\\n')[1]}")
            print(f"    📊 Shape: {digits_data.shape}")
            print(
                f"    📖 Story: Handwritten digit recognition for document processing"
            )

        except Exception as e:
            print(f"    ⚠️  Error loading datasets: {e}")
            print("    📝 Using synthetic fallback data...")
            datasets = self._create_synthetic_fallback()

        self.real_datasets = datasets
        return datasets

    def _create_synthetic_fallback(self):
        """Create synthetic data as fallback."""
        print("    Creating synthetic fallback datasets...")

        datasets = {}

        # Synthetic medical data
        n_records = 569
        medical_data = pd.DataFrame(
            {
                "patient_id": range(1, n_records + 1),
                "age_group": np.random.choice(
                    ["25-35", "36-45", "46-55", "56-65", "65+"], n_records
                ),
                "region": np.random.choice(["Urban", "Suburban", "Rural"], n_records),
                "diagnosis": np.random.choice([0, 1], n_records, p=[0.37, 0.63]),
                "feature_1": np.random.randn(n_records),
                "feature_2": np.random.randn(n_records),
                "feature_3": np.random.randn(n_records),
            }
        )
        datasets["breast_cancer"] = medical_data

        # Synthetic wine data
        n_records = 178
        wine_data = pd.DataFrame(
            {
                "quality": np.random.choice([0, 1, 2], n_records, p=[0.33, 0.40, 0.27]),
                "region": np.random.choice(["France", "Italy", "Spain"], n_records),
                "price_category": np.random.choice(
                    ["Budget", "Mid-range", "Premium"], n_records
                ),
                "feature_1": np.random.randn(n_records),
                "feature_2": np.random.randn(n_records),
            }
        )
        datasets["wine"] = wine_data

        # Synthetic digits data
        n_records = 1797
        digits_data = pd.DataFrame(
            {
                "digit": np.random.choice(range(10), n_records),
                "image_id": range(1, n_records + 1),
                "source": np.random.choice(
                    ["Handwritten", "Scanned", "Digital"], n_records
                ),
                "feature_1": np.random.randn(n_records),
                "feature_2": np.random.randn(n_records),
            }
        )
        datasets["digits"] = digits_data

        return datasets

    def create_audience_dataset(self):
        """Create audience analysis dataset for communication strategy."""
        # Load real datasets first
        self.load_real_datasets()

        # Create audience analysis dataset
        print("\n1.2 CREATING AUDIENCE ANALYSIS DATASET:")
        print("-" * 45)

        # Generate realistic audience data based on real-world scenarios
        n_audiences = 200

        audience_data = {
            "audience_id": range(1, n_audiences + 1),
            "audience_type": np.random.choice(
                ["Executive", "Technical", "Operational", "Client", "Public"],
                n_audiences,
                p=[0.2, 0.3, 0.25, 0.15, 0.1],
            ),
            "technical_expertise": np.random.choice(
                ["Low", "Medium", "High"], n_audiences, p=[0.4, 0.4, 0.2]
            ),
            "decision_making_power": np.random.choice(
                ["Low", "Medium", "High"], n_audiences, p=[0.5, 0.3, 0.2]
            ),
            "attention_span": np.random.choice(
                ["Short", "Medium", "Long"], n_audiences, p=[0.6, 0.3, 0.1]
            ),
            "preferred_format": np.random.choice(
                ["Visual", "Narrative", "Technical", "Executive Summary"],
                n_audiences,
                p=[0.3, 0.25, 0.25, 0.2],
            ),
            "communication_frequency": np.random.choice(
                ["Daily", "Weekly", "Monthly", "Quarterly"],
                n_audiences,
                p=[0.2, 0.4, 0.3, 0.1],
            ),
        }

        self.audience_data = pd.DataFrame(audience_data)

        print(
            f"  ✅ Audience dataset created: {len(self.audience_data):,} audience members"
        )
        print(f"  🔍 Audience type distribution:")
        print(self.audience_data["audience_type"].value_counts())
        print(f"  📊 Technical expertise distribution:")
        print(self.audience_data["technical_expertise"].value_counts())

        return self.audience_data

    def analyze_communication_needs(self):
        """Analyze communication needs by audience type."""
        print("\n1.2 COMMUNICATION NEEDS ANALYSIS:")
        print("-" * 40)

        # Analyze communication preferences by audience type
        print("  🔍 Communication Preferences by Audience Type:")
        print("    " + "=" * 60)

        for audience_type in self.audience_data["audience_type"].unique():
            type_data = self.audience_data[
                self.audience_data["audience_type"] == audience_type
            ]

            print(f"\n    📊 {audience_type} Audience ({len(type_data)} members):")

            # Technical expertise
            tech_expertise = type_data["technical_expertise"].value_counts()
            print(f"      Technical Expertise:")
            for level, count in tech_expertise.items():
                pct = (count / len(type_data)) * 100
                print(f"        {level}: {count} ({pct:.1f}%)")

            # Preferred format
            preferred_format = type_data["preferred_format"].value_counts()
            print(f"      Preferred Format:")
            for format_type, count in preferred_format.items():
                pct = (count / len(type_data)) * 100
                print(f"        {format_type}: {count} ({pct:.1f}%)")

            # Decision making power
            decision_power = type_data["decision_making_power"].value_counts()
            print(f"      Decision Making Power:")
            for power, count in decision_power.items():
                pct = (count / len(type_data)) * 100
                print(f"        {power}: {count} ({pct:.1f}%)")

        return self.audience_data


class DataStorytelling:
    """Demonstrate data storytelling techniques and frameworks."""

    def __init__(self):
        self.story_data = None
        self.story_structure = None

    def create_story_dataset(self):
        """Create a synthetic dataset for storytelling demonstration."""
        print("\n2. DATA STORYTELLING FUNDAMENTALS")
        print("=" * 50)

        print("\n2.1 CREATING STORYTELLING DATASET:")
        print("-" * 40)

        # Generate synthetic business data for storytelling
        n_periods = 24  # 2 years of monthly data

        # Create time series data
        dates = pd.date_range("2022-01-01", periods=n_periods, freq="M")

        # Generate business metrics with trends and seasonality
        base_sales = 10000
        trend = np.linspace(0, 5000, n_periods)  # Upward trend
        seasonality = 2000 * np.sin(
            2 * np.pi * np.arange(n_periods) / 12
        )  # Annual seasonality
        noise = np.random.normal(0, 500, n_periods)

        sales = base_sales + trend + seasonality + noise
        sales = np.maximum(sales, 5000)  # Minimum sales

        # Generate related metrics
        marketing_spend = sales * 0.15 + np.random.normal(0, 200, n_periods)
        customer_satisfaction = (
            4.0 + 0.0001 * sales + np.random.normal(0, 0.2, n_periods)
        )
        customer_satisfaction = np.clip(customer_satisfaction, 1.0, 5.0)

        # Create story dataset
        story_data = pd.DataFrame(
            {
                "date": dates,
                "month": dates.month_name(),
                "year": dates.year,
                "sales": sales,
                "marketing_spend": marketing_spend,
                "customer_satisfaction": customer_satisfaction,
                "roi": (sales - marketing_spend) / marketing_spend,
            }
        )

        self.story_data = story_data

        print(
            f"  ✅ Storytelling dataset created: {len(self.story_data):,} months of data"
        )
        print(
            f"  📊 Data spans: {self.story_data['date'].min()} to {self.story_data['date'].max()}"
        )
        print(f"  📈 Key metrics: Sales, Marketing Spend, Customer Satisfaction, ROI")

        return self.story_data

    def demonstrate_story_structure(self):
        """Demonstrate different storytelling structures."""
        print("\n2.2 STORY STRUCTURE ANALYSIS:")
        print("-" * 35)

        # Calculate key story elements
        total_sales = self.story_data["sales"].sum()
        avg_sales = self.story_data["sales"].mean()
        sales_growth = (
            (self.story_data["sales"].iloc[-1] - self.story_data["sales"].iloc[0])
            / self.story_data["sales"].iloc[0]
            * 100
        )

        best_month = self.story_data.loc[self.story_data["sales"].idxmax()]
        worst_month = self.story_data.loc[self.story_data["sales"].idxmin()]

        print("  📖 Story Structure Elements:")
        print(f"\n    🎯 Problem (Challenge):")
        print(
            f"      Starting point: ${self.story_data['sales'].iloc[0]:,.0f} in sales"
        )
        print(f"      Market challenges and competition")

        print(f"\n    🔍 Analysis (Journey):")
        print(f"      Data analysis over {len(self.story_data)} months")
        print(f"      Pattern identification and insights")
        print(
            f"      Best month: {best_month['month']} {best_month['year']} (${best_month['sales']:,.0f})"
        )
        print(
            f"      Worst month: {worst_month['month']} {worst_month['year']} (${worst_month['sales']:,.0f})"
        )

        print(f"\n    💡 Solution (Actions):")
        print(f"      Marketing optimization and strategy refinement")
        print(f"      Customer experience improvements")
        print(f"      Data-driven decision making")

        print(f"\n    🚀 Impact (Results):")
        print(f"      Total sales: ${total_sales:,.0f}")
        print(f"      Average monthly sales: ${avg_sales:,.0f}")
        print(f"      Growth: {sales_growth:+.1f}% over the period")

        print(f"\n    📊 Call to Action:")
        print(f"      Continue data-driven optimization")
        print(f"      Scale successful strategies")
        print(f"      Invest in customer experience")

        return {
            "total_sales": total_sales,
            "avg_sales": avg_sales,
            "sales_growth": sales_growth,
            "best_month": best_month,
            "worst_month": worst_month,
        }


def create_communication_visualizations():
    """Create comprehensive visualizations for communication concepts."""
    print("\n3. CREATING COMMUNICATION VISUALIZATIONS:")
    print("-" * 45)

    print("  Generating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Communication and Storytelling: Effective Data Science Communication",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Audience Type Distribution
    ax1 = axes[0, 0]
    audience_types = ["Executive", "Technical", "Operational", "Client", "Public"]
    audience_counts = [40, 60, 50, 30, 20]  # Simulated counts

    bars = ax1.pie(
        audience_counts, labels=audience_types, autopct="%1.1f%%", startangle=90
    )
    ax1.set_title("Audience Type Distribution", fontweight="bold")

    # 2. Communication Preferences by Audience
    ax2 = axes[0, 1]
    preferences = ["Visual", "Narrative", "Technical", "Executive Summary"]
    executive_prefs = [0.4, 0.3, 0.1, 0.2]  # Executive preferences
    technical_prefs = [0.2, 0.2, 0.5, 0.1]  # Technical preferences

    x = np.arange(len(preferences))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        executive_prefs,
        width,
        label="Executive",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        x + width / 2,
        technical_prefs,
        width,
        label="Technical",
        color="#4ECDC4",
        alpha=0.8,
    )

    ax2.set_title("Communication Preferences by Audience", fontweight="bold")
    ax2.set_ylabel("Preference Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(preferences, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 0.6)

    # 3. Story Structure Elements
    ax3 = axes[0, 2]
    story_elements = ["Problem", "Analysis", "Solution", "Impact", "Action"]
    importance_scores = [0.9, 0.8, 0.85, 0.95, 0.9]  # Importance scores

    bars = ax3.bar(
        story_elements,
        importance_scores,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"],
        alpha=0.8,
    )
    ax3.set_title("Story Structure Elements Importance", fontweight="bold")
    ax3.set_ylabel("Importance Score")
    ax3.set_ylim(0, 1)

    # Add value labels
    for bar, score in zip(bars, importance_scores):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Chart Type Selection Guide
    ax4 = axes[1, 0]
    chart_types = ["Bar", "Line", "Pie", "Scatter", "Heatmap"]
    effectiveness_scores = [0.9, 0.85, 0.7, 0.8, 0.75]  # Effectiveness scores

    bars = ax4.barh(
        chart_types,
        effectiveness_scores,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"],
        alpha=0.8,
    )
    ax4.set_title("Chart Type Effectiveness", fontweight="bold")
    ax4.set_xlabel("Effectiveness Score")
    ax4.set_xlim(0, 1)

    # Add value labels
    for bar, score in zip(bars, effectiveness_scores):
        width = bar.get_width()
        ax4.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # 5. Presentation Effectiveness by Duration
    ax5 = axes[1, 1]
    durations = ["15 min", "30 min", "45 min", "60 min"]
    effectiveness_scores = [0.85, 0.90, 0.75, 0.65]  # Effectiveness by duration

    bars = ax5.bar(
        durations,
        effectiveness_scores,
        color=["#98D8C8", "#4ECDC4", "#FFA07A", "#FF6B6B"],
        alpha=0.8,
    )
    ax5.set_title("Presentation Effectiveness by Duration", fontweight="bold")
    ax5.set_ylabel("Effectiveness Score")
    ax5.set_ylim(0, 1)

    # Add value labels
    for bar, score in zip(bars, effectiveness_scores):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 6. Communication Channel Effectiveness
    ax6 = axes[1, 2]
    channels = ["Email", "Presentation", "Report", "Dashboard", "Workshop"]
    effectiveness = [0.6, 0.85, 0.75, 0.8, 0.9]  # Channel effectiveness

    bars = ax6.barh(
        channels,
        effectiveness,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"],
        alpha=0.8,
    )
    ax6.set_title("Communication Channel Effectiveness", fontweight="bold")
    ax6.set_xlabel("Effectiveness Score")
    ax6.set_xlim(0, 1)

    # Add value labels
    for bar, score in zip(bars, effectiveness):
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
    output_file = "communication_storytelling.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✅ Visualization saved: {output_file}")

    plt.show()


def main():
    """Main function to run communication demonstrations."""
    try:
        print("=" * 80)
        print("CHAPTER 21: COMMUNICATION AND STORYTELLING")
        print("=" * 80)

        # Initialize communication demonstrations
        audience = AudienceAnalysis()
        storytelling = DataStorytelling()

        # Run audience analysis demonstrations
        print("\n" + "=" * 80)
        audience_data = audience.create_audience_dataset()
        audience_analysis = audience.analyze_communication_needs()

        # Run storytelling demonstrations
        print("\n" + "=" * 80)
        story_data = storytelling.create_story_dataset()
        story_structure = storytelling.demonstrate_story_structure()

        # Create visualizations
        create_communication_visualizations()

        print("\n" + "=" * 80)
        print("CHAPTER 21 - COMMUNICATION AND STORYTELLING COMPLETED!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Audience analysis and communication strategy development")
        print("  • Data storytelling frameworks and narrative structures")
        print("  • Visualization design principles and chart selection")
        print("  • Communication effectiveness and channel optimization")

        print("\n📊 Generated Visualizations:")
        print(
            "  • communication_storytelling.png - Comprehensive communication dashboard"
        )

        print("\n🚀 Next Steps:")
        print("  • Apply communication techniques to your data science projects")
        print("  • Practice storytelling with different audience types")
        print("  • Continue to Chapter 22: Portfolio Development")

    except Exception as e:
        print(f"\n❌ Error in Chapter 21: {str(e)}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
