#!/usr/bin/env python3
"""
Chapter 22: Portfolio Development
================================

This chapter covers creating showcase data science projects that
demonstrate technical expertise and business impact, including
project design, implementation, optimization, and career application.
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


class PortfolioProject:
    """Demonstrate portfolio project development and structure."""

    def __init__(self):
        self.project_data = None
        self.project_metrics = None

    def create_portfolio_dataset(self):
        """Create a synthetic dataset for portfolio project demonstration."""
        print("1. PORTFOLIO PROJECT DESIGN AND STRATEGY")
        print("=" * 60)

        print("\n1.1 CREATING PORTFOLIO PROJECT DATASET:")
        print("-" * 45)

        # Generate synthetic portfolio project data
        n_projects = 50

        # Project types and domains
        project_types = [
            "ML Prediction",
            "Data Visualization",
            "NLP Analysis",
            "Computer Vision",
            "Time Series",
            "Clustering",
            "Recommendation",
            "Anomaly Detection",
        ]
        domains = [
            "E-commerce",
            "Healthcare",
            "Finance",
            "Marketing",
            "Transportation",
            "Education",
            "Entertainment",
            "Manufacturing",
        ]
        complexity_levels = ["Beginner", "Intermediate", "Advanced", "Expert"]

        portfolio_data = {
            "project_id": range(1, n_projects + 1),
            "project_name": [f"Project_{i}" for i in range(1, n_projects + 1)],
            "project_type": np.random.choice(project_types, n_projects),
            "domain": np.random.choice(domains, n_projects),
            "complexity": np.random.choice(
                complexity_levels, n_projects, p=[0.3, 0.4, 0.2, 0.1]
            ),
            "development_hours": np.random.exponential(40, n_projects).astype(int),
            "lines_of_code": np.random.lognormal(8, 1.2, n_projects).astype(int),
            "github_stars": np.random.poisson(15, n_projects),
            "business_impact_score": np.random.uniform(0.6, 1.0, n_projects),
            "technical_skill_score": np.random.uniform(0.7, 1.0, n_projects),
            "documentation_quality": np.random.uniform(0.5, 1.0, n_projects),
            "deployment_status": np.random.choice(
                ["Local", "Cloud", "Web App", "API"], n_projects, p=[0.2, 0.3, 0.3, 0.2]
            ),
        }

        self.project_data = pd.DataFrame(portfolio_data)

        print(f"  ✅ Portfolio dataset created: {len(self.project_data):,} projects")
        print(f"  🔍 Project type distribution:")
        print(self.project_data["project_type"].value_counts())
        print(f"  📊 Domain coverage:")
        print(self.project_data["domain"].value_counts())

        return self.project_data

    def analyze_portfolio_strategy(self):
        """Analyze portfolio strategy and optimization opportunities."""
        print("\n1.2 PORTFOLIO STRATEGY ANALYSIS:")
        print("-" * 40)

        print("  🔍 Portfolio Strategy Insights:")
        print("    " + "=" * 50)

        # Analyze project diversity
        print(f"\n    📊 Project Diversity Analysis:")
        print(
            f"      Project Types: {self.project_data['project_type'].nunique()}/8 covered"
        )
        print(f"      Domains: {self.project_data['domain'].nunique()}/8 covered")
        print(
            f"      Complexity Levels: {self.project_data['complexity'].nunique()}/4 covered"
        )

        # Analyze skill coverage
        print(f"\n    🎯 Skill Coverage Analysis:")
        avg_technical = self.project_data["technical_skill_score"].mean()
        avg_business = self.project_data["business_impact_score"].mean()
        avg_documentation = self.project_data["documentation_quality"].mean()

        print(f"      Average Technical Skill Score: {avg_technical:.3f}")
        print(f"      Average Business Impact Score: {avg_business:.3f}")
        print(f"      Average Documentation Quality: {avg_documentation:.3f}")

        # Identify gaps and opportunities
        print(f"\n    💡 Portfolio Optimization Opportunities:")

        # Find underrepresented project types
        type_counts = self.project_data["project_type"].value_counts()
        underrepresented = type_counts[type_counts < type_counts.mean()]
        if len(underrepresented) > 0:
            print(
                f"      Underrepresented Project Types: {', '.join(underrepresented.index)}"
            )

        # Find underrepresented domains
        domain_counts = self.project_data["domain"].value_counts()
        underrepresented_domains = domain_counts[domain_counts < domain_counts.mean()]
        if len(underrepresented_domains) > 0:
            print(
                f"      Underrepresented Domains: {', '.join(underrepresented_domains.index)}"
            )

        # Complexity distribution
        complexity_dist = self.project_data["complexity"].value_counts()
        print(f"      Complexity Distribution:")
        for level, count in complexity_dist.items():
            pct = (count / len(self.project_data)) * 100
            print(f"        {level}: {count} projects ({pct:.1f}%)")

        return {
            "avg_technical": avg_technical,
            "avg_business": avg_business,
            "avg_documentation": avg_documentation,
            "type_coverage": self.project_data["project_type"].nunique(),
            "domain_coverage": self.project_data["domain"].nunique(),
        }


class TechnicalImplementation:
    """Demonstrate technical implementation and architecture."""

    def __init__(self):
        self.implementation_data = None

    def create_implementation_dataset(self):
        """Create a dataset for technical implementation analysis."""
        print("\n2. TECHNICAL IMPLEMENTATION AND ARCHITECTURE")
        print("=" * 60)

        print("\n2.1 CREATING IMPLEMENTATION DATASET:")
        print("-" * 45)

        # Generate synthetic implementation data
        n_implementations = 40

        # Technology stacks and frameworks
        languages = ["Python", "R", "SQL", "JavaScript", "Scala", "Java"]
        ml_frameworks = [
            "scikit-learn",
            "TensorFlow",
            "PyTorch",
            "XGBoost",
            "LightGBM",
            "Keras",
        ]
        viz_tools = ["Matplotlib", "Seaborn", "Plotly", "Tableau", "PowerBI", "D3.js"]
        deployment_platforms = ["AWS", "Azure", "GCP", "Heroku", "Docker", "Kubernetes"]

        implementation_data = {
            "implementation_id": range(1, n_implementations + 1),
            "project_id": np.random.choice(range(1, 51), n_implementations),
            "primary_language": np.random.choice(
                languages, n_implementations, p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05]
            ),
            "ml_framework": np.random.choice(
                ml_frameworks, n_implementations, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
            ),
            "visualization_tool": np.random.choice(
                viz_tools, n_implementations, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
            ),
            "deployment_platform": np.random.choice(
                deployment_platforms,
                n_implementations,
                p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05],
            ),
            "code_quality_score": np.random.uniform(0.6, 1.0, n_implementations),
            "performance_score": np.random.uniform(0.5, 1.0, n_implementations),
            "scalability_score": np.random.uniform(0.4, 1.0, n_implementations),
            "maintainability_score": np.random.uniform(0.5, 1.0, n_implementations),
            "testing_coverage": np.random.uniform(0.3, 1.0, n_implementations),
            "documentation_score": np.random.uniform(0.4, 1.0, n_implementations),
        }

        self.implementation_data = pd.DataFrame(implementation_data)

        print(
            f"  ✅ Implementation dataset created: {len(self.implementation_data):,} implementations"
        )
        print(f"  🔍 Technology stack distribution:")
        print(
            f"    Primary Languages: {self.implementation_data['primary_language'].value_counts().head(3).to_dict()}"
        )
        print(
            f"    ML Frameworks: {self.implementation_data['ml_framework'].value_counts().head(3).to_dict()}"
        )

        return self.implementation_data

    def analyze_technical_quality(self):
        """Analyze technical implementation quality metrics."""
        print("\n2.2 TECHNICAL QUALITY ANALYSIS:")
        print("-" * 40)

        print("  🔍 Technical Quality Metrics:")
        print("    " + "=" * 40)

        # Calculate average scores by technology
        print(f"\n    📊 Quality Scores by Primary Language:")
        lang_quality = self.implementation_data.groupby("primary_language")[
            [
                "code_quality_score",
                "performance_score",
                "scalability_score",
                "maintainability_score",
            ]
        ].mean()

        for language in lang_quality.index:
            scores = lang_quality.loc[language]
            print(f"      {language}:")
            print(f"        Code Quality: {scores['code_quality_score']:.3f}")
            print(f"        Performance: {scores['performance_score']:.3f}")
            print(f"        Scalability: {scores['scalability_score']:.3f}")
            print(f"        Maintainability: {scores['maintainability_score']:.3f}")

        # Overall quality metrics
        print(f"\n    📊 Overall Technical Quality:")
        overall_scores = self.implementation_data[
            [
                "code_quality_score",
                "performance_score",
                "scalability_score",
                "maintainability_score",
                "testing_coverage",
                "documentation_score",
            ]
        ].mean()

        for metric, score in overall_scores.items():
            print(f"      {metric.replace('_', ' ').title()}: {score:.3f}")

        # Technology recommendations
        print(f"\n    💡 Technology Recommendations:")
        best_language = lang_quality["code_quality_score"].idxmax()
        best_ml = (
            self.implementation_data.groupby("ml_framework")["performance_score"]
            .mean()
            .idxmax()
        )
        best_viz = (
            self.implementation_data.groupby("visualization_tool")["code_quality_score"]
            .mean()
            .idxmax()
        )

        print(f"      Best Language for Code Quality: {best_language}")
        print(f"      Best ML Framework for Performance: {best_ml}")
        print(f"      Best Visualization Tool for Quality: {best_viz}")

        return {
            "overall_scores": overall_scores,
            "language_quality": lang_quality,
            "best_technologies": {
                "language": best_language,
                "ml_framework": best_ml,
                "visualization": best_viz,
            },
        }


class PortfolioOptimization:
    """Demonstrate portfolio optimization and presentation strategies."""

    def __init__(self):
        self.optimization_data = None

    def create_optimization_dataset(self):
        """Create a dataset for portfolio optimization analysis."""
        print("\n3. PORTFOLIO OPTIMIZATION AND PRESENTATION")
        print("=" * 60)

        print("\n3.1 CREATING OPTIMIZATION DATASET:")
        print("-" * 45)

        # Generate synthetic optimization data
        n_optimizations = 60

        # Optimization strategies and metrics
        optimization_strategies = [
            "Documentation",
            "Deployment",
            "Testing",
            "Performance",
            "Visualization",
            "API Development",
            "Cloud Integration",
            "User Experience",
        ]
        presentation_formats = [
            "GitHub README",
            "Technical Blog",
            "Video Demo",
            "Live Demo",
            "Case Study",
            "Portfolio Website",
            "Conference Talk",
            "Workshop",
        ]

        optimization_data = {
            "optimization_id": range(1, n_optimizations + 1),
            "project_id": np.random.choice(range(1, 51), n_optimizations),
            "strategy": np.random.choice(optimization_strategies, n_optimizations),
            "presentation_format": np.random.choice(
                presentation_formats, n_optimizations
            ),
            "time_investment_hours": np.random.exponential(20, n_optimizations).astype(
                int
            ),
            "impact_score": np.random.uniform(0.5, 1.0, n_optimizations),
            "visibility_increase": np.random.uniform(0.1, 0.5, n_optimizations),
            "engagement_score": np.random.uniform(0.4, 1.0, n_optimizations),
            "career_impact": np.random.uniform(0.6, 1.0, n_optimizations),
            "implementation_difficulty": np.random.choice(
                ["Easy", "Medium", "Hard"], n_optimizations, p=[0.4, 0.4, 0.2]
            ),
        }

        self.optimization_data = pd.DataFrame(optimization_data)

        print(
            f"  ✅ Optimization dataset created: {len(self.optimization_data):,} optimization strategies"
        )
        print(f"  🔍 Strategy distribution:")
        print(self.optimization_data["strategy"].value_counts().head(5))
        print(f"  📊 Presentation format distribution:")
        print(self.optimization_data["presentation_format"].value_counts().head(5))

        return self.optimization_data

    def analyze_optimization_effectiveness(self):
        """Analyze optimization strategy effectiveness."""
        print("\n3.2 OPTIMIZATION EFFECTIVENESS ANALYSIS:")
        print("-" * 45)

        print("  🔍 Optimization Strategy Effectiveness:")
        print("    " + "=" * 50)

        # Analyze by strategy
        print(f"\n    📊 Effectiveness by Strategy:")
        strategy_effectiveness = self.optimization_data.groupby("strategy")[
            ["impact_score", "visibility_increase", "engagement_score", "career_impact"]
        ].mean()

        for strategy in strategy_effectiveness.index:
            scores = strategy_effectiveness.loc[strategy]
            print(f"      {strategy}:")
            print(f"        Impact Score: {scores['impact_score']:.3f}")
            print(f"        Visibility Increase: {scores['visibility_increase']:.3f}")
            print(f"        Engagement Score: {scores['engagement_score']:.3f}")
            print(f"        Career Impact: {scores['career_impact']:.3f}")

        # Analyze by presentation format
        print(f"\n    📊 Effectiveness by Presentation Format:")
        format_effectiveness = self.optimization_data.groupby("presentation_format")[
            ["impact_score", "visibility_increase", "engagement_score", "career_impact"]
        ].mean()

        for format_type in format_effectiveness.index:
            scores = format_effectiveness.loc[format_type]
            print(f"      {format_type}:")
            print(f"        Impact Score: {scores['impact_score']:.3f}")
            print(f"        Visibility Increase: {scores['visibility_increase']:.3f}")
            print(f"        Engagement Score: {scores['engagement_score']:.3f}")
            print(f"        Career Impact: {scores['career_impact']:.3f}")

        # ROI analysis
        print(f"\n    💡 ROI Analysis:")
        self.optimization_data["roi"] = (
            self.optimization_data["career_impact"]
            * self.optimization_data["visibility_increase"]
        ) / (self.optimization_data["time_investment_hours"] / 100)

        best_roi_strategies = self.optimization_data.nlargest(5, "roi")[
            ["strategy", "roi"]
        ]
        print(f"      Top 5 ROI Strategies:")
        for _, row in best_roi_strategies.iterrows():
            print(f"        {row['strategy']}: {row['roi']:.3f}")

        return {
            "strategy_effectiveness": strategy_effectiveness,
            "format_effectiveness": format_effectiveness,
            "roi_analysis": best_roi_strategies,
        }


class CareerApplication:
    """Demonstrate career application and advancement strategies."""

    def __init__(self):
        self.career_data = None

    def create_career_dataset(self):
        """Create a dataset for career application analysis."""
        print("\n4. CAREER APPLICATION AND ADVANCEMENT")
        print("=" * 60)

        print("\n4.1 CREATING CAREER APPLICATION DATASET:")
        print("-" * 45)

        # Generate synthetic career data
        n_applications = 80

        # Career application metrics
        job_levels = [
            "Entry Level",
            "Mid Level",
            "Senior",
            "Lead",
            "Manager",
            "Director",
        ]
        industries = [
            "Tech",
            "Finance",
            "Healthcare",
            "E-commerce",
            "Consulting",
            "Government",
            "Education",
            "Manufacturing",
        ]
        application_channels = [
            "Portfolio Website",
            "GitHub",
            "LinkedIn",
            "Direct Application",
            "Referral",
            "Recruiter",
            "Job Board",
            "Networking",
        ]

        career_data = {
            "application_id": range(1, n_applications + 1),
            "job_level": np.random.choice(
                job_levels, n_applications, p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
            ),
            "industry": np.random.choice(
                industries,
                n_applications,
                p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.03, 0.02, 0.02],
            ),
            "application_channel": np.random.choice(
                application_channels, n_applications
            ),
            "portfolio_quality_score": np.random.uniform(0.6, 1.0, n_applications),
            "technical_skill_match": np.random.uniform(0.5, 1.0, n_applications),
            "communication_skill_score": np.random.uniform(0.6, 1.0, n_applications),
            "project_relevance": np.random.uniform(0.5, 1.0, n_applications),
            "interview_performance": np.random.uniform(0.4, 1.0, n_applications),
            "offer_received": np.random.choice(
                [True, False], n_applications, p=[0.6, 0.4]
            ),
            "salary_negotiation_success": np.random.choice(
                [True, False], n_applications, p=[0.7, 0.3]
            ),
        }

        self.career_data = pd.DataFrame(career_data)

        print(
            f"  ✅ Career application dataset created: {len(self.career_data):,} applications"
        )
        print(f"  🔍 Job level distribution:")
        print(self.career_data["job_level"].value_counts())
        print(f"  📊 Industry distribution:")
        print(self.career_data["industry"].value_counts())

        return self.career_data

    def analyze_career_effectiveness(self):
        """Analyze career application effectiveness."""
        print("\n4.2 CAREER APPLICATION EFFECTIVENESS:")
        print("-" * 45)

        print("  🔍 Career Application Analysis:")
        print("    " + "=" * 45)

        # Analyze by application channel
        print(f"\n    📊 Effectiveness by Application Channel:")
        channel_effectiveness = self.career_data.groupby("application_channel")[
            [
                "portfolio_quality_score",
                "technical_skill_match",
                "communication_skill_score",
                "project_relevance",
                "interview_performance",
            ]
        ].mean()

        for channel in channel_effectiveness.index:
            scores = channel_effectiveness.loc[channel]
            print(f"      {channel}:")
            print(f"        Portfolio Quality: {scores['portfolio_quality_score']:.3f}")
            print(f"        Technical Match: {scores['technical_skill_match']:.3f}")
            print(f"        Communication: {scores['communication_skill_score']:.3f}")
            print(f"        Project Relevance: {scores['project_relevance']:.3f}")
            print(
                f"        Interview Performance: {scores['interview_performance']:.3f}"
            )

        # Success rate analysis
        print(f"\n    📊 Success Rate Analysis:")
        overall_offer_rate = (
            self.career_data["offer_received"].sum() / len(self.career_data)
        ) * 100
        print(f"      Overall Offer Rate: {overall_offer_rate:.1f}%")

        # Success by portfolio quality
        high_portfolio = self.career_data[
            self.career_data["portfolio_quality_score"] >= 0.8
        ]
        low_portfolio = self.career_data[
            self.career_data["portfolio_quality_score"] < 0.8
        ]

        high_offer_rate = (
            high_portfolio["offer_received"].sum() / len(high_portfolio)
        ) * 100
        low_offer_rate = (
            low_portfolio["offer_received"].sum() / len(low_portfolio)
        ) * 100

        print(f"      High Portfolio Quality (≥0.8): {high_offer_rate:.1f}% offer rate")
        print(f"      Low Portfolio Quality (<0.8): {low_offer_rate:.1f}% offer rate")

        # Industry insights
        print(f"\n    💡 Industry Insights:")
        industry_success = self.career_data.groupby("industry")["offer_received"].agg(
            ["count", "sum", "mean"]
        )
        industry_success["success_rate"] = industry_success["mean"] * 100

        top_industries = industry_success.nlargest(3, "success_rate")
        print(f"      Top 3 Industries by Success Rate:")
        for industry in top_industries.index:
            success_rate = top_industries.loc[industry, "success_rate"]
            print(f"        {industry}: {success_rate:.1f}%")

        return {
            "channel_effectiveness": channel_effectiveness,
            "overall_offer_rate": overall_offer_rate,
            "portfolio_impact": {
                "high_quality_rate": high_offer_rate,
                "low_quality_rate": low_offer_rate,
            },
            "top_industries": top_industries,
        }


def create_portfolio_visualizations():
    """Create comprehensive visualizations for portfolio development concepts."""
    print("\n5. CREATING PORTFOLIO VISUALIZATIONS:")
    print("-" * 45)

    print("  Generating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Portfolio Development: Strategic Project Creation and Optimization",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Project Type Distribution
    ax1 = axes[0, 0]
    project_types = [
        "ML Prediction",
        "Data Visualization",
        "NLP Analysis",
        "Computer Vision",
        "Time Series",
        "Clustering",
        "Recommendation",
        "Anomaly Detection",
    ]
    project_counts = [8, 7, 6, 5, 6, 5, 7, 6]  # Simulated counts

    bars = ax1.bar(
        project_types,
        project_counts,
        color=[
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFA07A",
            "#98D8C8",
            "#FFB6C1",
            "#DDA0DD",
        ],
        alpha=0.8,
    )
    ax1.set_title("Portfolio Project Type Distribution", fontweight="bold")
    ax1.set_ylabel("Number of Projects")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, count in zip(bars, project_counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Domain Coverage
    ax2 = axes[0, 1]
    domains = [
        "E-commerce",
        "Healthcare",
        "Finance",
        "Marketing",
        "Transportation",
        "Education",
        "Entertainment",
        "Manufacturing",
    ]
    domain_counts = [7, 6, 8, 6, 5, 4, 7, 7]  # Simulated counts

    bars = ax2.bar(
        domains,
        domain_counts,
        color=[
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFA07A",
            "#98D8C8",
            "#FFB6C1",
            "#DDA0DD",
        ],
        alpha=0.8,
    )
    ax2.set_title("Portfolio Domain Coverage", fontweight="bold")
    ax2.set_ylabel("Number of Projects")
    ax2.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, count in zip(bars, domain_counts):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Quality Metrics by Complexity
    ax3 = axes[0, 2]
    complexity_levels = ["Beginner", "Intermediate", "Advanced", "Expert"]
    technical_scores = [0.75, 0.82, 0.88, 0.92]  # Simulated scores
    business_scores = [0.70, 0.78, 0.85, 0.90]  # Simulated scores

    x = np.arange(len(complexity_levels))
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        technical_scores,
        width,
        label="Technical Skills",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax3.bar(
        x + width / 2,
        business_scores,
        width,
        label="Business Impact",
        color="#4ECDC4",
        alpha=0.8,
    )

    ax3.set_title("Quality Metrics by Complexity Level", fontweight="bold")
    ax3.set_ylabel("Quality Score")
    ax3.set_xticks(x)
    ax3.set_xticklabels(complexity_levels)
    ax3.legend()
    ax3.set_ylim(0, 1)

    # 4. Technology Stack Distribution
    ax4 = axes[1, 0]
    technologies = [
        "Python",
        "scikit-learn",
        "TensorFlow",
        "Matplotlib",
        "AWS",
        "Docker",
    ]
    usage_percentages = [85, 60, 25, 70, 40, 30]  # Simulated percentages

    bars = ax4.barh(
        technologies,
        usage_percentages,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax4.set_title("Technology Stack Usage", fontweight="bold")
    ax4.set_xlabel("Usage Percentage (%)")
    ax4.set_xlim(0, 100)

    # Add value labels
    for bar, pct in zip(bars, usage_percentages):
        width = bar.get_width()
        ax4.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct}%",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # 5. Optimization Strategy ROI
    ax5 = axes[1, 1]
    strategies = [
        "Documentation",
        "Deployment",
        "Testing",
        "Performance",
        "Visualization",
        "API Dev",
    ]
    roi_scores = [0.85, 0.78, 0.72, 0.68, 0.75, 0.80]  # Simulated ROI scores

    bars = ax5.bar(
        strategies,
        roi_scores,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax5.set_title("Optimization Strategy ROI", fontweight="bold")
    ax5.set_ylabel("ROI Score")
    ax5.set_ylim(0, 1)
    ax5.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, score in zip(bars, roi_scores):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 6. Career Impact by Project Type
    ax6 = axes[1, 2]
    project_categories = [
        "ML Projects",
        "Visualization",
        "NLP",
        "Computer Vision",
        "Time Series",
    ]
    career_impact = [0.88, 0.75, 0.82, 0.79, 0.85]  # Simulated impact scores

    bars = ax6.barh(
        project_categories,
        career_impact,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"],
        alpha=0.8,
    )
    ax6.set_title("Career Impact by Project Type", fontweight="bold")
    ax6.set_xlabel("Career Impact Score")
    ax6.set_xlim(0, 1)

    # Add value labels
    for bar, score in zip(bars, career_impact):
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
    output_file = "portfolio_development.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✅ Visualization saved: {output_file}")

    plt.show()


def main():
    """Main function to run portfolio development demonstrations."""
    try:
        print("=" * 80)
        print("CHAPTER 22: PORTFOLIO DEVELOPMENT")
        print("=" * 80)

        # Initialize portfolio development demonstrations
        portfolio = PortfolioProject()
        technical = TechnicalImplementation()
        optimization = PortfolioOptimization()
        career = CareerApplication()

        # Run portfolio project demonstrations
        print("\n" + "=" * 80)
        project_data = portfolio.create_portfolio_dataset()
        portfolio_analysis = portfolio.analyze_portfolio_strategy()

        # Run technical implementation demonstrations
        print("\n" + "=" * 80)
        implementation_data = technical.create_implementation_dataset()
        technical_analysis = technical.analyze_technical_quality()

        # Run optimization demonstrations
        print("\n" + "=" * 80)
        optimization_data = optimization.create_optimization_dataset()
        optimization_analysis = optimization.analyze_optimization_effectiveness()

        # Run career application demonstrations
        print("\n" + "=" * 80)
        career_data = career.create_career_dataset()
        career_analysis = career.analyze_career_effectiveness()

        # Create visualizations
        create_portfolio_visualizations()

        print("\n" + "=" * 80)
        print("CHAPTER 22 - PORTFOLIO DEVELOPMENT COMPLETED!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Portfolio project design and strategic planning")
        print("  • Technical implementation and architecture optimization")
        print("  • Portfolio optimization and presentation strategies")
        print("  • Career application and advancement techniques")

        print("\n📊 Generated Visualizations:")
        print(
            "  • portfolio_development.png - Comprehensive portfolio development dashboard"
        )

        print("\n🚀 Next Steps:")
        print("  • Apply portfolio strategies to your own projects")
        print("  • Optimize existing projects using the frameworks learned")
        print("  • Continue to Chapter 23: Career Development")

    except Exception as e:
        print(f"\n❌ Error in Chapter 22: {str(e)}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
