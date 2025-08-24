#!/usr/bin/env python3
"""
Chapter 24: Advanced Career Specializations and Industry Focus
==============================================================

This final chapter covers specialized data science career paths,
industry-specific development, emerging trends, and advanced career
strategies. You'll learn how to differentiate yourself in the
competitive data science market and build specialized expertise.
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


class CareerSpecializationPaths:
    """Demonstrate specialized data science career paths and opportunities."""

    def __init__(self):
        self.specialization_data = None

    def create_specialization_dataset(self):
        """Create a dataset for career specialization analysis."""
        print("1. SPECIALIZED DATA SCIENCE CAREER PATHS")
        print("=" * 60)

        print("\n1.1 CREATING SPECIALIZATION DATASET:")
        print("-" * 45)

        # Generate synthetic specialization data
        n_specializations = 180

        # Specialization characteristics
        career_paths = [
            "Machine Learning Engineering",
            "Data Engineering",
            "Research Scientist",
            "Analytics Leadership",
            "AI Ethics and Governance",
            "Data Science Consulting",
        ]

        technical_domains = [
            "Deep Learning",
            "Big Data Infrastructure",
            "Academic Research",
            "Business Strategy",
            "Responsible AI",
            "Client Engagement",
        ]

        experience_requirements = [
            "3-5 years",
            "4-6 years",
            "5-8 years",
            "6-10 years",
            "4-7 years",
            "5-9 years",
        ]

        specialization_data = {
            "specialization_id": range(1, n_specializations + 1),
            "career_path": np.random.choice(career_paths, n_specializations),
            "technical_domain": np.random.choice(technical_domains, n_specializations),
            "experience_requirement": np.random.choice(
                experience_requirements, n_specializations
            ),
            "technical_skill_score": np.random.uniform(0.7, 1.0, n_specializations),
            "business_acumen_score": np.random.uniform(0.6, 1.0, n_specializations),
            "leadership_score": np.random.uniform(0.5, 1.0, n_specializations),
            "innovation_score": np.random.uniform(0.6, 1.0, n_specializations),
            "market_demand": np.random.uniform(0.6, 1.0, n_specializations),
            "salary_premium": np.random.uniform(0.1, 0.4, n_specializations),
            "growth_potential": np.random.uniform(0.7, 1.0, n_specializations),
            "job_satisfaction": np.random.uniform(0.6, 1.0, n_specializations),
        }

        self.specialization_data = pd.DataFrame(specialization_data)

        print(
            f"  ✅ Specialization dataset created: {len(self.specialization_data):,} specializations"
        )
        print(f"  🔍 Career path distribution:")
        print(self.specialization_data["career_path"].value_counts())
        print(f"  📊 Technical domain distribution:")
        print(self.specialization_data["technical_domain"].value_counts())

        return self.specialization_data

    def analyze_specialization_opportunities(self):
        """Analyze specialization opportunities and market demand."""
        print("\n1.2 SPECIALIZATION OPPORTUNITY ANALYSIS:")
        print("-" * 45)

        print("  🔍 Specialization Market Insights:")
        print("    " + "=" * 50)

        # Career path analysis
        print(f"\n    📊 Career Path Analysis:")
        path_analysis = self.specialization_data.groupby("career_path")[
            ["market_demand", "salary_premium", "growth_potential", "job_satisfaction"]
        ].mean()

        for path in path_analysis.index:
            scores = path_analysis.loc[path]
            print(f"      {path}:")
            print(f"        Market Demand: {scores['market_demand']:.3f}")
            print(f"        Salary Premium: {scores['salary_premium']:.1%}")
            print(f"        Growth Potential: {scores['growth_potential']:.3f}")
            print(f"        Job Satisfaction: {scores['job_satisfaction']:.3f}")

        # Skill requirement analysis
        print(f"\n    📊 Skill Requirement Analysis:")
        skill_analysis = self.specialization_data.groupby("career_path")[
            ["technical_skill_score", "business_acumen_score", "leadership_score"]
        ].mean()

        for path in skill_analysis.index:
            scores = skill_analysis.loc[path]
            print(f"      {path}:")
            print(f"        Technical Skills: {scores['technical_skill_score']:.3f}")
            print(f"        Business Acumen: {scores['business_acumen_score']:.3f}")
            print(f"        Leadership: {scores['leadership_score']:.3f}")

        # Market demand ranking
        print(f"\n    💡 Market Demand Ranking:")
        demand_ranking = (
            self.specialization_data.groupby("career_path")["market_demand"]
            .mean()
            .sort_values(ascending=False)
        )
        for i, (path, demand) in enumerate(demand_ranking.items(), 1):
            print(f"      {i}. {path}: {demand:.3f}")

        return {
            "path_analysis": path_analysis,
            "skill_analysis": skill_analysis,
            "demand_ranking": demand_ranking,
        }


class IndustrySpecificDevelopment:
    """Demonstrate industry-specific career development strategies."""

    def __init__(self):
        self.industry_data = None

    def create_industry_dataset(self):
        """Create a dataset for industry-specific development analysis."""
        print("\n2. INDUSTRY-SPECIFIC CAREER DEVELOPMENT")
        print("=" * 60)

        print("\n2.1 CREATING INDUSTRY DATASET:")
        print("-" * 45)

        # Generate synthetic industry data
        n_industries = 150

        # Industry characteristics
        industries = [
            "Technology and Software",
            "Finance and Banking",
            "Healthcare and Life Sciences",
            "E-commerce and Retail",
            "Manufacturing and IoT",
            "Government and Public Sector",
        ]

        domain_areas = [
            "Product Analytics",
            "Risk Modeling",
            "Clinical Analytics",
            "Customer Analytics",
            "Predictive Maintenance",
            "Policy Analysis",
        ]

        industry_data = {
            "industry_id": range(1, n_industries + 1),
            "industry": np.random.choice(industries, n_industries),
            "domain_area": np.random.choice(domain_areas, n_industries),
            "technical_complexity": np.random.uniform(0.6, 1.0, n_industries),
            "business_impact": np.random.uniform(0.7, 1.0, n_industries),
            "regulatory_requirements": np.random.uniform(0.3, 1.0, n_industries),
            "innovation_opportunity": np.random.uniform(0.6, 1.0, n_industries),
            "career_growth": np.random.uniform(0.6, 1.0, n_industries),
            "salary_level": np.random.uniform(0.7, 1.0, n_industries),
            "work_life_balance": np.random.uniform(0.4, 1.0, n_industries),
            "job_stability": np.random.uniform(0.5, 1.0, n_industries),
        }

        self.industry_data = pd.DataFrame(industry_data)

        print(
            f"  ✅ Industry dataset created: {len(self.industry_data):,} industry specializations"
        )
        print(f"  🔍 Industry distribution:")
        print(self.industry_data["industry"].value_counts())
        print(f"  📊 Domain area distribution:")
        print(self.industry_data["domain_area"].value_counts())

        return self.industry_data


class AdvancedTechnicalSpecializations:
    """Demonstrate advanced technical specializations and skills."""

    def __init__(self):
        self.technical_data = None

    def create_technical_dataset(self):
        """Create a dataset for advanced technical specializations."""
        print("\n3. ADVANCED TECHNICAL SPECIALIZATIONS")
        print("=" * 60)

        print("\n3.1 CREATING TECHNICAL SPECIALIZATION DATASET:")
        print("-" * 45)

        # Generate synthetic technical data
        n_technical = 120

        # Technical specialization characteristics
        technical_areas = [
            "Deep Learning and Neural Networks",
            "Natural Language Processing",
            "Computer Vision",
            "Time Series and Forecasting",
            "Graph Analytics",
            "Edge Computing and IoT",
        ]

        skill_categories = [
            "Model Architecture",
            "Text Processing",
            "Image Analysis",
            "Temporal Modeling",
            "Network Analysis",
            "Real-time Systems",
        ]

        technical_data = {
            "technical_id": range(1, n_technical + 1),
            "technical_area": np.random.choice(technical_areas, n_technical),
            "skill_category": np.random.choice(skill_categories, n_technical),
            "mathematical_complexity": np.random.uniform(0.7, 1.0, n_technical),
            "computational_requirements": np.random.uniform(0.6, 1.0, n_technical),
            "research_opportunity": np.random.uniform(0.6, 1.0, n_technical),
            "industry_applicability": np.random.uniform(0.7, 1.0, n_technical),
            "learning_curve": np.random.uniform(0.5, 1.0, n_technical),
            "career_advantage": np.random.uniform(0.7, 1.0, n_technical),
            "future_relevance": np.random.uniform(0.8, 1.0, n_technical),
        }

        self.technical_data = pd.DataFrame(technical_data)

        print(
            f"  ✅ Technical dataset created: {len(self.technical_data):,} specializations"
        )
        print(f"  🔍 Technical area distribution:")
        print(self.technical_data["technical_area"].value_counts())
        print(f"  📊 Skill category distribution:")
        print(self.technical_data["skill_category"].value_counts())

        return self.technical_data


class LeadershipDevelopment:
    """Demonstrate leadership and management development strategies."""

    def __init__(self):
        self.leadership_data = None

    def create_leadership_dataset(self):
        """Create a dataset for leadership development analysis."""
        print("\n4. LEADERSHIP AND MANAGEMENT DEVELOPMENT")
        print("=" * 60)

        print("\n4.1 CREATING LEADERSHIP DATASET:")
        print("-" * 45)

        # Generate synthetic leadership data
        n_leadership = 100

        # Leadership characteristics
        leadership_areas = [
            "Team Leadership",
            "Strategic Planning",
            "Stakeholder Management",
            "Resource Planning",
            "Change Management",
            "Mentorship and Coaching",
        ]

        skill_levels = ["Beginner", "Intermediate", "Advanced", "Expert"]

        leadership_data = {
            "leadership_id": range(1, n_leadership + 1),
            "leadership_area": np.random.choice(leadership_areas, n_leadership),
            "skill_level": np.random.choice(
                skill_levels, n_leadership, p=[0.2, 0.4, 0.3, 0.1]
            ),
            "team_size_managed": np.random.choice(
                [0, 2, 5, 10, 15, 25, 50],
                n_leadership,
                p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.03, 0.02],
            ),
            "project_budget": np.random.exponential(100000, n_leadership).astype(int),
            "stakeholder_influence": np.random.uniform(0.5, 1.0, n_leadership),
            "decision_making_authority": np.random.uniform(0.4, 1.0, n_leadership),
            "mentorship_experience": np.random.uniform(0.3, 1.0, n_leadership),
            "strategic_impact": np.random.uniform(0.6, 1.0, n_leadership),
            "career_advancement": np.random.uniform(0.7, 1.0, n_leadership),
        }

        self.leadership_data = pd.DataFrame(leadership_data)

        print(
            f"  ✅ Leadership dataset created: {len(self.leadership_data):,} leadership areas"
        )
        print(f"  🔍 Leadership area distribution:")
        print(self.leadership_data["leadership_area"].value_counts())
        print(f"  📊 Skill level distribution:")
        print(self.leadership_data["skill_level"].value_counts())

        return self.leadership_data


def create_advanced_career_visualizations():
    """Create comprehensive visualizations for advanced career specializations."""
    print("\n5. CREATING ADVANCED CAREER VISUALIZATIONS:")
    print("-" * 45)

    print("  Generating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Advanced Career Specializations: Strategic Career Development and Industry Focus",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Career Path Market Demand
    ax1 = axes[0, 0]
    career_paths = [
        "ML Engineering",
        "Data Engineering",
        "Research Scientist",
        "Analytics Leadership",
        "AI Ethics",
        "Consulting",
    ]
    market_demand = [0.92, 0.88, 0.85, 0.90, 0.87, 0.89]  # Simulated scores

    bars = ax1.bar(
        career_paths,
        market_demand,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax1.set_title("Career Path Market Demand", fontweight="bold")
    ax1.set_ylabel("Market Demand Score")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, demand in zip(bars, market_demand):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{demand:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Industry Performance Comparison
    ax2 = axes[0, 1]
    industries = [
        "Technology",
        "Finance",
        "Healthcare",
        "E-commerce",
        "Manufacturing",
        "Government",
    ]
    technical_complexity = [0.95, 0.88, 0.92, 0.85, 0.90, 0.78]  # Simulated scores
    business_impact = [0.92, 0.95, 0.89, 0.93, 0.87, 0.85]  # Simulated scores

    x = np.arange(len(industries))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        technical_complexity,
        width,
        label="Technical Complexity",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        x + width / 2,
        business_impact,
        width,
        label="Business Impact",
        color="#4ECDC4",
        alpha=0.8,
    )

    ax2.set_title("Industry Performance Comparison", fontweight="bold")
    ax2.set_ylabel("Performance Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(industries, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 1)

    # 3. Technical Specialization Future Relevance
    ax3 = axes[0, 2]
    tech_areas = [
        "Deep Learning",
        "NLP",
        "Computer Vision",
        "Time Series",
        "Graph Analytics",
        "Edge Computing",
    ]
    future_relevance = [0.98, 0.95, 0.93, 0.90, 0.88, 0.92]  # Simulated scores

    bars = ax3.bar(
        tech_areas,
        future_relevance,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax3.set_title("Technical Specialization Future Relevance", fontweight="bold")
    ax3.set_ylabel("Future Relevance Score")
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, relevance in zip(bars, future_relevance):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{relevance:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Leadership Development Progression
    ax4 = axes[1, 0]
    leadership_areas = [
        "Team Leadership",
        "Strategic Planning",
        "Stakeholder Mgmt",
        "Resource Planning",
        "Change Mgmt",
        "Mentorship",
    ]
    career_advancement = [0.85, 0.92, 0.88, 0.90, 0.87, 0.89]  # Simulated scores

    bars = ax4.barh(
        leadership_areas,
        career_advancement,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax4.set_title("Leadership Development Progression", fontweight="bold")
    ax4.set_xlabel("Career Advancement Score")
    ax4.set_xlim(0, 1)

    # Add value labels
    for bar, score in zip(bars, career_advancement):
        width = bar.get_width()
        ax4.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # 5. Salary Premium by Specialization
    ax5 = axes[1, 1]
    specializations = [
        "ML Engineering",
        "Data Engineering",
        "Research",
        "Leadership",
        "AI Ethics",
        "Consulting",
    ]
    salary_premiums = [0.35, 0.28, 0.32, 0.40, 0.30, 0.38]  # Simulated percentages

    bars = ax5.bar(
        specializations,
        salary_premiums,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax5.set_title("Salary Premium by Specialization", fontweight="bold")
    ax5.set_ylabel("Salary Premium (%)")
    ax5.set_ylim(0, 0.5)
    ax5.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, premium in zip(bars, salary_premiums):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{premium:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 6. Global Career Opportunities
    ax6 = axes[1, 2]
    regions = [
        "North America",
        "Europe",
        "Asia Pacific",
        "Latin America",
        "Middle East",
        "Africa",
    ]
    opportunity_scores = [0.95, 0.88, 0.92, 0.78, 0.82, 0.75]  # Simulated scores

    bars = ax6.barh(
        regions,
        opportunity_scores,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax6.set_title("Global Career Opportunities", fontweight="bold")
    ax6.set_xlabel("Opportunity Score")
    ax6.set_xlim(0, 1)

    # Add value labels
    for bar, score in zip(bars, opportunity_scores):
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
    output_file = "advanced_career_specializations.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✅ Visualization saved: {output_file}")

    plt.show()


def main():
    """Main function to run advanced career specialization demonstrations."""
    try:
        print("=" * 80)
        print("CHAPTER 24: ADVANCED CAREER SPECIALIZATIONS AND INDUSTRY FOCUS")
        print("=" * 80)

        # Initialize advanced career specialization demonstrations
        specialization = CareerSpecializationPaths()
        industry = IndustrySpecificDevelopment()
        technical = AdvancedTechnicalSpecializations()
        leadership = LeadershipDevelopment()

        # Run specialization path demonstrations
        print("\n" + "=" * 80)
        specialization_data = specialization.create_specialization_dataset()
        specialization_analysis = specialization.analyze_specialization_opportunities()

        # Run industry development demonstrations
        print("\n" + "=" * 80)
        industry_data = industry.create_industry_dataset()

        # Run technical specialization demonstrations
        print("\n" + "=" * 80)
        technical_data = technical.create_technical_dataset()

        # Run leadership development demonstrations
        print("\n" + "=" * 80)
        leadership_data = leadership.create_leadership_dataset()

        # Create visualizations
        create_advanced_career_visualizations()

        print("\n" + "=" * 80)
        print("CHAPTER 24 - ADVANCED CAREER SPECIALIZATIONS COMPLETED!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Specialized data science career paths and opportunities")
        print("  • Industry-specific development strategies and requirements")
        print("  • Advanced technical specializations and future trends")
        print("  • Leadership and management development approaches")

        print("\n📊 Generated Visualizations:")
        print(
            "  • advanced_career_specializations.png - Comprehensive career specialization dashboard"
        )

        print("\n🚀 Next Steps:")
        print("  • Choose your specialization path and develop expertise")
        print("  • Build industry-specific knowledge and domain expertise")
        print("  • Develop advanced technical skills and leadership capabilities")
        print(
            "  • 🎉 CONGRATULATIONS! You've completed the comprehensive Data Science Book!"
        )
        print(
            "  • You now have complete mastery of data science from fundamentals to advanced career development!"
        )

    except Exception as e:
        print(f"\n❌ Error in Chapter 24: {str(e)}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
