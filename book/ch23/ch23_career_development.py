#!/usr/bin/env python3
"""
Chapter 23: Career Development
==============================

This chapter covers essential strategies for launching and advancing
your data science career, including job search techniques, interview
preparation, networking, and long-term career planning.
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


class JobSearchStrategy:
    """Demonstrate job search strategy and market analysis."""

    def __init__(self):
        self.market_data = None
        self.job_data = None

    def create_market_dataset(self):
        """Create a synthetic dataset for job market analysis."""
        print("1. JOB SEARCH STRATEGY AND MARKET ANALYSIS")
        print("=" * 60)

        print("\n1.1 CREATING JOB MARKET DATASET:")
        print("-" * 45)

        # Generate synthetic job market data
        n_jobs = 200

        # Job market characteristics
        job_titles = [
            "Data Scientist",
            "ML Engineer",
            "Data Analyst",
            "Data Engineer",
            "Research Scientist",
            "Analytics Manager",
            "AI Engineer",
            "Business Intelligence Analyst",
        ]
        industries = [
            "Technology",
            "Finance",
            "Healthcare",
            "E-commerce",
            "Consulting",
            "Manufacturing",
            "Education",
            "Government",
        ]
        locations = [
            "San Francisco",
            "New York",
            "Seattle",
            "Austin",
            "Boston",
            "Chicago",
            "Los Angeles",
            "Remote",
        ]
        experience_levels = [
            "Entry Level",
            "Mid Level",
            "Senior",
            "Lead",
            "Manager",
            "Director",
            "VP",
            "C-Level",
        ]

        market_data = {
            "job_id": range(1, n_jobs + 1),
            "job_title": np.random.choice(job_titles, n_jobs),
            "industry": np.random.choice(
                industries, n_jobs, p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.03, 0.02, 0.02]
            ),
            "location": np.random.choice(
                locations, n_jobs, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.02, 0.02, 0.01]
            ),
            "experience_level": np.random.choice(
                experience_levels,
                n_jobs,
                p=[0.2, 0.3, 0.25, 0.15, 0.05, 0.03, 0.01, 0.01],
            ),
            "salary_min": np.random.lognormal(11, 0.3, n_jobs).astype(int),
            "salary_max": np.random.lognormal(11.5, 0.3, n_jobs).astype(int),
            "remote_friendly": np.random.choice([True, False], n_jobs, p=[0.6, 0.4]),
            "equity_offered": np.random.choice([True, False], n_jobs, p=[0.7, 0.3]),
            "company_size": np.random.choice(
                ["Startup", "Small", "Medium", "Large", "Enterprise"],
                n_jobs,
                p=[0.2, 0.2, 0.2, 0.25, 0.15],
            ),
            "application_count": np.random.poisson(50, n_jobs),
            "days_posted": np.random.exponential(14, n_jobs).astype(int),
            "market_demand_score": np.random.uniform(0.6, 1.0, n_jobs),
        }

        # Calculate salary range
        market_data["salary_range"] = (
            market_data["salary_max"] - market_data["salary_min"]
        )
        market_data["avg_salary"] = (
            market_data["salary_max"] + market_data["salary_min"]
        ) / 2

        self.market_data = pd.DataFrame(market_data)

        print(
            f"  ✅ Job market dataset created: {len(self.market_data):,} job postings"
        )
        print(f"  🔍 Job title distribution:")
        print(self.market_data["job_title"].value_counts().head(5))
        print(f"  📊 Industry distribution:")
        print(self.market_data["industry"].value_counts().head(5))

        return self.market_data

    def analyze_market_conditions(self):
        """Analyze current job market conditions and opportunities."""
        print("\n1.2 JOB MARKET ANALYSIS:")
        print("-" * 35)

        print("  🔍 Market Condition Insights:")
        print("    " + "=" * 45)

        # Salary analysis by role and experience
        print(f"\n    📊 Salary Analysis by Role:")
        role_salary = (
            self.market_data.groupby("job_title")["avg_salary"]
            .agg(["mean", "count"])
            .round(0)
        )
        for role in role_salary.index:
            avg_salary = role_salary.loc[role, "mean"]
            count = role_salary.loc[role, "count"]
            print(f"      {role}: ${avg_salary:,.0f} ({count} positions)")

        # Market demand analysis
        print(f"\n    📊 Market Demand Analysis:")
        demand_by_industry = (
            self.market_data.groupby("industry")["market_demand_score"]
            .mean()
            .sort_values(ascending=False)
        )
        print(f"      Top Industries by Demand:")
        for industry in demand_by_industry.head(5).index:
            demand = demand_by_industry.loc[industry]
            print(f"        {industry}: {demand:.3f}")

        # Remote work opportunities
        remote_stats = self.market_data["remote_friendly"].value_counts()
        remote_pct = (remote_stats[True] / len(self.market_data)) * 100
        print(f"\n    💡 Remote Work Opportunities:")
        print(f"      Remote-friendly positions: {remote_pct:.1f}%")

        # Application competition analysis
        print(f"\n    📊 Application Competition:")
        avg_applications = self.market_data["application_count"].mean()
        print(f"      Average applications per job: {avg_applications:.0f}")

        # High-competition vs. low-competition jobs
        high_comp = self.market_data[
            self.market_data["application_count"] > avg_applications
        ]
        low_comp = self.market_data[
            self.market_data["application_count"] <= avg_applications
        ]

        print(
            f"      High-competition jobs: {len(high_comp)} ({len(high_comp)/len(self.market_data)*100:.1f}%)"
        )
        print(
            f"      Low-competition jobs: {len(low_comp)} ({len(low_comp)/len(self.market_data)*100:.1f}%)"
        )

        return {
            "role_salary": role_salary,
            "demand_by_industry": demand_by_industry,
            "remote_opportunities": remote_pct,
            "competition_analysis": {
                "avg_applications": avg_applications,
                "high_competition": len(high_comp),
                "low_competition": len(low_comp),
            },
        }


class ApplicationOptimization:
    """Demonstrate application and resume optimization strategies."""

    def __init__(self):
        self.application_data = None

    def create_application_dataset(self):
        """Create a dataset for application optimization analysis."""
        print("\n2. APPLICATION AND RESUME OPTIMIZATION")
        print("=" * 60)

        print("\n2.1 CREATING APPLICATION DATASET:")
        print("-" * 45)

        # Generate synthetic application data
        n_applications = 150

        # Application characteristics
        resume_formats = [
            "ATS Optimized",
            "Creative Design",
            "Traditional",
            "Infographic",
            "Portfolio Style",
        ]
        cover_letter_types = [
            "Customized",
            "Template Based",
            "Story Driven",
            "Technical Focus",
            "Business Focus",
        ]
        application_channels = [
            "Company Website",
            "LinkedIn",
            "Indeed",
            "Referral",
            "Recruiter",
            "Direct Email",
            "Job Board",
            "Networking",
        ]

        application_data = {
            "application_id": range(1, n_applications + 1),
            "resume_format": np.random.choice(
                resume_formats, n_applications, p=[0.4, 0.2, 0.2, 0.1, 0.1]
            ),
            "cover_letter_type": np.random.choice(
                cover_letter_types, n_applications, p=[0.5, 0.2, 0.15, 0.1, 0.05]
            ),
            "application_channel": np.random.choice(
                application_channels, n_applications
            ),
            "ats_score": np.random.uniform(0.6, 1.0, n_applications),
            "resume_quality_score": np.random.uniform(0.5, 1.0, n_applications),
            "cover_letter_quality": np.random.uniform(0.5, 1.0, n_applications),
            "portfolio_integration": np.random.uniform(0.4, 1.0, n_applications),
            "customization_level": np.random.uniform(0.3, 1.0, n_applications),
            "follow_up_actions": np.random.choice(
                [0, 1, 2, 3], n_applications, p=[0.3, 0.4, 0.2, 0.1]
            ),
            "response_received": np.random.choice(
                [True, False], n_applications, p=[0.4, 0.6]
            ),
            "interview_invitation": np.random.choice(
                [True, False], n_applications, p=[0.25, 0.75]
            ),
        }

        self.application_data = pd.DataFrame(application_data)

        print(
            f"  ✅ Application dataset created: {len(self.application_data):,} applications"
        )
        print(f"  🔍 Resume format distribution:")
        print(self.application_data["resume_format"].value_counts())
        print(f"  📊 Application channel distribution:")
        print(self.application_data["application_channel"].value_counts().head(5))

        return self.application_data


class InterviewPreparation:
    """Demonstrate technical interview preparation strategies."""

    def __init__(self):
        self.interview_data = None

    def create_interview_dataset(self):
        """Create a dataset for interview preparation analysis."""
        print("\n3. TECHNICAL INTERVIEW PREPARATION")
        print("=" * 60)

        print("\n3.1 CREATING INTERVIEW DATASET:")
        print("-" * 45)

        # Generate synthetic interview data
        n_interviews = 100

        # Interview characteristics
        interview_types = [
            "Technical Phone Screen",
            "Coding Challenge",
            "System Design",
            "Take Home Assignment",
            "Onsite Technical",
            "Behavioral",
            "Case Study",
            "Portfolio Review",
        ]
        question_categories = [
            "Machine Learning",
            "Statistics",
            "Programming",
            "Data Structures",
            "Algorithms",
            "Database",
            "Big Data",
            "Business Case",
        ]
        difficulty_levels = ["Easy", "Medium", "Hard", "Expert"]

        interview_data = {
            "interview_id": range(1, n_interviews + 1),
            "interview_type": np.random.choice(interview_types, n_interviews),
            "question_category": np.random.choice(question_categories, n_interviews),
            "difficulty_level": np.random.choice(
                difficulty_levels, n_interviews, p=[0.2, 0.4, 0.3, 0.1]
            ),
            "preparation_hours": np.random.exponential(20, n_interviews).astype(int),
            "technical_skill_score": np.random.uniform(0.5, 1.0, n_interviews),
            "problem_solving_score": np.random.uniform(0.4, 1.0, n_interviews),
            "communication_score": np.random.uniform(0.6, 1.0, n_interviews),
            "confidence_level": np.random.uniform(0.3, 1.0, n_interviews),
            "mock_interview_practice": np.random.choice(
                [0, 1, 2, 3, 4, 5], n_interviews, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]
            ),
            "interview_success": np.random.choice(
                [True, False], n_interviews, p=[0.6, 0.4]
            ),
        }

        self.interview_data = pd.DataFrame(interview_data)

        print(
            f"  ✅ Interview dataset created: {len(self.interview_data):,} interviews"
        )
        print(f"  🔍 Interview type distribution:")
        print(self.interview_data["interview_type"].value_counts().head(5))
        print(f"  📊 Question category distribution:")
        print(self.interview_data["question_category"].value_counts().head(5))

        return self.interview_data


class NetworkingStrategy:
    """Demonstrate networking and personal branding strategies."""

    def __init__(self):
        self.networking_data = None

    def create_networking_dataset(self):
        """Create a dataset for networking strategy analysis."""
        print("\n4. NETWORKING AND PERSONAL BRANDING")
        print("=" * 60)

        print("\n4.1 CREATING NETWORKING DATASET:")
        print("-" * 45)

        # Generate synthetic networking data
        n_connections = 120

        # Networking characteristics
        connection_types = [
            "Former Colleague",
            "Industry Professional",
            "Conference Contact",
            "LinkedIn Connection",
            "Alumni",
            "Mentor",
            "Mentee",
            "Industry Leader",
        ]
        networking_activities = [
            "LinkedIn Posts",
            "Blog Articles",
            "Conference Speaking",
            "Meetup Attendance",
            "Open Source Contributions",
            "Social Media Engagement",
            "Professional Groups",
            "Webinars",
        ]

        networking_data = {
            "connection_id": range(1, n_connections + 1),
            "connection_type": np.random.choice(connection_types, n_connections),
            "networking_activity": np.random.choice(
                networking_activities, n_connections
            ),
            "connection_strength": np.random.uniform(0.3, 1.0, n_connections),
            "professional_value": np.random.uniform(0.4, 1.0, n_connections),
            "content_creation_score": np.random.uniform(0.2, 1.0, n_connections),
            "engagement_level": np.random.uniform(0.3, 1.0, n_connections),
            "career_opportunities": np.random.choice(
                [0, 1, 2, 3, 4, 5], n_connections, p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
            ),
            "knowledge_sharing": np.random.uniform(0.4, 1.0, n_connections),
            "brand_visibility": np.random.uniform(0.3, 1.0, n_connections),
            "long_term_value": np.random.uniform(0.5, 1.0, n_connections),
        }

        self.networking_data = pd.DataFrame(networking_data)

        print(
            f"  ✅ Networking dataset created: {len(self.networking_data):,} connections"
        )
        print(f"  🔍 Connection type distribution:")
        print(self.networking_data["connection_type"].value_counts().head(5))
        print(f"  📊 Networking activity distribution:")
        print(self.networking_data["networking_activity"].value_counts().head(5))

        return self.networking_data


def create_career_visualizations():
    """Create comprehensive visualizations for career development concepts."""
    print("\n5. CREATING CAREER DEVELOPMENT VISUALIZATIONS:")
    print("-" * 45)

    print("  Generating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Career Development: Strategic Career Planning and Advancement",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Job Market Salary Distribution
    ax1 = axes[0, 0]
    salary_ranges = [
        "$50K-75K",
        "$75K-100K",
        "$100K-125K",
        "$125K-150K",
        "$150K-175K",
        "$175K-200K",
        "$200K+",
    ]
    salary_counts = [15, 25, 35, 40, 30, 20, 15]  # Simulated counts

    bars = ax1.bar(
        salary_ranges,
        salary_counts,
        color=[
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFA07A",
            "#98D8C8",
            "#FFB6C1",
        ],
        alpha=0.8,
    )
    ax1.set_title("Job Market Salary Distribution", fontweight="bold")
    ax1.set_ylabel("Number of Positions")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, count in zip(bars, salary_counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Application Success Rates by Channel
    ax2 = axes[0, 1]
    channels = [
        "Company Website",
        "LinkedIn",
        "Referral",
        "Recruiter",
        "Job Board",
        "Networking",
    ]
    response_rates = [0.35, 0.42, 0.65, 0.58, 0.28, 0.52]  # Simulated rates
    interview_rates = [0.18, 0.22, 0.38, 0.32, 0.12, 0.28]  # Simulated rates

    x = np.arange(len(channels))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        response_rates,
        width,
        label="Response Rate",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        x + width / 2,
        interview_rates,
        width,
        label="Interview Rate",
        color="#4ECDC4",
        alpha=0.8,
    )

    ax2.set_title("Application Success Rates by Channel", fontweight="bold")
    ax2.set_ylabel("Success Rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 0.8)

    # 3. Interview Preparation Impact
    ax3 = axes[0, 2]
    prep_hours = ["0-10", "10-20", "20-30", "30-40", "40+"]
    success_rates = [0.45, 0.58, 0.72, 0.81, 0.85]  # Simulated rates

    bars = ax3.bar(
        prep_hours,
        success_rates,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"],
        alpha=0.8,
    )
    ax3.set_title("Interview Success vs. Preparation", fontweight="bold")
    ax3.set_ylabel("Success Rate")
    ax3.set_ylim(0, 1)

    # Add value labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Networking Activity Effectiveness
    ax4 = axes[1, 0]
    activities = [
        "LinkedIn Posts",
        "Blog Articles",
        "Conference Speaking",
        "Meetup Attendance",
        "Open Source",
        "Social Media",
    ]
    effectiveness_scores = [0.75, 0.82, 0.88, 0.68, 0.85, 0.72]  # Simulated scores

    bars = ax4.barh(
        activities,
        effectiveness_scores,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax4.set_title("Networking Activity Effectiveness", fontweight="bold")
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

    # 5. Career Progression Timeline
    ax5 = axes[1, 1]
    career_stages = [
        "Entry Level",
        "Mid Level",
        "Senior",
        "Lead",
        "Manager",
        "Director",
        "VP",
        "C-Level",
    ]
    avg_years = [0, 2, 5, 8, 12, 15, 18, 22]  # Simulated years

    bars = ax5.bar(
        career_stages,
        avg_years,
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
    ax5.set_title("Career Progression Timeline", fontweight="bold")
    ax5.set_ylabel("Years of Experience")
    ax5.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, years in zip(bars, avg_years):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.2,
            f"{years}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 6. Skill Development Investment
    ax6 = axes[1, 2]
    skill_areas = [
        "Technical Skills",
        "Business Acumen",
        "Communication",
        "Leadership",
        "Domain Knowledge",
        "Networking",
    ]
    investment_hours = [120, 80, 60, 100, 90, 70]  # Simulated hours

    bars = ax6.barh(
        skill_areas,
        investment_hours,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax6.set_title("Skill Development Investment", fontweight="bold")
    ax6.set_xlabel("Hours per Month")

    # Add value labels
    for bar, hours in zip(bars, investment_hours):
        width = bar.get_width()
        ax6.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2.0,
            f"{hours}h",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save visualization
    output_file = "career_development.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✅ Visualization saved: {output_file}")

    plt.show()


def main():
    """Main function to run career development demonstrations."""
    try:
        print("=" * 80)
        print("CHAPTER 23: CAREER DEVELOPMENT")
        print("=" * 80)

        # Initialize career development demonstrations
        job_search = JobSearchStrategy()
        application = ApplicationOptimization()
        interview = InterviewPreparation()
        networking = NetworkingStrategy()

        # Run job search strategy demonstrations
        print("\n" + "=" * 80)
        market_data = job_search.create_market_dataset()
        market_analysis = job_search.analyze_market_conditions()

        # Run application optimization demonstrations
        print("\n" + "=" * 80)
        application_data = application.create_application_dataset()

        # Run interview preparation demonstrations
        print("\n" + "=" * 80)
        interview_data = interview.create_interview_dataset()

        # Run networking strategy demonstrations
        print("\n" + "=" * 80)
        networking_data = networking.create_networking_dataset()

        # Create visualizations
        create_career_visualizations()

        print("\n" + "=" * 80)
        print("CHAPTER 23 - CAREER DEVELOPMENT COMPLETED!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Job search strategy and market analysis techniques")
        print("  • Application and resume optimization strategies")
        print("  • Technical interview preparation and success factors")
        print("  • Networking and personal branding effectiveness")

        print("\n📊 Generated Visualizations:")
        print("  • career_development.png - Comprehensive career development dashboard")

        print("\n🚀 Next Steps:")
        print("  • Apply career strategies to your job search")
        print("  • Optimize your application materials and interview preparation")
        print("  • Build your professional network and personal brand")
        print(
            "  • Congratulations! You've completed the comprehensive Data Science Book!"
        )

    except Exception as e:
        print(f"\n❌ Error in Chapter 23: {str(e)}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
