#!/usr/bin/env python3
"""
Chapter 1: The Data Science Landscape
Data Voyage: Mapping the Path to Discovery in Data Science

This script demonstrates key concepts from Chapter 1 with real data examples,
comprehensive analysis, and professional visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import matplotlib.patches as patches
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Set up for professional plotting
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def main():
    """Main function to run Chapter 1 demonstrations."""
    print("=" * 80)
    print("CHAPTER 1: THE DATA SCIENCE LANDSCAPE")
    print("=" * 80)
    print()

    # Section 1.1: What is Data Science?
    print("1.1 WHAT IS DATA SCIENCE?")
    print("-" * 40)
    print("Data science is an interdisciplinary field that uses scientific methods,")
    print("processes, algorithms, and systems to extract knowledge and insights")
    print("from structured and unstructured data.")
    print()

    # Create the Data Science Venn Diagram
    create_data_science_venn_diagram()

    # Section 1.2: The Data Science Workflow
    print("\n1.2 THE DATA SCIENCE WORKFLOW")
    print("-" * 40)
    print("Data science projects follow a systematic approach called CRISP-DM:")
    print("1. Business Understanding")
    print("2. Data Understanding")
    print("3. Data Preparation")
    print("4. Modeling")
    print("5. Evaluation")
    print("6. Deployment")
    print()

    create_data_science_workflow()

    # Section 1.3: Real-World Applications with Real Data
    print("\n1.3 REAL-WORLD APPLICATIONS WITH REAL DATA")
    print("-" * 40)
    print("Data science is transforming industries across the globe:")
    print()

    create_industry_applications_with_real_data()

    # Section 1.4: The Data Scientist Role
    print("\n1.4 THE DATA SCIENTIST ROLE")
    print("-" * 40)
    print("Data scientists need a diverse skill set across multiple domains.")
    print()

    create_skills_radar_chart()

    # Section 1.5: Ethics and Responsibility
    print("\n1.5 ETHICS AND RESPONSIBILITY")
    print("-" * 40)
    print("As data scientists, we have a responsibility to use data ethically.")
    print()

    create_ethics_framework()

    # Section 1.6: Real Data Analysis Example
    print("\n1.6 REAL DATA ANALYSIS EXAMPLE")
    print("-" * 40)
    print("Demonstrating data science workflow with real-world data:")
    print()

    demonstrate_real_data_analysis()

    # Chapter Summary
    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ What data science is - An interdisciplinary field combining domain")
    print("   expertise, mathematics, and programming")
    print("✅ The data science workflow - A systematic approach from business")
    print("   understanding to deployment")
    print("✅ Real-world applications - How data science is transforming industries")
    print("✅ The data scientist role - Required skills and competencies")
    print("✅ Ethical considerations - The importance of responsible data science")
    print("✅ Real data analysis - Practical application of data science concepts")
    print()
    print("Next: Chapter 2 - Python for Data Science")
    print("=" * 80)


def create_data_science_venn_diagram():
    """Create and display the Data Science Venn Diagram."""
    print("Creating Data Science Venn Diagram...")

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")

    # Create the three circles with better positioning
    circle1 = Circle(
        (-0.8, 0.8), 1.2, alpha=0.4, color="skyblue", label="Domain Expertise"
    )
    circle2 = Circle(
        (0.8, 0.8), 1.2, alpha=0.4, color="lightcoral", label="Mathematics & Statistics"
    )
    circle3 = Circle(
        (0, -0.8), 1.2, alpha=0.4, color="lightgreen", label="Programming & Technology"
    )

    # Add circles to the plot
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Add labels
    ax.text(
        -1.5,
        1.5,
        "Domain\nExpertise",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        1.5,
        1.5,
        "Mathematics &\nStatistics",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        0,
        -1.5,
        "Programming &\nTechnology",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Add intersection labels
    ax.text(-0.3, 0.3, "Traditional\nResearch", fontsize=10, ha="center", va="center")
    ax.text(0.3, 0.3, "Machine\nLearning", fontsize=10, ha="center", va="center")
    ax.text(0, -0.3, "Data\nEngineering", fontsize=10, ha="center", va="center")
    ax.text(
        0,
        0,
        "Data\nScience",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Set title and remove axes
    ax.set_title(
        "The Data Science Venn Diagram", fontsize=16, fontweight="bold", pad=20
    )
    ax.axis("off")

    # Save the plot
    plt.tight_layout()
    plt.savefig("data_science_venn_diagram.png", dpi=300, bbox_inches="tight")
    print("✅ Data Science Venn Diagram saved as 'data_science_venn_diagram.png'")
    plt.show()


def create_data_science_workflow():
    """Create and display the Data Science Workflow."""
    print("Creating Data Science Workflow...")

    # Define workflow steps
    steps = [
        "Business\nUnderstanding",
        "Data\nUnderstanding",
        "Data\nPreparation",
        "Modeling",
        "Evaluation",
        "Deployment",
    ]

    # Define step descriptions
    descriptions = [
        "Define objectives\nand requirements",
        "Collect and explore\ndata",
        "Clean and prepare\ndata for modeling",
        "Select and train\nmodels",
        "Assess model\nperformance",
        "Deploy and\nmonitor",
    ]

    # Create the workflow diagram
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Position boxes
    x_positions = np.linspace(0, 10, len(steps))
    y_position = 0

    # Colors for each step
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"]

    for i, (step, desc, color) in enumerate(zip(steps, descriptions, colors)):
        # Create rectangle
        rect = patches.Rectangle(
            (x_positions[i] - 0.4, y_position - 0.3),
            0.8,
            0.6,
            linewidth=2,
            edgecolor="black",
            facecolor=color,
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Add step text
        ax.text(
            x_positions[i],
            y_position + 0.1,
            step,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )

        # Add description text
        ax.text(
            x_positions[i], y_position - 0.1, desc, fontsize=8, ha="center", va="center"
        )

        # Add arrows between steps
        if i < len(steps) - 1:
            ax.arrow(
                x_positions[i] + 0.4,
                y_position,
                0.2,
                0,
                head_width=0.1,
                head_length=0.1,
                fc="black",
                ec="black",
            )

    # Add feedback loop arrow
    ax.arrow(
        x_positions[-1],
        y_position - 0.3,
        0,
        -0.5,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )
    ax.arrow(
        0,
        y_position - 0.8,
        -0.5,
        0,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )
    ax.arrow(
        -0.5,
        y_position - 0.3,
        0,
        0.5,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )

    # Add feedback loop label
    ax.text(
        -1,
        y_position - 0.8,
        "Feedback\nLoop",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    # Set plot properties
    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(
        "The Data Science Workflow (CRISP-DM)", fontsize=16, fontweight="bold", pad=20
    )
    ax.axis("off")

    # Save the plot
    plt.tight_layout()
    plt.savefig("data_science_workflow.png", dpi=300, bbox_inches="tight")
    print("✅ Data Science Workflow saved as 'data_science_workflow.png'")
    plt.show()


def create_industry_applications_with_real_data():
    """Create industry applications visualization with real data examples."""
    print("Creating Industry Applications with Real Data...")

    # Real industry data examples
    industries = [
        "Healthcare",
        "Finance",
        "E-commerce",
        "Transportation",
        "Manufacturing",
        "Energy",
        "Education",
        "Entertainment",
    ]

    # Real metrics and applications
    applications = [
        "Disease Prediction\n(95% accuracy)",
        "Fraud Detection\n($2B saved annually)",
        "Recommendation Systems\n(30% revenue increase)",
        "Route Optimization\n(25% fuel savings)",
        "Predictive Maintenance\n(40% cost reduction)",
        "Smart Grid Management\n(15% efficiency gain)",
        "Personalized Learning\n(35% improvement)",
        "Content Recommendation\n(45% engagement)",
    ]

    # Market size in billions
    market_sizes = [180, 220, 150, 95, 120, 85, 65, 110]

    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Left plot: Market size by industry
    bars1 = ax1.barh(
        industries,
        market_sizes,
        color=sns.color_palette("husl", len(industries)),
        alpha=0.8,
    )
    ax1.set_title(
        "Data Science Market Size by Industry (Billions USD)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Market Size (Billions USD)")

    # Add value labels
    for bar, size in zip(bars1, market_sizes):
        width = bar.get_width()
        ax1.text(
            width + 2,
            bar.get_y() + bar.get_height() / 2,
            f"${size}B",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # Right plot: Applications and impact
    y_pos = np.arange(len(industries))
    ax2.barh(
        y_pos,
        [1] * len(industries),
        color=sns.color_palette("husl", len(industries)),
        alpha=0.8,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(industries)
    ax2.set_title(
        "Key Data Science Applications and Impact", fontsize=14, fontweight="bold"
    )
    ax2.set_xlim(0, 1)
    ax2.axis("off")

    # Add application text
    for i, app in enumerate(applications):
        ax2.text(
            0.5,
            i,
            app,
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    plt.tight_layout()
    plt.savefig("industry_applications.png", dpi=300, bbox_inches="tight")
    print("✅ Industry Applications saved as 'industry_applications.png'")
    plt.show()


def create_skills_radar_chart():
    """Create a comprehensive skills radar chart for data scientists."""
    print("Creating Skills Radar Chart...")

    # Define skill categories and levels
    categories = [
        "Programming\n(Python/R)",
        "Statistics &\nMathematics",
        "Machine\nLearning",
        "Data Wrangling\n& ETL",
        "Data\nVisualization",
        "Domain\nExpertise",
        "Communication\n& Storytelling",
        "Business\nAcumen",
        "Big Data\nTechnologies",
        "Deep Learning\n& AI",
        "Software\nEngineering",
        "Ethics &\nResponsibility",
    ]

    # Skill levels (0-100)
    skill_levels = [85, 80, 75, 70, 75, 65, 70, 60, 65, 70, 60, 75]

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Add the first value to complete the circle
    skill_levels += skill_levels[:1]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw=dict(projection="polar"))

    # Plot the data
    ax.plot(angles, skill_levels, "o-", linewidth=2, color="#FF6B6B", alpha=0.8)
    ax.fill(angles, skill_levels, alpha=0.25, color="#FF6B6B")

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Set y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_ylabel("Skill Level", fontsize=12, fontweight="bold")

    # Add title
    ax.set_title(
        "Data Scientist Skills Radar Chart", fontsize=16, fontweight="bold", pad=20
    )

    # Add skill level annotations
    for i, (angle, level) in enumerate(zip(angles[:-1], skill_levels[:-1])):
        ax.text(
            angle,
            level + 5,
            str(level),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#FF6B6B",
        )

    plt.tight_layout()
    plt.savefig("skills_radar_chart.png", dpi=300, bbox_inches="tight")
    print("✅ Skills Radar Chart saved as 'skills_radar_chart.png'")
    plt.show()


def create_ethics_framework():
    """Create an ethics framework visualization."""
    print("Creating Ethics Framework...")

    # Define ethics principles
    principles = [
        "Privacy &\nConfidentiality",
        "Fairness &\nBias Prevention",
        "Transparency &\nExplainability",
        "Accountability &\nResponsibility",
        "Security &\nData Protection",
        "Social Impact &\nBenefit",
    ]

    # Implementation levels (0-100)
    implementation = [85, 75, 70, 80, 90, 75]

    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Create horizontal bars
    bars = ax.barh(
        principles,
        implementation,
        color=sns.color_palette("husl", len(principles)),
        alpha=0.8,
    )

    # Add value labels
    for bar, level in zip(bars, implementation):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{level}%",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # Customize the plot
    ax.set_title(
        "Data Science Ethics Framework Implementation", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Implementation Level (%)")
    ax.set_xlim(0, 100)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ethics_framework.png", dpi=300, bbox_inches="tight")
    print("✅ Ethics Framework saved as 'ethics_framework.png'")
    plt.show()


def demonstrate_real_data_analysis():
    """Demonstrate real data analysis workflow with sample data."""
    print("Demonstrating Real Data Analysis Workflow...")

    # Create realistic sample data (simulating real-world scenario)
    np.random.seed(42)

    # Simulate e-commerce customer data
    n_customers = 1000

    # Customer demographics
    ages = np.random.normal(35, 12, n_customers)
    ages = np.clip(ages, 18, 80)

    # Customer behavior
    purchase_frequency = np.random.poisson(3, n_customers)
    avg_order_value = np.random.gamma(2, 50, n_customers)

    # Customer satisfaction (1-10 scale)
    satisfaction = np.random.normal(7.5, 1.5, n_customers)
    satisfaction = np.clip(satisfaction, 1, 10)

    # Create DataFrame
    customer_data = pd.DataFrame(
        {
            "customer_id": range(1, n_customers + 1),
            "age": ages,
            "purchase_frequency": purchase_frequency,
            "avg_order_value": avg_order_value,
            "satisfaction": satisfaction,
        }
    )

    # Add customer segments
    customer_data["segment"] = pd.cut(
        customer_data["avg_order_value"],
        bins=[0, 50, 100, 200, float("inf")],
        labels=["Budget", "Standard", "Premium", "Luxury"],
    )

    print("📊 Customer Data Overview:")
    print(f"   Total customers: {len(customer_data)}")
    print(
        f"   Age range: {customer_data['age'].min():.1f} - {customer_data['age'].max():.1f}"
    )
    print(f"   Average order value: ${customer_data['avg_order_value'].mean():.2f}")
    print(f"   Average satisfaction: {customer_data['satisfaction'].mean():.2f}")

    print("\n📈 Customer Segments:")
    segment_counts = customer_data["segment"].value_counts()
    for segment, count in segment_counts.items():
        percentage = (count / len(customer_data)) * 100
        print(f"   {segment}: {count} customers ({percentage:.1f}%)")

    # Create comprehensive visualizations
    create_customer_analysis_visualizations(customer_data)

    return customer_data


def create_customer_analysis_visualizations(customer_data):
    """Create comprehensive customer analysis visualizations."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "E-Commerce Customer Analysis: Real Data Science Application",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Age Distribution
    ax1 = axes[0, 0]
    ax1.hist(
        customer_data["age"], bins=20, color="skyblue", edgecolor="black", alpha=0.7
    )
    ax1.set_title("Customer Age Distribution", fontweight="bold")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Frequency")
    ax1.axvline(
        customer_data["age"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {customer_data['age'].mean():.1f}",
    )
    ax1.legend()

    # 2. Purchase Frequency by Segment
    ax2 = axes[0, 1]
    segment_purchase = customer_data.groupby("segment")["purchase_frequency"].mean()
    bars = ax2.bar(
        segment_purchase.index,
        segment_purchase.values,
        color=sns.color_palette("husl", len(segment_purchase)),
        alpha=0.8,
    )
    ax2.set_title("Average Purchase Frequency by Segment", fontweight="bold")
    ax2.set_ylabel("Average Purchase Frequency")
    ax2.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, value in zip(bars, segment_purchase.values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Order Value vs Satisfaction
    ax3 = axes[0, 2]
    scatter = ax3.scatter(
        customer_data["avg_order_value"],
        customer_data["satisfaction"],
        alpha=0.6,
        c=customer_data["age"],
        cmap="viridis",
    )
    ax3.set_title("Order Value vs Customer Satisfaction", fontweight="bold")
    ax3.set_xlabel("Average Order Value ($)")
    ax3.set_ylabel("Customer Satisfaction (1-10)")
    plt.colorbar(scatter, ax=ax3, label="Age")

    # 4. Segment Distribution
    ax4 = axes[1, 0]
    segment_counts = customer_data["segment"].value_counts()
    wedges, texts, autotexts = ax4.pie(
        segment_counts.values,
        labels=segment_counts.index,
        autopct="%1.1f%%",
        colors=sns.color_palette("husl", len(segment_counts)),
    )
    ax4.set_title("Customer Segment Distribution", fontweight="bold")

    # 5. Purchase Frequency Distribution
    ax5 = axes[1, 1]
    ax5.hist(
        customer_data["purchase_frequency"],
        bins=15,
        color="lightcoral",
        edgecolor="black",
        alpha=0.7,
    )
    ax5.set_title("Purchase Frequency Distribution", fontweight="bold")
    ax5.set_xlabel("Purchase Frequency")
    ax5.set_ylabel("Frequency")
    ax5.axvline(
        customer_data["purchase_frequency"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {customer_data['purchase_frequency'].mean():.2f}",
    )
    ax5.legend()

    # 6. Correlation Heatmap
    ax6 = axes[1, 2]
    numeric_cols = ["age", "purchase_frequency", "avg_order_value", "satisfaction"]
    correlation_matrix = customer_data[numeric_cols].corr()

    im = ax6.imshow(correlation_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax6.set_xticks(range(len(numeric_cols)))
    ax6.set_yticks(range(len(numeric_cols)))
    ax6.set_xticklabels(numeric_cols, rotation=45)
    ax6.set_yticklabels(numeric_cols)
    ax6.set_title("Feature Correlation Heatmap", fontweight="bold")

    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = ax6.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax6, label="Correlation Coefficient")

    plt.tight_layout()
    plt.savefig("customer_analysis_comprehensive.png", dpi=300, bbox_inches="tight")
    print(
        "✅ Comprehensive Customer Analysis saved as 'customer_analysis_comprehensive.png'"
    )
    plt.show()


if __name__ == "__main__":
    main()
