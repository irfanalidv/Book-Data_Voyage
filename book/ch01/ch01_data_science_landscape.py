#!/usr/bin/env python3
"""
Chapter 1: The Data Science Landscape
Data Voyage: Mapping the Path to Discovery in Data Science

This script demonstrates key concepts from Chapter 1 with actual code execution
and output examples.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.patches as patches


def main():
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

    # Section 1.3: Real-World Applications
    print("\n1.3 REAL-WORLD APPLICATIONS")
    print("-" * 40)
    print("Data science is transforming industries across the globe:")
    print()

    create_industry_applications()

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
    print()
    print("Next: Chapter 2 - Python for Data Science")
    print("=" * 80)


def create_data_science_venn_diagram():
    """Create and display the Data Science Venn Diagram."""
    print("Creating Data Science Venn Diagram...")

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

    # Create the three circles
    circle1 = Circle(
        (-0.5, 0.5), 1, alpha=0.3, color="skyblue", label="Domain Expertise"
    )
    circle2 = Circle(
        (0.5, 0.5), 1, alpha=0.3, color="lightcoral", label="Mathematics & Statistics"
    )
    circle3 = Circle(
        (0, -0.5), 1, alpha=0.3, color="lightgreen", label="Programming & Technology"
    )

    # Add circles to the plot
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Add labels
    ax.text(
        -0.5,
        0.5,
        "Domain\nExpertise",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.5,
        "Mathematics &\nStatistics",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0,
        -0.5,
        "Programming &\nTechnology",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0,
        0,
        "Data Science",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    # Customize the plot
    ax.set_title(
        "The Data Science Venn Diagram", fontsize=18, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("data_science_venn_diagram.png", dpi=300, bbox_inches="tight")
    print("✅ Venn diagram saved as 'data_science_venn_diagram.png'")
    plt.close()


def create_data_science_workflow():
    """Create and display the Data Science Workflow diagram."""
    print("Creating Data Science Workflow diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Define the workflow steps
    steps = [
        "Business\nUnderstanding",
        "Data\nUnderstanding",
        "Data\nPreparation",
        "Modeling",
        "Evaluation",
        "Deployment",
    ]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    # Create boxes for each step
    for i, (step, color) in enumerate(zip(steps, colors)):
        x = i * 2 - 5
        rect = patches.Rectangle(
            (x - 0.8, -0.5),
            1.6,
            1,
            linewidth=2,
            edgecolor="black",
            facecolor=color,
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(x, 0, step, ha="center", va="center", fontsize=10, fontweight="bold")

        # Add arrows between steps
        if i < len(steps) - 1:
            ax.arrow(
                x + 0.8,
                0,
                0.4,
                0,
                head_width=0.1,
                head_length=0.1,
                fc="black",
                ec="black",
                linewidth=2,
            )

    # Add feedback loop arrow
    ax.arrow(
        5,
        -0.5,
        -0.5,
        -1,
        head_width=0.1,
        head_length=0.1,
        fc="red",
        ec="red",
        linewidth=2,
        linestyle="--",
    )
    ax.arrow(
        -5,
        -1.5,
        0.5,
        1,
        head_width=0.1,
        head_length=0.1,
        fc="red",
        ec="red",
        linewidth=2,
        linestyle="--",
    )
    ax.text(
        0,
        -2,
        "Iterative Process",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="red",
    )

    # Customize the plot
    ax.set_title(
        "The Data Science Workflow (CRISP-DM)", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 2)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("data_science_workflow.png", dpi=300, bbox_inches="tight")
    print("✅ Workflow diagram saved as 'data_science_workflow.png'")
    plt.close()


def create_industry_applications():
    """Create and display industry applications visualization."""
    print("Creating industry applications visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Define industries and their applications
    industries = {
        "Healthcare": [
            "Disease Prediction",
            "Drug Discovery",
            "Medical Imaging",
            "Patient Care",
        ],
        "Finance": [
            "Fraud Detection",
            "Risk Assessment",
            "Algorithmic Trading",
            "Credit Scoring",
        ],
        "E-commerce": [
            "Recommendation Systems",
            "Demand Forecasting",
            "Customer Segmentation",
            "Price Optimization",
        ],
        "Transportation": [
            "Route Optimization",
            "Autonomous Vehicles",
            "Traffic Prediction",
            "Maintenance Scheduling",
        ],
        "Manufacturing": [
            "Predictive Maintenance",
            "Quality Control",
            "Supply Chain Optimization",
            "Energy Efficiency",
        ],
    }

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    # Create industry boxes
    y_pos = 4
    for i, (industry, applications) in enumerate(industries.items()):
        x_pos = i * 2.5 - 5

        # Main industry box
        rect = patches.Rectangle(
            (x_pos - 1, y_pos - 0.5),
            2,
            1,
            linewidth=2,
            edgecolor="black",
            facecolor=colors[i],
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            x_pos,
            y_pos,
            industry,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        # Application boxes
        for j, app in enumerate(applications):
            app_y = y_pos - 1.5 - j * 0.4
            app_rect = patches.Rectangle(
                (x_pos - 0.8, app_y - 0.15),
                1.6,
                0.3,
                linewidth=1,
                edgecolor="black",
                facecolor=colors[i],
                alpha=0.6,
            )
            ax.add_patch(app_rect)
            ax.text(x_pos, app_y, app, ha="center", va="center", fontsize=8)

            # Connect industry to applications
            ax.arrow(
                x_pos,
                y_pos - 0.5,
                0,
                -0.3,
                head_width=0.05,
                head_length=0.05,
                fc="black",
                ec="black",
                linewidth=1,
            )

    # Customize the plot
    ax.set_title(
        "Data Science Applications Across Industries",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 5)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("industry_applications.png", dpi=300, bbox_inches="tight")
    print("✅ Industry applications diagram saved as 'industry_applications.png'")
    plt.close()


def create_skills_radar_chart():
    """Create and display data scientist skills radar chart."""
    print("Creating skills radar chart...")

    # Create a skills radar chart
    categories = [
        "Programming",
        "Statistics",
        "Machine Learning",
        "Domain Knowledge",
        "Communication",
        "Business Acumen",
        "Data Engineering",
        "Visualization",
    ]

    # Sample skill levels (1-10 scale)
    values = [8, 9, 8, 7, 6, 7, 6, 8]

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Make the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Add values to the end to close the plot
    values += values[:1]

    # Plot the data
    ax.plot(angles, values, "o-", linewidth=2, color="#4ECDC4")
    ax.fill(angles, values, alpha=0.25, color="#4ECDC4")

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    # Set the y-axis limits
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=10)

    # Add title
    ax.set_title(
        "Data Scientist Skills Profile", fontsize=16, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.savefig("skills_radar_chart.png", dpi=300, bbox_inches="tight")
    print("✅ Skills radar chart saved as 'skills_radar_chart.png'")
    plt.close()


def create_ethics_framework():
    """Create and display ethics framework visualization."""
    print("Creating ethics framework visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Define ethical principles
    principles = ["Privacy", "Fairness", "Transparency", "Accountability", "Security"]
    importance = [9, 9, 8, 8, 9]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    # Create horizontal bar chart
    y_pos = np.arange(len(principles))
    bars = ax.barh(
        y_pos, importance, color=colors, alpha=0.8, edgecolor="black", linewidth=1
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val + 0.1, i, f"{val}/10", va="center", fontweight="bold")

    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(principles, fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Importance Level (1-10)", fontsize=12)
    ax.set_title(
        "Ethical Principles in Data Science", fontsize=16, fontweight="bold", pad=20
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("ethics_framework.png", dpi=300, bbox_inches="tight")
    print("✅ Ethics framework chart saved as 'ethics_framework.png'")
    plt.close()


if __name__ == "__main__":
    main()
