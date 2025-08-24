#!/usr/bin/env python3
"""
Chapter 8: Statistical Inference and Hypothesis Testing
Data Voyage: Making Data-Driven Decisions and Drawing Conclusions

This script covers essential statistical inference and hypothesis testing concepts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("CHAPTER 8: STATISTICAL INFERENCE AND HYPOTHESIS TESTING")
    print("=" * 80)
    print()

    # Section 8.1: Sampling and Sampling Distributions
    print("8.1 SAMPLING AND SAMPLING DISTRIBUTIONS")
    print("-" * 50)
    demonstrate_sampling_distributions()

    # Section 8.2: Confidence Intervals
    print("\n8.2 CONFIDENCE INTERVALS")
    print("-" * 40)
    demonstrate_confidence_intervals()

    # Section 8.3: Hypothesis Testing
    print("\n8.3 HYPOTHESIS TESTING")
    print("-" * 40)
    demonstrate_hypothesis_testing()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Sampling distributions and central limit theorem")
    print("✅ Confidence intervals and estimation")
    print("✅ Hypothesis testing and statistical significance")
    print()
    print("Next: Chapter 9 - Machine Learning Fundamentals")
    print("=" * 80)


def demonstrate_sampling_distributions():
    """Demonstrate sampling distributions and central limit theorem."""
    print("Sampling Distributions and Central Limit Theorem:")
    print("-" * 40)

    # Create population data
    print("Creating population data...")

    np.random.seed(42)
    population_size = 5000

    # Create different population distributions
    normal_population = np.random.normal(50, 15, population_size)
    exponential_population = np.random.exponential(20, population_size)

    print(f"✅ Created population with {population_size} observations")
    print(f"Population distributions: Normal, Exponential")
    print()

    # Population Statistics
    print("1. POPULATION STATISTICS:")
    print("-" * 30)

    populations = {"Normal": normal_population, "Exponential": exponential_population}

    for name, pop in populations.items():
        print(f"{name} Population:")
        print(f"  Mean: {pop.mean():.2f}")
        print(f"  Std: {pop.std():.2f}")
        print(f"  Skewness: {stats.skew(pop):.3f}")
        print()

    # Sampling from Populations
    print("2. SAMPLING FROM POPULATIONS:")
    print("-" * 30)

    sample_sizes = [10, 30, 100]
    n_samples = 500

    for pop_name, population in populations.items():
        print(f"{pop_name} Population Sampling:")
        print("-" * 25)

        for sample_size in sample_sizes:
            # Take multiple samples
            sample_means = []

            for _ in range(n_samples):
                sample = np.random.choice(population, size=sample_size, replace=False)
                sample_means.append(sample.mean())

            sample_means = np.array(sample_means)

            print(f"  Sample size {sample_size}:")
            print(f"    Mean of sample means: {sample_means.mean():.2f}")
            print(f"    Std of sample means: {sample_means.std():.2f}")
            print(
                f"    Expected std of means: {population.std()/np.sqrt(sample_size):.2f}"
            )
            print()

    # Store data for later use
    global population_data
    population_data = {
        "normal": normal_population,
        "exponential": exponential_population,
    }


def demonstrate_confidence_intervals():
    """Demonstrate confidence intervals and estimation."""
    print("Confidence Intervals and Estimation:")
    print("-" * 40)

    # Use population data from previous section
    if "population_data" not in globals():
        print("Population data not available. Please run sampling distributions first.")
        return

    normal_pop = population_data["normal"]

    print("Working with Normal Population Data:")
    print(f"Population mean: {normal_pop.mean():.2f}")
    print(f"Population std: {normal_pop.std():.2f}")
    print()

    # Single Sample Confidence Interval
    print("1. SINGLE SAMPLE CONFIDENCE INTERVAL:")
    print("-" * 35)

    # Take a sample
    sample_size = 100
    sample = np.random.choice(normal_pop, size=sample_size, replace=False)

    print(f"Sample size: {sample_size}")
    print(f"Sample mean: {sample.mean():.2f}")
    print(f"Sample std: {sample.std():.2f}")
    print()

    # Calculate confidence intervals for different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]

    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)

        # Standard error
        se = sample.std() / np.sqrt(sample_size)

        # Margin of error
        margin_of_error = t_critical * se

        # Confidence interval
        ci_lower = sample.mean() - margin_of_error
        ci_upper = sample.mean() + margin_of_error

        print(f"{conf_level*100:.0f}% Confidence Interval:")
        print(f"  Lower bound: {ci_lower:.2f}")
        print(f"  Upper bound: {ci_upper:.2f}")
        print(f"  Margin of error: {margin_of_error:.2f}")
        print(
            f"  Population mean in CI: {'Yes' if ci_lower <= normal_pop.mean() <= ci_upper else 'No'}"
        )
        print()


def demonstrate_hypothesis_testing():
    """Demonstrate hypothesis testing and statistical significance."""
    print("Hypothesis Testing and Statistical Significance:")
    print("-" * 40)

    # Create sample data for hypothesis testing
    print("Creating sample data for hypothesis testing...")

    np.random.seed(42)

    # Group 1: Control group
    control_group = np.random.normal(100, 15, 50)

    # Group 2: Treatment group (slightly different mean)
    treatment_group = np.random.normal(105, 15, 50)

    print(f"✅ Created 2 groups with 50 observations each")
    print(f"Control group mean: {control_group.mean():.2f}")
    print(f"Treatment group mean: {treatment_group.mean():.2f}")
    print()

    # One-Sample t-test
    print("1. ONE-SAMPLE T-TEST:")
    print("-" * 25)

    # Test if control group mean is different from 100
    hypothesized_mean = 100
    t_stat, p_value = stats.ttest_1samp(control_group, hypothesized_mean)

    print(f"One-sample t-test:")
    print(f"  Hypothesized mean: {hypothesized_mean}")
    print(f"  Sample mean: {control_group.mean():.2f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print()

    # Two-Sample t-test (Independent)
    print("2. TWO-SAMPLE T-TEST (INDEPENDENT):")
    print("-" * 35)

    # Test if control and treatment groups are different
    t_stat, p_value = stats.ttest_ind(control_group, treatment_group)

    print(f"Independent t-test (Control vs Treatment):")
    print(f"  Control mean: {control_group.mean():.2f}")
    print(f"  Treatment mean: {treatment_group.mean():.2f}")
    print(f"  Mean difference: {treatment_group.mean() - control_group.mean():.2f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print()

    # Effect Size (Cohen's d)
    print("3. EFFECT SIZE (COHEN'S D):")
    print("-" * 25)

    pooled_std = np.sqrt(
        (
            (len(control_group) - 1) * control_group.var()
            + (len(treatment_group) - 1) * treatment_group.var()
        )
        / (len(control_group) + len(treatment_group) - 2)
    )
    cohens_d = (treatment_group.mean() - control_group.mean()) / pooled_std

    print(f"Effect Size (Cohen's d): {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        effect_interpretation = "Small"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "Medium"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "Large"
    else:
        effect_interpretation = "Very Large"

    print(f"Effect interpretation: {effect_interpretation}")
    print()

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Group distributions
    plt.subplot(2, 3, 1)
    plt.hist(
        control_group,
        bins=15,
        alpha=0.7,
        label="Control",
        color="skyblue",
        edgecolor="black",
    )
    plt.hist(
        treatment_group,
        bins=15,
        alpha=0.7,
        label="Treatment",
        color="lightgreen",
        edgecolor="black",
    )
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Group Distributions")
    plt.legend()

    # Box plots
    plt.subplot(2, 3, 2)
    data_to_plot = [control_group, treatment_group]
    labels = ["Control", "Treatment"]
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel("Value")
    plt.title("Group Comparisons (Box Plots)")

    # P-value visualization
    plt.subplot(2, 3, 3)
    test_names = ["One-sample\nt-test", "Two-sample\nt-test"]
    p_values = [
        stats.ttest_1samp(control_group, 100)[1],
        stats.ttest_ind(control_group, treatment_group)[1],
    ]
    plt.bar(test_names, p_values, color=["skyblue", "lightgreen"])
    plt.axhline(y=0.05, color="red", linestyle="--", label="α = 0.05")
    plt.ylabel("P-value")
    plt.title("P-values for Tests")
    plt.legend()

    # Effect size
    plt.subplot(2, 3, 4)
    plt.bar(["Effect Size"], [cohens_d], color="lightcoral")
    plt.ylabel("Cohen's d")
    plt.title("Effect Size")

    # Confidence intervals
    plt.subplot(2, 3, 5)
    groups = [control_group, treatment_group]
    group_names = ["Control", "Treatment"]

    means = [g.mean() for g in groups]
    sems = [g.std() / np.sqrt(len(g)) for g in groups]

    plt.errorbar(
        group_names, means, yerr=sems, fmt="o", capsize=5, capthick=2, markersize=8
    )
    plt.ylabel("Value")
    plt.title("Group Means with Standard Errors")

    # Power analysis
    plt.subplot(2, 3, 6)
    sample_sizes = [10, 20, 30, 50, 100]
    powers = []

    for n in sample_sizes:
        power = 0
        for _ in range(100):
            g1 = np.random.normal(100, 15, n)
            g2 = np.random.normal(105, 15, n)
            _, p_val = stats.ttest_ind(g1, g2)
            if p_val < 0.05:
                power += 1
        powers.append(power / 100)

    plt.plot(sample_sizes, powers, "bo-", linewidth=2, markersize=8)
    plt.axhline(y=0.8, color="red", linestyle="--", label="Power = 0.8")
    plt.xlabel("Sample Size per Group")
    plt.ylabel("Power")
    plt.title("Power vs Sample Size")
    plt.legend()

    plt.tight_layout()
    plt.savefig("hypothesis_testing.png", dpi=300, bbox_inches="tight")
    print("✅ Hypothesis testing visualization saved as 'hypothesis_testing.png'")
    plt.close()

    # Final summary
    print("\nHYPOTHESIS TESTING SUMMARY:")
    print("-" * 30)
    print(f"One-sample t-test p-value: {stats.ttest_1samp(control_group, 100)[1]:.4f}")
    print(
        f"Two-sample t-test p-value: {stats.ttest_ind(control_group, treatment_group)[1]:.4f}"
    )
    print(f"Effect size (Cohen's d): {cohens_d:.3f} ({effect_interpretation})")
    print()
    print("Statistical inference and hypothesis testing complete!")
    print(
        "Key concepts demonstrated: sampling, confidence intervals, and hypothesis testing."
    )


if __name__ == "__main__":
    main()
