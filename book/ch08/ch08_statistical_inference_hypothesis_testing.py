#!/usr/bin/env python3
"""
Chapter 8: Statistical Inference and Hypothesis Testing
Data Voyage: Making Data-Driven Decisions and Drawing Conclusions with Real Data

This script covers essential statistical inference and hypothesis testing concepts using REAL DATA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
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
    print("✅ Sampling distributions and central limit theorem with real data")
    print("✅ Confidence intervals and estimation on actual datasets")
    print("✅ Hypothesis testing and statistical significance on real data")
    print()
    print("Next: Chapter 9 - Machine Learning Fundamentals")
    print("=" * 80)


def demonstrate_sampling_distributions():
    """Demonstrate sampling distributions and central limit theorem using real data."""
    print("Sampling Distributions and Central Limit Theorem:")
    print("-" * 40)

    # Load real datasets for analysis
    print("Loading real datasets for statistical analysis...")

    iris = load_iris()
    diabetes = load_diabetes()
    breast_cancer = load_breast_cancer()

    # Create DataFrames
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target
    iris_df["species"] = [iris.target_names[i] for i in iris.target]

    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df["target"] = diabetes.target

    breast_cancer_df = pd.DataFrame(
        breast_cancer.data, columns=breast_cancer.feature_names
    )
    breast_cancer_df["target"] = breast_cancer.target
    breast_cancer_df["diagnosis"] = [
        "Malignant" if t == 1 else "Benign" for t in breast_cancer.target
    ]

    print(f"✅ Loaded real datasets:")
    print(f"  • Iris: {iris_df.shape[0]} samples, {iris_df.shape[1]-2} features")
    print(
        f"  • Diabetes: {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]-1} features"
    )
    print(
        f"  • Breast Cancer: {breast_cancer_df.shape[0]} samples, {breast_cancer_df.shape[1]-2} features"
    )
    print()

    # 1. Population Statistics (using real data)
    print("1. POPULATION STATISTICS:")
    print("-" * 30)

    # Use iris sepal length as our main population
    population = iris_df["sepal length (cm)"].values
    population_size = len(population)

    print(f"Population: Iris Sepal Length (cm)")
    print(f"  Size: {population_size} measurements")
    print(f"  Mean: {population.mean():.3f}")
    print(f"  Std: {population.std():.3f}")
    print(f"  Skewness: {stats.skew(population):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(population):.3f}")
    print()

    # 2. Sampling from Real Population
    print("2. SAMPLING FROM REAL POPULATION:")
    print("-" * 35)

    sample_sizes = [10, 30, 100]
    n_samples = 500

    print(
        f"Taking {n_samples} samples of different sizes from iris sepal length data..."
    )
    print()

    for sample_size in sample_sizes:
        # Take multiple samples
        sample_means = []
        sample_stds = []

        for _ in range(n_samples):
            sample = np.random.choice(population, size=sample_size, replace=False)
            sample_means.append(sample.mean())
            sample_stds.append(sample.std())

        # Calculate sampling distribution statistics
        mean_of_means = np.mean(sample_means)
        std_of_means = np.std(sample_means)
        theoretical_std = population.std() / np.sqrt(sample_size)

        print(f"Sample Size: {sample_size}")
        print(f"  Sample means distribution:")
        print(
            f"    Mean of means: {mean_of_means:.3f} (Population mean: {population.mean():.3f})"
        )
        print(f"    Std of means: {std_of_means:.3f}")
        print(f"    Theoretical std: {theoretical_std:.3f}")
        print(
            f"    Central Limit Theorem holds: {'Yes' if abs(std_of_means - theoretical_std) < 0.1 else 'No'}"
        )
        print()

    # 3. Visualize Sampling Distributions
    print("3. VISUALIZING SAMPLING DISTRIBUTIONS:")
    print("-" * 40)

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Sampling Distributions - Real Iris Data", fontsize=16, fontweight="bold"
        )

        # 1. Population distribution
        axes[0, 0].hist(
            population, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].axvline(
            population.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {population.mean():.2f}",
        )
        axes[0, 0].set_title("Population Distribution: Iris Sepal Length")
        axes[0, 0].set_xlabel("Sepal Length (cm)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Sampling distribution for n=10
        sample_means_10 = []
        for _ in range(n_samples):
            sample = np.random.choice(population, size=10, replace=False)
            sample_means_10.append(sample.mean())

        axes[0, 1].hist(
            sample_means_10, bins=30, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        axes[0, 1].axvline(
            np.mean(sample_means_10),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(sample_means_10):.2f}",
        )
        axes[0, 1].set_title("Sampling Distribution: n=10")
        axes[0, 1].set_xlabel("Sample Mean")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Sampling distribution for n=30
        sample_means_30 = []
        for _ in range(n_samples):
            sample = np.random.choice(population, size=30, replace=False)
            sample_means_30.append(sample.mean())

        axes[1, 0].hist(
            sample_means_30, bins=30, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        axes[1, 0].axvline(
            np.mean(sample_means_30),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(sample_means_30):.2f}",
        )
        axes[1, 0].set_title("Sampling Distribution: n=30")
        axes[1, 0].set_xlabel("Sample Mean")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Sampling distribution for n=100
        sample_means_100 = []
        for _ in range(n_samples):
            sample = np.random.choice(population, size=100, replace=False)
            sample_means_100.append(sample.mean())

        axes[1, 1].hist(
            sample_means_100, bins=30, alpha=0.7, color="gold", edgecolor="black"
        )
        axes[1, 1].axvline(
            np.mean(sample_means_100),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(sample_means_100):.2f}",
        )
        axes[1, 1].set_title("Sampling Distribution: n=100")
        axes[1, 1].set_xlabel("Sample Mean")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the visualization
        output_file = "sampling_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")

        plt.show()

    except Exception as e:
        print(f"  ❌ Error creating visualization: {e}")

    print()

    # Store datasets globally for other functions
    global iris_data, diabetes_data, breast_cancer_data
    iris_data = iris_df
    diabetes_data = diabetes_df
    breast_cancer_data = breast_cancer_df

    return iris_df, diabetes_df, breast_cancer_df


def demonstrate_confidence_intervals():
    """Demonstrate confidence intervals using real data."""
    print("Confidence Intervals and Estimation:")
    print("-" * 40)

    if "iris_data" not in globals() or iris_data is None:
        print("❌ No iris data available for confidence intervals")
        return

    df = iris_data

    # 1. Confidence Intervals for Population Mean
    print("1. CONFIDENCE INTERVALS FOR POPULATION MEAN:")
    print("-" * 45)

    # Calculate confidence intervals for different features
    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    confidence_levels = [0.90, 0.95, 0.99]

    for feature in features:
        print(f"\n{feature}:")
        data = df[feature].dropna()
        n = len(data)
        mean = data.mean()
        std = data.std()

        for conf_level in confidence_levels:
            # Calculate critical value (using t-distribution for small samples)
            if n < 30:
                # Use t-distribution
                alpha = 1 - conf_level
                df_t = n - 1
                critical_value = stats.t.ppf(1 - alpha / 2, df_t)
            else:
                # Use normal distribution
                alpha = 1 - conf_level
                critical_value = stats.norm.ppf(1 - alpha / 2)

            # Calculate margin of error
            margin_of_error = critical_value * (std / np.sqrt(n))

            # Calculate confidence interval
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error

            print(f"  {conf_level*100}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"    Margin of error: ±{margin_of_error:.3f}")
            print(f"    Sample size: {n}")

    print()

    # 2. Confidence Intervals by Species
    print("2. CONFIDENCE INTERVALS BY SPECIES:")
    print("-" * 35)

    species_list = df["species"].unique()
    feature = "sepal length (cm)"
    conf_level = 0.95

    print(f"95% Confidence Intervals for {feature} by species:")
    print()

    for species in species_list:
        species_data = df[df["species"] == species][feature].dropna()
        n = len(species_data)
        mean = species_data.mean()
        std = species_data.std()

        # Use t-distribution for small samples
        alpha = 1 - conf_level
        df_t = n - 1
        critical_value = stats.t.ppf(1 - alpha / 2, df_t)

        margin_of_error = critical_value * (std / np.sqrt(n))
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error

        print(f"{species}:")
        print(f"  Sample size: {n}")
        print(f"  Mean: {mean:.3f}")
        print(f"  Std: {std:.3f}")
        print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  Margin of error: ±{margin_of_error:.3f}")
        print()

    # 3. Bootstrap Confidence Intervals
    print("3. BOOTSTRAP CONFIDENCE INTERVALS:")
    print("-" * 35)

    # Use petal length for bootstrap analysis
    feature = "petal length (cm)"
    data = df[feature].dropna()

    print(f"Bootstrap confidence intervals for {feature}:")
    print(f"  Population size: {len(data)}")
    print(f"  Population mean: {data.mean():.3f}")
    print()

    # Perform bootstrap sampling
    n_bootstrap = 1000
    bootstrap_means = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(bootstrap_sample.mean())

    bootstrap_means = np.array(bootstrap_means)

    # Calculate bootstrap confidence intervals
    ci_90 = np.percentile(bootstrap_means, [5, 95])
    ci_95 = np.percentile(bootstrap_means, [2.5, 97.5])
    ci_99 = np.percentile(bootstrap_means, [0.5, 99.5])

    print("Bootstrap Results:")
    print(f"  90% CI: [{ci_90[0]:.3f}, {ci_90[1]:.3f}]")
    print(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
    print(f"  99% CI: [{ci_99[0]:.3f}, {ci_99[1]:.3f}]")
    print(f"  Bootstrap mean: {bootstrap_means.mean():.3f}")
    print(f"  Bootstrap std: {bootstrap_means.std():.3f}")
    print()

    return True


def demonstrate_hypothesis_testing():
    """Demonstrate hypothesis testing using real data."""
    print("Hypothesis Testing and Statistical Significance:")
    print("-" * 50)

    if "iris_data" not in globals() or iris_data is None:
        print("❌ No iris data available for hypothesis testing")
        return

    df = iris_data

    # 1. One-Sample t-Test
    print("1. ONE-SAMPLE T-TEST:")
    print("-" * 25)

    # Test if mean sepal length is different from 5.5 cm
    feature = "sepal length (cm)"
    data = df[feature].dropna()
    hypothesized_mean = 5.5

    print(f"Testing if mean {feature} is different from {hypothesized_mean} cm")
    print(f"  Sample size: {len(data)}")
    print(f"  Sample mean: {data.mean():.3f}")
    print(f"  Sample std: {data.std():.3f}")
    print(f"  Hypothesized mean: {hypothesized_mean}")

    # Perform t-test
    t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean)

    print(f"\nResults:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")
    print()

    # 2. Two-Sample t-Test (Independent)
    print("2. TWO-SAMPLE T-TEST (INDEPENDENT):")
    print("-" * 40)

    # Compare sepal length between setosa and virginica
    feature = "sepal length (cm)"
    setosa_data = df[df["species"] == "setosa"][feature].dropna()
    virginica_data = df[df["species"] == "virginica"][feature].dropna()

    print(f"Comparing {feature} between Setosa and Virginica species")
    print(
        f"  Setosa: n={len(setosa_data)}, mean={setosa_data.mean():.3f}, std={setosa_data.std():.3f}"
    )
    print(
        f"  Virginica: n={len(virginica_data)}, mean={virginica_data.mean():.3f}, std={virginica_data.std():.3f}"
    )

    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(setosa_data, virginica_data)

    print(f"\nResults:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")
    print()

    # 3. Paired t-Test
    print("3. PAIRED T-TEST:")
    print("-" * 20)

    # Compare sepal length vs petal length (paired measurements)
    feature1 = "sepal length (cm)"
    feature2 = "petal length (cm)"

    # Get paired data (remove rows with missing values in either feature)
    paired_data = df[[feature1, feature2]].dropna()
    n_paired = len(paired_data)

    print(f"Comparing {feature1} vs {feature2} (paired measurements)")
    print(f"  Paired samples: {n_paired}")
    print(f"  {feature1} mean: {paired_data[feature1].mean():.3f}")
    print(f"  {feature2} mean: {paired_data[feature2].mean():.3f}")
    print(
        f"  Mean difference: {(paired_data[feature1] - paired_data[feature2]).mean():.3f}"
    )

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(paired_data[feature1], paired_data[feature2])

    print(f"\nResults:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")
    print()

    # 4. ANOVA (Analysis of Variance)
    print("4. ANOVA (ANALYSIS OF VARIANCE):")
    print("-" * 35)

    # Test if sepal length differs across all three species
    feature = "sepal length (cm)"
    species_groups = []
    species_names = []

    for species in df["species"].unique():
        species_data = df[df["species"] == species][feature].dropna()
        species_groups.append(species_data)
        species_names.append(species)
        print(f"  {species}: n={len(species_data)}, mean={species_data.mean():.3f}")

    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*species_groups)

    print(f"\nANOVA Results:")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")
    print()

    # 5. Chi-Square Test for Independence
    print("5. CHI-SQUARE TEST FOR INDEPENDENCE:")
    print("-" * 40)

    # Test if species and size categories are independent
    # Create size categories based on sepal length
    df["size_category"] = pd.cut(
        df["sepal length (cm)"],
        bins=[0, 5.5, 6.5, float("inf")],
        labels=["Small", "Medium", "Large"],
    )

    # Create contingency table
    contingency_table = pd.crosstab(df["species"], df["size_category"], dropna=True)

    print("Contingency Table (Species vs Size Category):")
    print(contingency_table)
    print()

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print("Chi-Square Test Results:")
    print(f"  Chi-square statistic: {chi2_stat:.3f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")
    print()

    # 6. Create hypothesis testing visualization
    print("6. CREATING HYPOTHESIS TESTING VISUALIZATION:")
    print("-" * 45)

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Hypothesis Testing Results - Real Iris Data",
            fontsize=16,
            fontweight="bold",
        )

        # 1. One-sample t-test visualization
        feature = "sepal length (cm)"
        data = df[feature].dropna()
        hypothesized_mean = 5.5

        axes[0, 0].hist(data, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 0].axvline(
            data.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Sample Mean: {data.mean():.2f}",
        )
        axes[0, 0].axvline(
            hypothesized_mean,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Hypothesized Mean: {hypothesized_mean}",
        )
        axes[0, 0].set_title("One-Sample t-Test: Sepal Length")
        axes[0, 0].set_xlabel("Sepal Length (cm)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Two-sample t-test visualization
        feature = "sepal length (cm)"
        setosa_data = df[df["species"] == "setosa"][feature].dropna()
        virginica_data = df[df["species"] == "virginica"][feature].dropna()

        axes[0, 1].hist(
            setosa_data, bins=15, alpha=0.7, label="Setosa", color="lightcoral"
        )
        axes[0, 1].hist(
            virginica_data, bins=15, alpha=0.7, label="Virginica", color="lightgreen"
        )
        axes[0, 1].set_title("Two-Sample t-Test: Setosa vs Virginica")
        axes[0, 1].set_xlabel("Sepal Length (cm)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. ANOVA visualization
        feature = "sepal length (cm)"
        species_data = [
            df[df["species"] == species][feature].dropna()
            for species in df["species"].unique()
        ]
        species_names = list(df["species"].unique())

        axes[1, 0].boxplot(species_data, labels=species_names)
        axes[1, 0].set_title("ANOVA: Sepal Length by Species")
        axes[1, 0].set_ylabel("Sepal Length (cm)")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Chi-square contingency table heatmap
        contingency_table = pd.crosstab(df["species"], df["size_category"], dropna=True)
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1])
        axes[1, 1].set_title("Chi-Square Test: Species vs Size Category")
        axes[1, 1].set_xlabel("Size Category")
        axes[1, 1].set_ylabel("Species")

        plt.tight_layout()

        # Save the visualization
        output_file = "hypothesis_testing.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")

        plt.show()

    except Exception as e:
        print(f"  ❌ Error creating visualization: {e}")

    print()
    return True


if __name__ == "__main__":
    main()
