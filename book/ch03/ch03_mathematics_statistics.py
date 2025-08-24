#!/usr/bin/env python3
"""
Chapter 3: Mathematics and Statistics
Data Voyage: Building the Mathematical Foundation for Data Science

This script covers essential mathematics and statistics concepts with
actual code execution and output examples.
"""

import math
import random
import statistics
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=" * 80)
    print("CHAPTER 3: MATHEMATICS AND STATISTICS")
    print("=" * 80)
    print()
    
    # Section 3.1: Mathematical Foundations
    print("3.1 MATHEMATICAL FOUNDATIONS")
    print("-" * 40)
    demonstrate_mathematical_foundations()
    
    # Section 3.2: Descriptive Statistics
    print("\n3.2 DESCRIPTIVE STATISTICS")
    print("-" * 40)
    demonstrate_descriptive_statistics()
    
    # Section 3.3: Probability Concepts
    print("\n3.3 PROBABILITY CONCEPTS")
    print("-" * 40)
    demonstrate_probability_concepts()
    
    # Section 3.4: Hypothesis Testing
    print("\n3.4 HYPOTHESIS TESTING")
    print("-" * 40)
    demonstrate_hypothesis_testing()
    
    # Section 3.5: Linear Algebra Basics
    print("\n3.5 LINEAR ALGEBRA BASICS")
    print("-" * 40)
    demonstrate_linear_algebra()
    
    # Section 3.6: Correlation and Regression
    print("\n3.6 CORRELATION AND REGRESSION")
    print("-" * 40)
    demonstrate_correlation_and_regression()
    
    # Chapter Summary
    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Mathematical foundations - Algebra, calculus, and mathematical functions")
    print("✅ Descriptive statistics - Central tendency, dispersion, and distributions")
    print("✅ Probability concepts - Basic probability, distributions, and sampling")
    print("✅ Hypothesis testing - Statistical significance and confidence intervals")
    print("✅ Linear algebra - Matrices, vectors, and transformations")
    print("✅ Correlation and regression - Relationships between variables")
    print()
    print("Next: Chapter 4 - Data Types and Sources")
    print("=" * 80)

def demonstrate_mathematical_foundations():
    """Demonstrate fundamental mathematical concepts."""
    print("Basic Mathematical Operations:")
    print("-" * 30)
    
    # Arithmetic operations
    a, b = 15, 7
    print(f"Numbers: a = {a}, b = {b}")
    print(f"Addition: {a} + {b} = {a + b}")
    print(f"Subtraction: {a} - {b} = {a - b}")
    print(f"Multiplication: {a} * {b} = {a * b}")
    print(f"Division: {a} / {b} = {a / b}")
    print(f"Floor division: {a} // {b} = {a // b}")
    print(f"Modulo: {a} % {b} = {a % b}")
    print(f"Exponentiation: {a} ** {b} = {a ** b}")
    print()
    
    print("Mathematical Functions:")
    print("-" * 30)
    
    # Trigonometric functions
    angle_deg = 45
    angle_rad = math.radians(angle_deg)
    print(f"Angle: {angle_deg}° = {angle_rad:.4f} radians")
    print(f"sin({angle_deg}°) = {math.sin(angle_rad):.4f}")
    print(f"cos({angle_deg}°) = {math.cos(angle_rad):.4f}")
    print(f"tan({angle_deg}°) = {math.tan(angle_rad):.4f}")
    print()
    
    # Logarithmic functions
    x = 100
    print(f"Natural log of {x}: ln({x}) = {math.log(x):.4f}")
    print(f"Log base 10 of {x}: log₁₀({x}) = {math.log10(x):.4f}")
    print(f"Log base 2 of {x}: log₂({x}) = {math.log2(x):.4f}")
    print()
    
    # Other mathematical functions
    print(f"Square root of {x}: √{x} = {math.sqrt(x):.4f}")
    print(f"Ceiling of π: ⌈π⌉ = {math.ceil(math.pi)}")
    print(f"Floor of π: ⌊π⌋ = {math.floor(math.pi)}")
    print(f"Absolute value of -15: |-15| = {abs(-15)}")
    print()
    
    print("Mathematical Constants:")
    print("-" * 30)
    print(f"π (pi) = {math.pi:.6f}")
    print(f"e (Euler's number) = {math.e:.6f}")
    print(f"√2 = {math.sqrt(2):.6f}")
    print(f"√3 = {math.sqrt(3):.6f}")
    print()

def demonstrate_descriptive_statistics():
    """Demonstrate descriptive statistics concepts."""
    print("Sample Dataset:")
    print("-" * 30)
    
    # Generate sample data
    np.random.seed(42)  # For reproducible results
    sample_data = np.random.normal(100, 15, 50)  # Normal distribution
    sample_data = np.round(sample_data, 2)
    
    print(f"Sample size: {len(sample_data)}")
    print(f"Data: {sample_data[:10]}...")  # Show first 10 values
    print()
    
    print("Measures of Central Tendency:")
    print("-" * 30)
    
    # Mean
    mean_value = np.mean(sample_data)
    print(f"Mean (arithmetic average): {mean_value:.2f}")
    
    # Median
    median_value = np.median(sample_data)
    print(f"Median (middle value): {median_value:.2f}")
    
    # Mode
    mode_value = statistics.mode(sample_data)
    print(f"Mode (most frequent): {mode_value:.2f}")
    
    # Geometric mean
    geometric_mean = statistics.geometric_mean(sample_data[sample_data > 0])
    print(f"Geometric mean: {geometric_mean:.2f}")
    
    # Harmonic mean
    harmonic_mean = statistics.harmonic_mean(sample_data[sample_data > 0])
    print(f"Harmonic mean: {harmonic_mean:.2f}")
    print()
    
    print("Measures of Dispersion:")
    print("-" * 30)
    
    # Variance and standard deviation
    variance = np.var(sample_data)
    std_dev = np.std(sample_data)
    print(f"Variance: {variance:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")
    
    # Range
    data_range = np.max(sample_data) - np.min(sample_data)
    print(f"Range: {data_range:.2f}")
    
    # Interquartile range
    q1 = np.percentile(sample_data, 25)
    q3 = np.percentile(sample_data, 75)
    iqr = q3 - q1
    print(f"Q1 (25th percentile): {q1:.2f}")
    print(f"Q3 (75th percentile): {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print()
    
    print("Percentiles and Quartiles:")
    print("-" * 30)
    
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        value = np.percentile(sample_data, p)
        print(f"{p}th percentile: {value:.2f}")
    print()
    
    print("Data Distribution Analysis:")
    print("-" * 30)
    
    # Skewness and kurtosis
    skewness = statistics.mean([((x - mean_value) / std_dev) ** 3 for x in sample_data])
    kurtosis = statistics.mean([((x - mean_value) / std_dev) ** 4 for x in sample_data]) - 3
    
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")
    
    # Interpret skewness
    if abs(skewness) < 0.5:
        skew_interpretation = "approximately symmetric"
    elif skewness > 0.5:
        skew_interpretation = "right-skewed (positive skew)"
    else:
        skew_interpretation = "left-skewed (negative skew)"
    
    print(f"Distribution shape: {skew_interpretation}")
    print()
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(sample_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='--', label=f'Median: {median_value:.2f}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Sample Data Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Data distribution histogram saved as 'data_distribution.png'")
    plt.close()

def demonstrate_probability_concepts():
    """Demonstrate probability concepts and distributions."""
    print("Basic Probability Concepts:")
    print("-" * 30)
    
    # Coin flip simulation
    print("Coin Flip Simulation (1000 flips):")
    flips = [random.choice(['H', 'T']) for _ in range(1000)]
    heads_count = flips.count('H')
    tails_count = flips.count('T')
    
    print(f"Heads: {heads_count} (Probability: {heads_count/1000:.3f})")
    print(f"Tails: {tails_count} (Probability: {tails_count/1000:.3f})")
    print()
    
    # Dice roll simulation
    print("Dice Roll Simulation (1000 rolls):")
    rolls = [random.randint(1, 6) for _ in range(1000)]
    roll_counts = Counter(rolls)
    
    print("Roll distribution:")
    for face in sorted(roll_counts.keys()):
        count = roll_counts[face]
        probability = count / 1000
        expected = 1000 / 6
        print(f"  {face}: {count} times (P={probability:.3f}, Expected={expected:.1f})")
    print()
    
    print("Probability Distributions:")
    print("-" * 30)
    
    # Normal distribution
    normal_data = np.random.normal(0, 1, 1000)
    print(f"Normal Distribution (μ=0, σ=1):")
    print(f"  Mean: {np.mean(normal_data):.3f}")
    print(f"  Std Dev: {np.std(normal_data):.3f}")
    print(f"  P(X > 1): {np.mean(normal_data > 1):.3f}")
    print(f"  P(X < -1): {np.mean(normal_data < -1):.3f}")
    print(f"  P(-1 < X < 1): {np.mean((normal_data > -1) & (normal_data < 1)):.3f}")
    print()
    
    # Uniform distribution
    uniform_data = np.random.uniform(0, 10, 1000)
    print(f"Uniform Distribution (0 to 10):")
    print(f"  Mean: {np.mean(uniform_data):.3f}")
    print(f"  Std Dev: {np.std(uniform_data):.3f}")
    print(f"  P(X > 5): {np.mean(uniform_data > 5):.3f}")
    print()
    
    # Exponential distribution
    exp_data = np.random.exponential(2, 1000)
    print(f"Exponential Distribution (λ=2):")
    print(f"  Mean: {np.mean(exp_data):.3f}")
    print(f"  Std Dev: {np.std(exp_data):.3f}")
    print(f"  P(X > 2): {np.mean(exp_data > 2):.3f}")
    print()
    
    # Create distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normal distribution
    axes[0].hist(normal_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Normal Distribution')
    axes[0].set_xlabel('Values')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Uniform distribution
    axes[1].hist(uniform_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title('Uniform Distribution')
    axes[1].set_xlabel('Values')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Exponential distribution
    axes[2].hist(exp_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[2].set_title('Exponential Distribution')
    axes[2].set_xlabel('Values')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_distributions.png', dpi=300, bbox_inches='tight')
    print("✅ Probability distributions plot saved as 'probability_distributions.png'")
    plt.close()

def demonstrate_hypothesis_testing():
    """Demonstrate hypothesis testing concepts."""
    print("Hypothesis Testing Fundamentals:")
    print("-" * 30)
    
    # Generate two sample groups
    np.random.seed(42)
    group1 = np.random.normal(100, 15, 30)  # Control group
    group2 = np.random.normal(110, 15, 30)  # Treatment group
    
    print(f"Group 1 (Control): n={len(group1)}, Mean={np.mean(group1):.2f}, Std={np.std(group1):.2f}")
    print(f"Group 2 (Treatment): n={len(group2)}, Mean={np.mean(group2):.2f}, Std={np.std(group2):.2f}")
    print()
    
    print("Hypothesis Test Setup:")
    print("-" * 30)
    print("H₀ (Null Hypothesis): μ₁ = μ₂ (no difference between groups)")
    print("H₁ (Alternative Hypothesis): μ₁ ≠ μ₂ (groups are different)")
    print("Significance level: α = 0.05")
    print()
    
    # Calculate test statistic (t-test)
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled variance
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    # t-statistic
    t_stat = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    print("Test Results:")
    print("-" * 30)
    print(f"t-statistic: {t_stat:.4f}")
    print(f"Degrees of freedom: {df}")
    print(f"Sample difference: {mean1 - mean2:.2f}")
    print()
    
    # Critical values (approximate for t-distribution)
    critical_value_95 = 2.0  # Approximate for df=58 at α=0.05
    print(f"Critical value (α=0.05, two-tailed): ±{critical_value_95}")
    
    if abs(t_stat) > critical_value_95:
        print("Decision: Reject H₀ (groups are significantly different)")
        print("Conclusion: There is a significant difference between the groups")
    else:
        print("Decision: Fail to reject H₀ (no significant difference)")
        print("Conclusion: No significant difference between the groups")
    print()
    
    print("Confidence Intervals:")
    print("-" * 30)
    
    # 95% confidence interval for the difference
    standard_error = np.sqrt(pooled_var * (1/n1 + 1/n2))
    margin_of_error = critical_value_95 * standard_error
    ci_lower = (mean1 - mean2) - margin_of_error
    ci_upper = (mean1 - mean2) + margin_of_error
    
    print(f"95% Confidence Interval for μ₁ - μ₂:")
    print(f"  [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  Interpretation: We are 95% confident that the true difference")
    print(f"  between population means lies in this interval")
    print()
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    plt.boxplot([group1, group2], labels=['Control', 'Treatment'])
    plt.title('Group Comparison (Box Plot)')
    plt.ylabel('Values')
    plt.grid(True, alpha=0.3)
    
    # Histogram comparison
    plt.subplot(1, 2, 2)
    plt.hist(group1, alpha=0.5, label='Control', bins=15, color='skyblue')
    plt.hist(group2, alpha=0.5, label='Treatment', bins=15, color='lightcoral')
    plt.axvline(mean1, color='blue', linestyle='--', label=f'Control Mean: {mean1:.1f}')
    plt.axvline(mean2, color='red', linestyle='--', label=f'Treatment Mean: {mean2:.1f}')
    plt.title('Group Comparison (Histogram)')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hypothesis_testing.png', dpi=300, bbox_inches='tight')
    print("✅ Hypothesis testing visualization saved as 'hypothesis_testing.png'")
    plt.close()

def demonstrate_linear_algebra():
    """Demonstrate linear algebra concepts."""
    print("Linear Algebra Fundamentals:")
    print("-" * 30)
    
    # Create matrices
    A = np.array([[2, 1], [1, 3]])
    B = np.array([[1, 2], [3, 4]])
    
    print("Matrix A:")
    print(A)
    print()
    print("Matrix B:")
    print(B)
    print()
    
    print("Matrix Operations:")
    print("-" * 30)
    
    # Addition
    C = A + B
    print("A + B:")
    print(C)
    print()
    
    # Multiplication
    D = A @ B  # Matrix multiplication
    print("A × B (Matrix multiplication):")
    print(D)
    print()
    
    # Element-wise multiplication
    E = A * B
    print("A ⊙ B (Element-wise multiplication):")
    print(E)
    print()
    
    print("Matrix Properties:")
    print("-" * 30)
    
    # Determinant
    det_A = np.linalg.det(A)
    det_B = np.linalg.det(B)
    print(f"Determinant of A: |A| = {det_A:.2f}")
    print(f"Determinant of B: |B| = {det_B:.2f}")
    print()
    
    # Inverse
    try:
        A_inv = np.linalg.inv(A)
        print("Inverse of A (A⁻¹):")
        print(A_inv)
        print()
        
        # Verify inverse
        identity_check = A @ A_inv
        print("A × A⁻¹ (should be identity matrix):")
        print(identity_check)
        print()
    except np.linalg.LinAlgError:
        print("Matrix A is not invertible")
        print()
    
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(A)
    print("Eigenvalues of A:")
    print(eigenvals)
    print()
    print("Eigenvectors of A:")
    print(eigenvecs)
    print()
    
    # Transpose
    A_T = A.T
    print("Transpose of A (Aᵀ):")
    print(A_T)
    print()
    
    print("Vector Operations:")
    print("-" * 30)
    
    # Create vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    print()
    
    # Vector operations
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 · v2 (dot product) = {np.dot(v1, v2)}")
    print(f"||v1|| (magnitude) = {np.linalg.norm(v1):.4f}")
    print(f"||v2|| (magnitude) = {np.linalg.norm(v2):.4f}")
    print()
    
    # Angle between vectors
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    angle_deg = np.degrees(angle_rad)
    print(f"Angle between v1 and v2: {angle_deg:.2f}°")
    print()
    
    # Create matrix visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Matrix A heatmap
    im1 = axes[0, 0].imshow(A, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Matrix A')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, f'{A[i, j]:.1f}', ha='center', va='center', color='white', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Matrix B heatmap
    im2 = axes[0, 1].imshow(B, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Matrix B')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, f'{B[i, j]:.1f}', ha='center', va='center', color='white', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Matrix multiplication result
    im3 = axes[1, 0].imshow(D, cmap='coolwarm', aspect='auto')
    axes[1, 0].set_title('A × B')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f'{D[i, j]:.1f}', ha='center', va='center', color='white', fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Vector visualization
    axes[1, 1].quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
    axes[1, 1].quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', label='v2')
    axes[1, 1].set_xlim(-1, 7)
    axes[1, 1].set_ylim(-1, 7)
    axes[1, 1].set_title('Vectors v1 and v2')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('linear_algebra.png', dpi=300, bbox_inches='tight')
    print("✅ Linear algebra visualization saved as 'linear_algebra.png'")
    plt.close()

def demonstrate_correlation_and_regression():
    """Demonstrate correlation and regression analysis."""
    print("Correlation Analysis:")
    print("-" * 30)
    
    # Generate correlated data
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    # Create y with some correlation to x
    y_positive = 0.7 * x + np.random.normal(0, 0.5, 100)  # Positive correlation
    y_negative = -0.6 * x + np.random.normal(0, 0.5, 100)  # Negative correlation
    y_no_corr = np.random.normal(0, 1, 100)  # No correlation
    
    print("Correlation Coefficients:")
    print("-" * 30)
    
    # Calculate correlations
    corr_positive = np.corrcoef(x, y_positive)[0, 1]
    corr_negative = np.corrcoef(x, y_negative)[0, 1]
    corr_none = np.corrcoef(x, y_no_corr)[0, 1]
    
    print(f"X vs Y (positive correlation): r = {corr_positive:.3f}")
    print(f"X vs Y (negative correlation): r = {corr_negative:.3f}")
    print(f"X vs Y (no correlation): r = {corr_negative:.3f}")
    print()
    
    # Interpret correlations
    def interpret_correlation(r):
        if abs(r) < 0.1:
            return "negligible"
        elif abs(r) < 0.3:
            return "weak"
        elif abs(r) < 0.5:
            return "moderate"
        elif abs(r) < 0.7:
            return "strong"
        else:
            return "very strong"
    
    print("Correlation Interpretation:")
    print(f"  Positive correlation: {interpret_correlation(corr_positive)} ({corr_positive:.3f})")
    print(f"  Negative correlation: {interpret_correlation(abs(corr_negative))} ({corr_negative:.3f})")
    print(f"  No correlation: {interpret_correlation(abs(corr_none))} ({corr_none:.3f})")
    print()
    
    print("Linear Regression Analysis:")
    print("-" * 30)
    
    # Perform linear regression
    # For positive correlation data
    n = len(x)
    x_mean, y_mean = np.mean(x), np.mean(y_positive)
    
    # Calculate regression coefficients
    numerator = np.sum((x - x_mean) * (y_positive - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    print(f"Regression Equation: y = {slope:.3f}x + {intercept:.3f}")
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print()
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y_positive - y_pred) ** 2)
    ss_tot = np.sum((y_positive - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"R-squared: {r_squared:.3f}")
    print(f"Interpretation: {r_squared*100:.1f}% of the variance in y is explained by x")
    print()
    
    # Prediction example
    x_new = 2.0
    y_pred_new = slope * x_new + intercept
    print(f"Prediction Example:")
    print(f"  For x = {x_new}, predicted y = {y_pred_new:.3f}")
    print()
    
    # Create correlation and regression plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Positive correlation
    axes[0, 0].scatter(x, y_positive, alpha=0.6, color='blue')
    axes[0, 0].plot(x, slope * x + intercept, color='red', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')
    axes[0, 0].set_title(f'Positive Correlation (r = {corr_positive:.3f})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Negative correlation
    axes[0, 1].scatter(x, y_negative, alpha=0.6, color='green')
    axes[0, 1].set_title(f'Negative Correlation (r = {corr_negative:.3f})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].grid(True, alpha=0.3)
    
    # No correlation
    axes[1, 0].scatter(x, y_no_corr, alpha=0.6, color='orange')
    axes[1, 0].set_title(f'No Correlation (r = {corr_none:.3f})')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_positive - y_pred
    axes[1, 1].scatter(y_pred, residuals, alpha=0.6, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_title('Residuals Plot')
    axes[1, 1].set_xlabel('Predicted Y')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_regression.png', dpi=300, bbox_inches='tight')
    print("✅ Correlation and regression plots saved as 'correlation_regression.png'")
    plt.close()

if __name__ == "__main__":
    main()
