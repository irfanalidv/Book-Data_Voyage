#!/usr/bin/env python3
"""
Chapter 3 Practical Examples: Mathematics and Statistics for Data Science
Real-world applications and exercises to reinforce mathematical concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats

def main():
    print("🚀 CHAPTER 3 PRACTICAL EXAMPLES: MATHEMATICS AND STATISTICS")
    print("=" * 80)
    print("This script demonstrates practical mathematics and statistics applications.")
    print("=" * 80)
    
    # Example 1: Financial Data Analysis
    print("\n1. FINANCIAL DATA ANALYSIS")
    print("-" * 50)
    demonstrate_financial_analysis()
    
    # Example 2: Quality Control and Process Monitoring
    print("\n2. QUALITY CONTROL AND PROCESS MONITORING")
    print("-" * 50)
    demonstrate_quality_control()
    
    # Example 3: A/B Testing and Statistical Significance
    print("\n3. A/B TESTING AND STATISTICAL SIGNIFICANCE")
    print("-" * 50)
    demonstrate_ab_testing()
    
    print("\n" + "=" * 80)
    print("🎉 PRACTICAL EXAMPLES COMPLETE!")
    print("=" * 80)

def demonstrate_financial_analysis():
    """Demonstrate financial data analysis using mathematics and statistics."""
    print("Analyzing stock price data and calculating financial metrics...")
    
    # Generate sample stock price data
    np.random.seed(42)
    initial_price = 100
    daily_returns = np.random.normal(0.001, 0.02, 100)
    prices = [initial_price]
    
    for return_rate in daily_returns:
        new_price = prices[-1] * (1 + return_rate)
        prices.append(new_price)
    
    prices = np.array(prices)
    print(f"✅ Generated {len(prices)} daily stock prices")
    print(f"Starting price: ${prices[0]:.2f}")
    print(f"Ending price: ${prices[-1]:.2f}")
    print()
    
    # Calculate financial metrics
    total_return = (prices[-1] - prices[0]) / prices[0]
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns) * np.sqrt(252)
    
    print("Financial Metrics:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Annualized Volatility: {volatility:.2%}")
    print()
    
    # Create financial analysis plot
    plt.figure(figsize=(10, 6))
    plt.plot(prices, color='blue', linewidth=2)
    plt.title('Stock Price Over Time')
    plt.xlabel('Trading Day')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('financial_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Financial analysis plot saved as 'financial_analysis.png'")
    plt.close()

def demonstrate_quality_control():
    """Demonstrate quality control and statistical process control."""
    print("Implementing statistical process control for manufacturing quality...")
    
    # Generate sample manufacturing data
    np.random.seed(42)
    target_dimension = 10.0
    process_std = 0.1
    
    # Generate measurements from different batches
    batches = []
    for batch in range(5):
        batch_mean = target_dimension + np.random.normal(0, 0.02)
        batch_data = np.random.normal(batch_mean, process_std, 30)
        batches.append(batch_data)
    
    all_data = np.concatenate(batches)
    print(f"✅ Generated {len(all_data)} measurements from 5 batches")
    print()
    
    # Calculate control limits
    overall_mean = np.mean(all_data)
    overall_std = np.std(all_data, ddof=1)
    
    ucl = overall_mean + 3 * overall_std
    lcl = overall_mean - 3 * overall_std
    
    print("Control Chart Results:")
    print(f"  Overall mean: {overall_mean:.3f} mm")
    print(f"  UCL: {ucl:.3f} mm")
    print(f"  LCL: {lcl:.3f} mm")
    print()
    
    # Create control chart
    plt.figure(figsize=(10, 6))
    plt.plot(all_data, 'bo-', markersize=4)
    plt.axhline(overall_mean, color='green', linestyle='-', label=f'Mean: {overall_mean:.3f}')
    plt.axhline(ucl, color='red', linestyle='--', label=f'UCL: {ucl:.3f}')
    plt.axhline(lcl, color='red', linestyle='--', label=f'LCL: {lcl:.3f}')
    plt.title('Control Chart for Manufacturing Process')
    plt.xlabel('Measurement Number')
    plt.ylabel('Dimension (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quality_control.png', dpi=300, bbox_inches='tight')
    print("✅ Quality control chart saved as 'quality_control.png'")
    plt.close()

def demonstrate_ab_testing():
    """Demonstrate A/B testing and statistical significance testing."""
    print("Conducting A/B test analysis for website conversion rates...")
    
    # Generate A/B test data
    np.random.seed(42)
    
    # Control group (A)
    n_a = 1000
    conversion_rate_a = 0.12
    conversions_a = np.random.binomial(n_a, conversion_rate_a)
    
    # Treatment group (B)
    n_b = 1000
    conversion_rate_b = 0.15
    conversions_b = np.random.binomial(n_b, conversion_rate_b)
    
    print(f"✅ Generated A/B test data:")
    print(f"  Control Group (A): {conversions_a} conversions out of {n_a} visitors")
    print(f"  Treatment Group (B): {conversions_b} conversions out of {n_b} visitors")
    print()
    
    # Calculate conversion rates
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    
    print("Conversion Rate Analysis:")
    print(f"  Control Group (A): {p_a:.3%}")
    print(f"  Treatment Group (B): {p_b:.3%}")
    print(f"  Improvement: {((p_b - p_a) / p_a):.1%}")
    print()
    
    # Statistical significance testing
    pooled_p = (conversions_a + conversions_b) / (n_a + n_b)
    standard_error = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b))
    z_stat = (p_b - p_a) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print("Statistical Significance:")
    print(f"  Z-statistic: {z_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    
    alpha = 0.05
    if p_value < alpha:
        decision = "Reject null hypothesis - Significant difference"
    else:
        decision = "Fail to reject null hypothesis - No significant difference"
    
    print(f"  Decision: {decision}")
    print()
    
    # Create A/B test visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Conversion rates comparison
    groups = ['Control (A)', 'Treatment (B)']
    rates = [p_a, p_b]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax1.bar(groups, rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Conversion Rate Comparison')
    ax1.set_ylabel('Conversion Rate')
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Statistical significance indicator
    significance_color = 'green' if p_value < alpha else 'red'
    ax2.text(0.5, 0.5, f'P-value: {p_value:.4f}\n{decision}', 
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor=significance_color, alpha=0.3))
    ax2.set_title('Statistical Significance')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('ab_testing.png', dpi=300, bbox_inches='tight')
    print("✅ A/B testing visualization saved as 'ab_testing.png'")
    plt.close()

if __name__ == "__main__":
    main()
