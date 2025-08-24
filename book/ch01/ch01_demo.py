#!/usr/bin/env python3
"""
Chapter 1 Demo: Basic Data Science Concepts
This script demonstrates fundamental data science concepts with real examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def demo_data_types():
    """Demonstrate different data types in data science."""
    print("=" * 60)
    print("DEMO: DATA TYPES IN DATA SCIENCE")
    print("=" * 60)
    
    # Numerical data
    print("\n1. NUMERICAL DATA:")
    print("-" * 30)
    
    # Sample sales data
    sales_data = [1250, 1890, 2100, 1750, 2300, 1950, 2800, 1650]
    print(f"Sales data: {sales_data}")
    print(f"Mean sales: ${np.mean(sales_data):.2f}")
    print(f"Median sales: ${np.median(sales_data):.2f}")
    print(f"Standard deviation: ${np.std(sales_data):.2f}")
    
    # Categorical data
    print("\n2. CATEGORICAL DATA:")
    print("-" * 30)
    
    product_categories = ['Electronics', 'Clothing', 'Books', 'Electronics', 'Clothing', 'Books']
    print(f"Product categories: {product_categories}")
    
    # Count categories
    from collections import Counter
    category_counts = Counter(product_categories)
    print("Category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    # Time series data
    print("\n3. TIME SERIES DATA:")
    print("-" * 30)
    
    # Generate sample time series
    dates = [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)]
    daily_sales = [1200, 1350, 1100, 1600, 1400, 1800, 2000]
    
    print("Daily sales for the past week:")
    for date, sales in zip(dates, daily_sales):
        print(f"  {date.strftime('%Y-%m-%d')}: ${sales}")
    
    return sales_data, product_categories, dates, daily_sales

def demo_data_visualization(sales_data, dates, daily_sales):
    """Demonstrate basic data visualization."""
    print("\n" + "=" * 60)
    print("DEMO: DATA VISUALIZATION")
    print("=" * 60)
    
    # Create a simple line chart
    plt.figure(figsize=(10, 6))
    plt.plot(dates, daily_sales, marker='o', linewidth=2, markersize=8)
    plt.title('Daily Sales Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('daily_sales_trend.png', dpi=300, bbox_inches='tight')
    print("✅ Daily sales trend chart saved as 'daily_sales_trend.png'")
    plt.close()
    
    # Create a histogram
    plt.figure(figsize=(8, 6))
    plt.hist(sales_data, bins=5, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Sales Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Sales Amount ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sales_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Sales distribution histogram saved as 'sales_distribution.png'")
    plt.close()

def demo_basic_statistics(sales_data):
    """Demonstrate basic statistical concepts."""
    print("\n" + "=" * 60)
    print("DEMO: BASIC STATISTICS")
    print("=" * 60)
    
    print("Sales Data Analysis:")
    print("-" * 20)
    
    # Basic statistics
    mean_sales = np.mean(sales_data)
    median_sales = np.median(sales_data)
    std_sales = np.std(sales_data)
    min_sales = np.min(sales_data)
    max_sales = np.max(sales_data)
    
    print(f"Sample size: {len(sales_data)}")
    print(f"Mean: ${mean_sales:.2f}")
    print(f"Median: ${median_sales:.2f}")
    print(f"Standard deviation: ${std_sales:.2f}")
    print(f"Range: ${min_sales} - ${max_sales}")
    print(f"Variance: ${np.var(sales_data):.2f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    print(f"  25th percentile: ${np.percentile(sales_data, 25):.2f}")
    print(f"  75th percentile: ${np.percentile(sales_data, 75):.2f}")
    
    # Outlier detection (simple IQR method)
    Q1 = np.percentile(sales_data, 25)
    Q3 = np.percentile(sales_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = [x for x in sales_data if x < lower_bound or x > upper_bound]
    print(f"\nOutlier Analysis (IQR method):")
    print(f"  Lower bound: ${lower_bound:.2f}")
    print(f"  Upper bound: ${upper_bound:.2f}")
    print(f"  Outliers found: {outliers if outliers else 'None'}")
    
    return mean_sales, std_sales

def demo_data_frame_creation():
    """Demonstrate creating and working with pandas DataFrames."""
    print("\n" + "=" * 60)
    print("DEMO: PANDAS DATAFRAME")
    print("=" * 60)
    
    # Create sample data
    data = {
        'Date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'Product': ['Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone', 'Tablet', 'Laptop'],
        'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'Price': [1200, 800, 500, 1200, 800, 500, 1200, 800, 500, 1200],
        'Quantity': [2, 5, 8, 1, 3, 6, 4, 7, 2, 3],
        'Revenue': [2400, 4000, 4000, 1200, 2400, 3000, 4800, 5600, 1000, 3600]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print("Sample Sales Data:")
    print("-" * 20)
    print(df)
    
    print(f"\nDataFrame Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Data types:\n{df.dtypes}")
    
    print(f"\nSummary Statistics:")
    print("-" * 20)
    print(df.describe())
    
    print(f"\nCategory Analysis:")
    print("-" * 20)
    print(df.groupby('Product')['Revenue'].sum().sort_values(ascending=False))
    
    return df

def demo_correlation_analysis(df):
    """Demonstrate correlation analysis."""
    print("\n" + "=" * 60)
    print("DEMO: CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Calculate correlations
    numeric_cols = ['Price', 'Quantity', 'Revenue']
    correlation_matrix = df[numeric_cols].corr()
    
    print("Correlation Matrix:")
    print("-" * 20)
    print(correlation_matrix)
    
    # Create correlation heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
    
    # Add correlation values as text
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")
    plt.close()
    
    # Interpret correlations
    print(f"\nCorrelation Interpretation:")
    print("-" * 30)
    price_quantity_corr = correlation_matrix.loc['Price', 'Quantity']
    price_revenue_corr = correlation_matrix.loc['Price', 'Revenue']
    quantity_revenue_corr = correlation_matrix.loc['Quantity', 'Revenue']
    
    print(f"Price vs Quantity: {price_quantity_corr:.3f}")
    print(f"  {'Strong' if abs(price_quantity_corr) > 0.7 else 'Moderate' if abs(price_quantity_corr) > 0.3 else 'Weak'} {'positive' if price_quantity_corr > 0 else 'negative'} correlation")
    
    print(f"Price vs Revenue: {price_revenue_corr:.3f}")
    print(f"  {'Strong' if abs(price_revenue_corr) > 0.7 else 'Moderate' if abs(price_revenue_corr) > 0.3 else 'Weak'} {'positive' if price_revenue_corr > 0 else 'negative'} correlation")
    
    print(f"Quantity vs Revenue: {quantity_revenue_corr:.3f}")
    print(f"  {'Strong' if abs(quantity_revenue_corr) > 0.7 else 'Moderate' if abs(quantity_revenue_corr) > 0.3 else 'Weak'} {'positive' if quantity_revenue_corr > 0 else 'negative'} correlation")

def main():
    """Main function to run all demonstrations."""
    print("🚀 CHAPTER 1 DEMONSTRATION: BASIC DATA SCIENCE CONCEPTS")
    print("=" * 80)
    print("This script demonstrates fundamental data science concepts with real examples.")
    print("=" * 80)
    
    try:
        # Demo 1: Data Types
        sales_data, product_categories, dates, daily_sales = demo_data_types()
        
        # Demo 2: Data Visualization
        demo_data_visualization(sales_data, dates, daily_sales)
        
        # Demo 3: Basic Statistics
        mean_sales, std_sales = demo_basic_statistics(sales_data)
        
        # Demo 4: Pandas DataFrame
        df = demo_data_frame_creation()
        
        # Demo 5: Correlation Analysis
        demo_correlation_analysis(df)
        
        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("Generated files:")
        print("✅ daily_sales_trend.png - Line chart of sales over time")
        print("✅ sales_distribution.png - Histogram of sales distribution")
        print("✅ correlation_heatmap.png - Correlation analysis visualization")
        print("\nKey concepts demonstrated:")
        print("• Different data types (numerical, categorical, time series)")
        print("• Basic statistical analysis (mean, median, standard deviation)")
        print("• Data visualization (charts and graphs)")
        print("• Pandas DataFrame operations")
        print("• Correlation analysis and interpretation")
        print("\nNext: Continue with Chapter 1 concepts or move to Chapter 2!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Make sure you have all required packages installed:")
        print("pip install numpy pandas matplotlib")

if __name__ == "__main__":
    main()
