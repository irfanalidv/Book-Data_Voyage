#!/usr/bin/env python3
"""
Chapter 2: Python for Data Science
Data Voyage: Building the Foundation for Data Analysis

This script covers Python fundamentals essential for data science with
real data examples, comprehensive analysis, and professional visualizations.
"""

import sys
import platform
import time
from datetime import datetime
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set up for professional plotting
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def main():
    """Main function to run Chapter 2 demonstrations."""
    print("=" * 80)
    print("CHAPTER 2: PYTHON FOR DATA SCIENCE")
    print("=" * 80)
    print()

    # Section 2.1: Python Basics
    print("2.1 PYTHON BASICS")
    print("-" * 40)
    demonstrate_python_basics()

    # Section 2.2: Data Structures
    print("\n2.2 DATA STRUCTURES")
    print("-" * 40)
    demonstrate_data_structures()

    # Section 2.3: Control Flow
    print("\n2.3 CONTROL FLOW")
    print("-" * 40)
    demonstrate_control_flow()

    # Section 2.4: Functions and OOP
    print("\n2.4 FUNCTIONS AND OBJECT-ORIENTED PROGRAMMING")
    print("-" * 40)
    demonstrate_functions_and_oop()

    # Section 2.5: File I/O and Error Handling
    print("\n2.5 FILE I/O AND ERROR HANDLING")
    print("-" * 40)
    demonstrate_file_io_and_error_handling()

    # Section 2.6: Python Packages for Data Science
    print("\n2.6 PYTHON PACKAGES FOR DATA SCIENCE")
    print("-" * 40)
    demonstrate_data_science_packages()

    # Section 2.7: Real Data Analysis Example
    print("\n2.7 REAL DATA ANALYSIS EXAMPLE")
    print("-" * 40)
    demonstrate_real_data_analysis()

    # Chapter Summary
    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Python fundamentals - Variables, data types, and basic operations")
    print("✅ Data structures - Lists, dictionaries, sets, and tuples")
    print("✅ Control flow - Conditionals, loops, and comprehensions")
    print("✅ Functions and OOP - Modular code and object-oriented design")
    print("✅ File I/O and error handling - Working with files and exceptions")
    print("✅ Data science packages - Essential libraries for analysis")
    print("✅ Real data analysis - Practical application of Python skills")
    print()
    print("Next: Chapter 3 - Mathematics and Statistics")
    print("=" * 80)


def demonstrate_python_basics():
    """Demonstrate basic Python concepts with real examples."""
    print("Python Environment Information:")
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Variables and Data Types:")
    print("-" * 30)

    # Numeric types with real examples
    population = 331002651  # US population
    gdp_growth = 2.3  # GDP growth percentage
    inflation_rate = 3.2 + 0.1j  # Complex number example

    print(f"Population: {population:,} (type: {type(population)})")
    print(f"GDP Growth: {gdp_growth}% (type: {type(gdp_growth)})")
    print(f"Inflation Rate: {inflation_rate} (type: {type(inflation_rate)})")
    print()

    # String operations with real data
    company_name = "TechCorp Analytics"
    print(f"Company: '{company_name}'")
    print(f"Length: {len(company_name)} characters")
    print(f"Uppercase: '{company_name.upper()}'")
    print(f"Lowercase: '{company_name.lower()}'")
    print(f"Words: {company_name.split()}")
    print(f"Contains 'Analytics': {'Analytics' in company_name}")
    print()

    # Boolean operations
    is_public = True
    has_ai_division = True
    is_profitable = False

    print("Company Status:")
    print(f"  Public Company: {is_public}")
    print(f"  AI Division: {has_ai_division}")
    print(f"  Profitable: {is_profitable}")
    print(f"  Good Investment: {is_public and has_ai_division and not is_profitable}")
    print()


def demonstrate_data_structures():
    """Demonstrate Python data structures with real examples."""
    print("Data Structures in Python:")
    print("-" * 30)

    # Lists - Stock prices
    print("1. LISTS - Stock Prices:")
    print("-" * 20)
    stock_prices = [150.25, 152.80, 148.90, 155.30, 153.45, 157.20]
    print(f"Stock prices: {stock_prices}")
    print(f"Number of days: {len(stock_prices)}")
    print(f"Highest price: ${max(stock_prices):.2f}")
    print(f"Lowest price: ${min(stock_prices):.2f}")
    print(f"Average price: ${sum(stock_prices)/len(stock_prices):.2f}")

    # List operations
    stock_prices.append(159.80)
    stock_prices.insert(0, 149.50)
    print(f"After operations: {stock_prices}")
    print()

    # Dictionaries - Company information
    print("2. DICTIONARIES - Company Information:")
    print("-" * 20)
    company_info = {
        "name": "TechCorp Analytics",
        "industry": "Technology",
        "employees": 1250,
        "revenue_2023": 85.6,
        "headquarters": "San Francisco",
        "founded": 2018,
    }

    print("Company Details:")
    for key, value in company_info.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: ${value}B")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

    # Dictionary operations
    company_info["valuation"] = 2.5
    company_info["employees"] = 1300
    print(f"\nUpdated employees: {company_info['employees']}")
    print(f"Company valuation: ${company_info['valuation']}B")
    print()

    # Sets - Unique skills
    print("3. SETS - Unique Skills:")
    print("-" * 20)
    data_scientist_skills = {"Python", "R", "SQL", "Machine Learning", "Statistics"}
    ml_engineer_skills = {
        "Python",
        "TensorFlow",
        "PyTorch",
        "Machine Learning",
        "Deep Learning",
    }

    print(f"Data Scientist skills: {data_scientist_skills}")
    print(f"ML Engineer skills: {ml_engineer_skills}")
    print(f"Common skills: {data_scientist_skills & ml_engineer_skills}")
    print(f"All unique skills: {data_scientist_skills | ml_engineer_skills}")
    print(f"Data Scientist specific: {data_scientist_skills - ml_engineer_skills}")
    print()

    # Tuples - Immutable data
    print("4. TUPLES - Immutable Data:")
    print("-" * 20)
    coordinates = (37.7749, -122.4194)  # San Francisco coordinates
    print(f"Coordinates: {coordinates}")
    print(f"Latitude: {coordinates[0]}")
    print(f"Longitude: {coordinates[1]}")
    print(f"Tuple type: {type(coordinates)}")
    print()


def demonstrate_control_flow():
    """Demonstrate control flow with real examples."""
    print("Control Flow in Python:")
    print("-" * 30)

    # Conditionals - Investment decision
    print("1. CONDITIONALS - Investment Decision:")
    print("-" * 20)

    stock_price = 155.30
    pe_ratio = 25.5
    market_cap = 2.5  # in billions

    print(f"Stock Price: ${stock_price}")
    print(f"P/E Ratio: {pe_ratio}")
    print(f"Market Cap: ${market_cap}B")

    if pe_ratio < 20 and market_cap > 1.0:
        recommendation = "BUY"
        reason = "Undervalued with good market presence"
    elif pe_ratio < 30 and market_cap > 0.5:
        recommendation = "HOLD"
        reason = "Fairly valued, monitor for opportunities"
    else:
        recommendation = "SELL"
        reason = "Overvalued or too small"

    print(f"Recommendation: {recommendation}")
    print(f"Reason: {reason}")
    print()

    # Loops - Data processing
    print("2. LOOPS - Data Processing:")
    print("-" * 20)

    # Process stock prices
    stock_prices = [150.25, 152.80, 148.90, 155.30, 153.45, 157.20]
    daily_returns = []

    print("Daily Stock Returns:")
    for i in range(1, len(stock_prices)):
        daily_return = (
            (stock_prices[i] - stock_prices[i - 1]) / stock_prices[i - 1]
        ) * 100
        daily_returns.append(daily_return)
        print(f"  Day {i}: {daily_return:+.2f}%")

    print(f"Average daily return: {np.mean(daily_returns):.2f}%")
    print(f"Volatility: {np.std(daily_returns):.2f}%")
    print()

    # List comprehensions - Data transformation
    print("3. LIST COMPREHENSIONS - Data Transformation:")
    print("-" * 20)

    # Convert temperatures from Celsius to Fahrenheit
    celsius_temps = [0, 10, 20, 30, 40]
    fahrenheit_temps = [(c * 9 / 5) + 32 for c in celsius_temps]

    print("Temperature Conversion:")
    for c, f in zip(celsius_temps, fahrenheit_temps):
        print(f"  {c}°C = {f:.1f}°F")

    # Filter high returns
    high_returns = [ret for ret in daily_returns if ret > 1.0]
    print(f"\nHigh returns (>1%): {[f'{r:.2f}%' for r in high_returns]}")
    print()

    # While loop - Investment simulation
    print("4. WHILE LOOP - Investment Simulation:")
    print("-" * 20)

    initial_investment = 10000
    current_value = initial_investment
    target_value = 15000
    year = 0
    annual_return = 0.08  # 8% annual return

    print("Investment Growth Simulation:")
    while current_value < target_value and year < 10:
        year += 1
        current_value *= 1 + annual_return
        print(f"  Year {year}: ${current_value:,.2f}")

    if current_value >= target_value:
        print(f"✅ Target reached in {year} years!")
    else:
        print(f"❌ Target not reached in 10 years. Final value: ${current_value:,.2f}")
    print()


def demonstrate_functions_and_oop():
    """Demonstrate functions and OOP with real examples."""
    print("Functions and Object-Oriented Programming:")
    print("-" * 30)

    # Functions - Financial calculations
    print("1. FUNCTIONS - Financial Calculations:")
    print("-" * 20)

    def calculate_compound_interest(principal, rate, time, compounds_per_year=12):
        """Calculate compound interest."""
        rate_decimal = rate / 100
        amount = principal * (1 + rate_decimal / compounds_per_year) ** (
            compounds_per_year * time
        )
        return amount

    def calculate_monthly_payment(principal, annual_rate, years):
        """Calculate monthly mortgage payment."""
        monthly_rate = annual_rate / 100 / 12
        num_payments = years * 12
        if monthly_rate == 0:
            return principal / num_payments
        payment = (
            principal
            * (monthly_rate * (1 + monthly_rate) ** num_payments)
            / ((1 + monthly_rate) ** num_payments - 1)
        )
        return payment

    # Test functions
    investment = 10000
    rate = 7.5
    years = 10

    final_amount = calculate_compound_interest(investment, rate, years)
    print(f"Investment: ${investment:,}")
    print(f"Annual Rate: {rate}%")
    print(f"Time: {years} years")
    print(f"Final Amount: ${final_amount:,.2f}")
    print(f"Total Interest: ${final_amount - investment:,.2f}")

    # Mortgage calculation
    home_price = 500000
    down_payment = 100000
    loan_amount = home_price - down_payment
    mortgage_rate = 4.5
    loan_years = 30

    monthly_payment = calculate_monthly_payment(loan_amount, mortgage_rate, loan_years)
    print(f"\nHome Price: ${home_price:,}")
    print(f"Loan Amount: ${loan_amount:,}")
    print(f"Interest Rate: {mortgage_rate}%")
    print(f"Monthly Payment: ${monthly_payment:,.2f}")
    print()

    # Classes - Portfolio management
    print("2. CLASSES - Portfolio Management:")
    print("-" * 20)

    class Stock:
        def __init__(self, symbol, name, shares, price):
            self.symbol = symbol
            self.name = name
            self.shares = shares
            self.price = price

        def get_value(self):
            return self.shares * self.price

        def get_info(self):
            return f"{self.symbol}: {self.shares} shares @ ${self.price:.2f} = ${self.get_value():,.2f}"

    class Portfolio:
        def __init__(self, name):
            self.name = name
            self.stocks = []

        def add_stock(self, stock):
            self.stocks.append(stock)

        def get_total_value(self):
            return sum(stock.get_value() for stock in self.stocks)

        def display_portfolio(self):
            print(f"\nPortfolio: {self.name}")
            print("-" * 40)
            for stock in self.stocks:
                print(f"  {stock.get_info()}")
            print(f"Total Value: ${self.get_total_value():,.2f}")

    # Create portfolio
    portfolio = Portfolio("Tech Growth Portfolio")

    # Add stocks
    aapl = Stock("AAPL", "Apple Inc.", 50, 175.50)
    msft = Stock("MSFT", "Microsoft Corp.", 30, 320.75)
    googl = Stock("GOOGL", "Alphabet Inc.", 25, 140.25)

    portfolio.add_stock(aapl)
    portfolio.add_stock(msft)
    portfolio.add_stock(googl)

    portfolio.display_portfolio()
    print()


def demonstrate_file_io_and_error_handling():
    """Demonstrate file I/O and error handling with real examples."""
    print("File I/O and Error Handling:")
    print("-" * 30)

    # File operations - Data export
    print("1. FILE OPERATIONS - Data Export:")
    print("-" * 20)

    # Create sample data
    sales_data = [
        {"month": "Jan", "revenue": 125000, "customers": 1250},
        {"month": "Feb", "revenue": 138000, "customers": 1380},
        {"month": "Mar", "revenue": 142000, "customers": 1420},
        {"month": "Apr", "revenue": 156000, "customers": 1560},
        {"month": "May", "revenue": 168000, "customers": 1680},
    ]

    # Write to CSV
    try:
        import csv

        with open("sales_data.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["month", "revenue", "customers"])
            writer.writeheader()
            writer.writerows(sales_data)
        print("✅ Sales data exported to 'sales_data.csv'")
    except Exception as e:
        print(f"❌ Error writing CSV: {e}")

    # Read from CSV
    try:
        with open("sales_data.csv", "r") as file:
            reader = csv.DictReader(file)
            print("\nImported data:")
            for row in reader:
                print(
                    f"  {row['month']}: ${row['revenue']:,} revenue, {row['customers']} customers"
                )
    except FileNotFoundError:
        print("❌ File not found")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

    print()

    # Error handling - Data validation
    print("2. ERROR HANDLING - Data Validation:")
    print("-" * 20)

    def validate_financial_data(data):
        """Validate financial data with error handling."""
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")

            required_fields = ["revenue", "expenses", "profit"]
            for field in required_fields:
                if field not in data:
                    raise KeyError(f"Missing required field: {field}")

                if not isinstance(data[field], (int, float)):
                    raise TypeError(f"{field} must be numeric")

                if data[field] < 0:
                    raise ValueError(f"{field} cannot be negative")

            # Calculate profit margin
            profit_margin = (data["profit"] / data["revenue"]) * 100

            return {
                "is_valid": True,
                "profit_margin": profit_margin,
                "message": "Data validation successful",
            }

        except (ValueError, KeyError, TypeError) as e:
            return {
                "is_valid": False,
                "error": str(e),
                "message": "Data validation failed",
            }
        except ZeroDivisionError:
            return {
                "is_valid": False,
                "error": "Revenue cannot be zero",
                "message": "Data validation failed",
            }

    # Test validation
    test_data = [
        {"revenue": 100000, "expenses": 80000, "profit": 20000},
        {"revenue": 50000, "expenses": 60000, "profit": -10000},
        {"revenue": 0, "expenses": 10000, "profit": -10000},
        {"revenue": "invalid", "expenses": 50000, "profit": 10000},
    ]

    print("Data Validation Results:")
    for i, data in enumerate(test_data, 1):
        result = validate_financial_data(data)
        print(f"  Dataset {i}: {result['message']}")
        if result["is_valid"]:
            print(f"    Profit Margin: {result['profit_margin']:.1f}%")
        else:
            print(f"    Error: {result['error']}")
        print()

    # Clean up
    import os

    if os.path.exists("sales_data.csv"):
        os.remove("sales_data.csv")
        print("✅ Cleaned up temporary files")


def demonstrate_data_science_packages():
    """Demonstrate essential data science packages."""
    print("Python Packages for Data Science:")
    print("-" * 30)

    # NumPy - Numerical computing
    print("1. NUMPY - Numerical Computing:")
    print("-" * 20)

    # Create arrays
    stock_returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005])
    print(f"Stock returns: {stock_returns}")
    print(f"Mean return: {np.mean(stock_returns):.3f}")
    print(f"Standard deviation: {np.std(stock_returns):.3f}")
    print(f"Cumulative return: {np.prod(1 + stock_returns) - 1:.3f}")

    # Array operations
    returns_annualized = stock_returns * 252  # Assuming 252 trading days
    print(f"Annualized returns: {returns_annualized}")
    print()

    # Pandas - Data manipulation
    print("2. PANDAS - Data Manipulation:")
    print("-" * 20)

    # Create DataFrame
    company_data = pd.DataFrame(
        {
            "Company": ["Apple", "Microsoft", "Google", "Amazon", "Tesla"],
            "Market_Cap": [2800, 2500, 1800, 1600, 800],
            "Revenue": [394, 198, 307, 514, 81],
            "Employees": [164000, 221000, 156500, 1608000, 127855],
        }
    )

    print("Company Financial Data:")
    print(company_data)
    print()

    # Data analysis
    print("Data Analysis:")
    print(f"  Total market cap: ${company_data['Market_Cap'].sum():,}B")
    print(f"  Average revenue: ${company_data['Revenue'].mean():.1f}B")
    print(
        f"  Largest employer: {company_data.loc[company_data['Employees'].idxmax(), 'Company']}"
    )
    print()

    # Matplotlib/Seaborn - Visualization
    print("3. VISUALIZATION - Creating Charts:")
    print("-" * 20)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Market cap comparison
    bars1 = ax1.bar(
        company_data["Company"],
        company_data["Market_Cap"],
        color=sns.color_palette("husl", len(company_data)),
        alpha=0.8,
    )
    ax1.set_title("Market Capitalization Comparison", fontweight="bold")
    ax1.set_ylabel("Market Cap (Billions USD)")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, value in zip(bars1, company_data["Market_Cap"]):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            f"${value}B",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Revenue vs Employees scatter
    ax2.scatter(
        company_data["Revenue"],
        company_data["Employees"],
        s=company_data["Market_Cap"] / 10,
        alpha=0.7,
        c=range(len(company_data)),
        cmap="viridis",
    )
    ax2.set_title("Revenue vs Employees", fontweight="bold")
    ax2.set_xlabel("Revenue (Billions USD)")
    ax2.set_ylabel("Employees")

    # Add company labels
    for i, company in enumerate(company_data["Company"]):
        ax2.annotate(
            company,
            (company_data["Revenue"].iloc[i], company_data["Employees"].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("company_analysis.png", dpi=300, bbox_inches="tight")
    print("✅ Company analysis visualization saved as 'company_analysis.png'")
    plt.show()


def demonstrate_real_data_analysis():
    """Demonstrate real data analysis workflow with Python."""
    print("Real Data Analysis Example:")
    print("-" * 30)

    # Create realistic financial dataset
    np.random.seed(42)

    # Generate company performance data
    n_companies = 100

    # Company characteristics
    company_sizes = np.random.choice(
        ["Small", "Medium", "Large"], n_companies, p=[0.4, 0.4, 0.2]
    )
    industries = np.random.choice(
        ["Tech", "Finance", "Healthcare", "Manufacturing", "Retail"], n_companies
    )

    # Financial metrics
    revenues = np.random.lognormal(mean=5, sigma=1, size=n_companies)  # Millions USD
    profit_margins = np.random.normal(0.15, 0.08, n_companies)  # 15% average, 8% std
    profit_margins = np.clip(profit_margins, 0.02, 0.35)  # Realistic range

    # Market performance
    market_caps = (
        revenues * (1 + np.random.normal(0.5, 0.3, n_companies)) * 10
    )  # Revenue multiple
    stock_returns = np.random.normal(
        0.12, 0.25, n_companies
    )  # 12% average, 25% volatility

    # Create DataFrame
    company_df = pd.DataFrame(
        {
            "company_id": range(1, n_companies + 1),
            "size": company_sizes,
            "industry": industries,
            "revenue_millions": revenues,
            "profit_margin": profit_margins,
            "market_cap_millions": market_caps,
            "stock_return": stock_returns,
        }
    )

    # Calculate derived metrics
    company_df["profit_millions"] = (
        company_df["revenue_millions"] * company_df["profit_margin"]
    )
    company_df["pe_ratio"] = (
        company_df["market_cap_millions"] / company_df["profit_millions"]
    )
    company_df["pe_ratio"] = np.clip(
        company_df["pe_ratio"], 5, 50
    )  # Realistic P/E range

    print("📊 Company Performance Dataset:")
    print(f"   Total companies: {len(company_df)}")
    print(f"   Industries: {company_df['industry'].nunique()}")
    print(f"   Size distribution: {company_df['size'].value_counts().to_dict()}")
    print()

    print("📈 Financial Summary:")
    print(f"   Average revenue: ${company_df['revenue_millions'].mean():.1f}M")
    print(f"   Average profit margin: {company_df['profit_margin'].mean():.1%}")
    print(f"   Average P/E ratio: {company_df['pe_ratio'].mean():.1f}")
    print(f"   Average stock return: {company_df['stock_return'].mean():.1%}")
    print()

    # Industry analysis
    print("🏭 Industry Analysis:")
    industry_summary = (
        company_df.groupby("industry")
        .agg(
            {
                "revenue_millions": "mean",
                "profit_margin": "mean",
                "stock_return": "mean",
                "pe_ratio": "mean",
            }
        )
        .round(3)
    )

    print(industry_summary)
    print()

    # Create comprehensive visualizations
    create_comprehensive_analysis_visualizations(company_df)

    return company_df


def create_comprehensive_analysis_visualizations(company_df):
    """Create comprehensive analysis visualizations."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Comprehensive Company Performance Analysis", fontsize=16, fontweight="bold"
    )

    # 1. Revenue Distribution by Industry
    ax1 = axes[0, 0]
    industry_revenue = (
        company_df.groupby("industry")["revenue_millions"]
        .mean()
        .sort_values(ascending=True)
    )
    bars = ax1.barh(
        industry_revenue.index,
        industry_revenue.values,
        color=sns.color_palette("husl", len(industry_revenue)),
        alpha=0.8,
    )
    ax1.set_title("Average Revenue by Industry", fontweight="bold")
    ax1.set_xlabel("Revenue (Millions USD)")

    # Add value labels
    for bar, value in zip(bars, industry_revenue.values):
        width = bar.get_width()
        ax1.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"${value:.1f}M",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # 2. Profit Margin Distribution
    ax2 = axes[0, 1]
    ax2.hist(
        company_df["profit_margin"],
        bins=20,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax2.set_title("Profit Margin Distribution", fontweight="bold")
    ax2.set_xlabel("Profit Margin")
    ax2.set_ylabel("Frequency")
    ax2.axvline(
        company_df["profit_margin"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {company_df['profit_margin'].mean():.1%}",
    )
    ax2.legend()

    # 3. Market Cap vs Revenue
    ax3 = axes[0, 2]
    scatter = ax3.scatter(
        company_df["revenue_millions"],
        company_df["market_cap_millions"],
        c=company_df["profit_margin"],
        cmap="viridis",
        alpha=0.7,
        s=50,
    )
    ax3.set_title("Market Cap vs Revenue", fontweight="bold")
    ax3.set_xlabel("Revenue (Millions USD)")
    ax3.set_ylabel("Market Cap (Millions USD)")
    plt.colorbar(scatter, ax=ax3, label="Profit Margin")

    # 4. Stock Returns by Company Size
    ax4 = axes[1, 0]
    size_returns = company_df.groupby("size")["stock_return"].mean()
    bars = ax4.bar(
        size_returns.index,
        size_returns.values,
        color=sns.color_palette("husl", len(size_returns)),
        alpha=0.8,
    )
    ax4.set_title("Average Stock Returns by Company Size", fontweight="bold")
    ax4.set_ylabel("Average Stock Return")

    # Add value labels
    for bar, value in zip(bars, size_returns.values):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 5. P/E Ratio by Industry
    ax5 = axes[1, 1]
    industry_pe = (
        company_df.groupby("industry")["pe_ratio"].mean().sort_values(ascending=True)
    )
    bars = ax5.barh(
        industry_pe.index,
        industry_pe.values,
        color=sns.color_palette("husl", len(industry_pe)),
        alpha=0.8,
    )
    ax5.set_title("Average P/E Ratio by Industry", fontweight="bold")
    ax5.set_xlabel("P/E Ratio")

    # Add value labels
    for bar, value in zip(bars, industry_pe.values):
        width = bar.get_width()
        ax5.text(
            width + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # 6. Correlation Heatmap
    ax6 = axes[1, 2]
    numeric_cols = [
        "revenue_millions",
        "profit_margin",
        "market_cap_millions",
        "stock_return",
        "pe_ratio",
    ]
    correlation_matrix = company_df[numeric_cols].corr()

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
    plt.savefig("comprehensive_company_analysis.png", dpi=300, bbox_inches="tight")
    print(
        "✅ Comprehensive Company Analysis saved as 'comprehensive_company_analysis.png'"
    )
    plt.show()


if __name__ == "__main__":
    main()
