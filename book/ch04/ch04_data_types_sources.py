#!/usr/bin/env python3
"""
Chapter 4: Data Types and Sources
Data Voyage: Understanding Data Fundamentals and Collection Methods

This script covers essential data types and sources with actual code execution.
"""

import numpy as np
import pandas as pd
import json
import csv
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import sqlite3


def main():
    print("=" * 80)
    print("CHAPTER 4: DATA TYPES AND SOURCES")
    print("=" * 80)
    print()

    # Section 4.1: Data Types and Structures
    print("4.1 DATA TYPES AND STRUCTURES")
    print("-" * 40)
    demonstrate_data_types()

    # Section 4.2: Data Sources and Collection
    print("\n4.2 DATA SOURCES AND COLLECTION")
    print("-" * 40)
    demonstrate_data_sources()

    # Section 4.3: Data Quality and Validation
    print("\n4.3 DATA QUALITY AND VALIDATION")
    print("-" * 40)
    demonstrate_data_quality()

    # Section 4.4: Data Storage and Formats
    print("\n4.4 DATA STORAGE AND FORMATS")
    print("-" * 40)
    demonstrate_data_storage()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Data types and structures - Understanding data fundamentals")
    print("✅ Data sources and collection - Various data acquisition methods")
    print("✅ Data quality and validation - Ensuring data reliability")
    print("✅ Data storage and formats - Working with different file types")
    print()
    print("Next: Chapter 5 - Data Collection and Storage")
    print("=" * 80)


def demonstrate_data_types():
    """Demonstrate different data types and structures."""
    print("Understanding Data Types in Data Science:")
    print("-" * 40)

    # 1. Structured Data
    print("1. STRUCTURED DATA:")
    print("-" * 20)

    # Tabular data
    structured_data = {
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "city": ["NYC", "LA", "Chicago", "Boston", "Seattle"],
        "purchase_amount": [150.50, 200.75, 89.99, 300.25, 175.00],
    }

    df = pd.DataFrame(structured_data)
    print("Customer Data (Structured):")
    print(df)
    print(f"Data types: {df.dtypes}")
    print()

    # 2. Semi-structured Data
    print("2. SEMI-STRUCTURED DATA:")
    print("-" * 20)

    # JSON data
    semi_structured = [
        {
            "id": 1,
            "name": "Product A",
            "category": "Electronics",
            "tags": ["wireless", "portable"],
            "metadata": {
                "weight": "0.5kg",
                "dimensions": {"length": 10, "width": 5, "height": 2},
            },
        },
        {
            "id": 2,
            "name": "Product B",
            "category": "Books",
            "tags": ["fiction", "bestseller"],
            "metadata": {"pages": 350, "language": "English"},
        },
    ]

    print("Product Data (Semi-structured JSON):")
    for product in semi_structured:
        print(f"  {product['name']} - {product['category']}")
        print(f"    Tags: {', '.join(product['tags'])}")
        if "weight" in product["metadata"]:
            print(f"    Weight: {product['metadata']['weight']}")
        if "pages" in product["metadata"]:
            print(f"    Pages: {product['metadata']['pages']}")
    print()

    # 3. Unstructured Data
    print("3. UNSTRUCTURED DATA:")
    print("-" * 20)

    # Text data
    text_data = [
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Big data refers to data sets that are too large or complex to be dealt with by traditional data-processing application software.",
    ]

    print("Text Data (Unstructured):")
    for i, text in enumerate(text_data, 1):
        print(f"  Text {i}: {text[:80]}...")
        print(f"    Length: {len(text)} characters")
        print(f"    Word count: {len(text.split())}")
    print()

    # 4. Time Series Data
    print("4. TIME SERIES DATA:")
    print("-" * 20)

    # Generate time series data
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    values = np.random.normal(100, 10, 30).cumsum()

    time_series = pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "rolling_avg": pd.Series(values).rolling(window=7).mean(),
        }
    )

    print("Time Series Data:")
    print(time_series.head(10))
    print(f"Data shape: {time_series.shape}")
    print()


def demonstrate_data_sources():
    """Demonstrate various data sources and collection methods."""
    print("Data Sources and Collection Methods:")
    print("-" * 40)

    # 1. Internal Data Sources
    print("1. INTERNAL DATA SOURCES:")
    print("-" * 20)

    # Database simulation
    print("Database Records:")
    sales_data = [
        {"date": "2024-01-01", "product": "Laptop", "quantity": 5, "revenue": 2500},
        {"date": "2024-01-02", "product": "Phone", "quantity": 10, "revenue": 8000},
        {"date": "2024-01-03", "product": "Tablet", "quantity": 3, "revenue": 1500},
    ]

    for record in sales_data:
        print(
            f"  {record['date']}: {record['product']} - Qty: {record['quantity']}, Revenue: ${record['revenue']}"
        )
    print()

    # 2. External Data Sources
    print("2. EXTERNAL DATA SOURCES:")
    print("-" * 20)

    # API simulation
    print("API Data (Simulated):")
    api_data = {
        "weather": {"temperature": 72, "humidity": 65, "condition": "Sunny"},
        "stock_price": {"symbol": "AAPL", "price": 150.25, "change": 2.5},
        "news": {"headline": "Tech stocks rally", "source": "Financial Times"},
    }

    for category, data in api_data.items():
        print(f"  {category.title()}: {data}")
    print()

    # 3. Web Scraping Simulation
    print("3. WEB SCRAPING (Simulated):")
    print("-" * 20)

    # Simulate scraped data
    scraped_data = [
        {
            "title": "Data Science Trends 2024",
            "url": "https://example.com/trends",
            "views": 15000,
        },
        {
            "title": "Machine Learning Guide",
            "url": "https://example.com/ml-guide",
            "views": 22000,
        },
        {
            "title": "Python for Data Analysis",
            "url": "https://example.com/python",
            "views": 18000,
        },
    ]

    print("Scraped Web Content:")
    for item in scraped_data:
        print(f"  {item['title']} - {item['views']:,} views")
    print()

    # 4. Sensor/IoT Data
    print("4. SENSOR/IOT DATA:")
    print("-" * 20)

    # Generate sensor data
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=24, freq="H")
    temperature = np.random.normal(70, 5, 24)
    humidity = np.random.normal(50, 10, 24)

    sensor_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_f": temperature,
            "humidity_percent": humidity,
        }
    )

    print("Sensor Data (24-hour period):")
    print(sensor_data.head(6))
    print(f"Temperature range: {temperature.min():.1f}°F - {temperature.max():.1f}°F")
    print(f"Humidity range: {humidity.min():.1f}% - {humidity.max():.1f}%")
    print()


def demonstrate_data_quality():
    """Demonstrate data quality assessment and validation."""
    print("Data Quality Assessment and Validation:")
    print("-" * 40)

    # Create sample data with quality issues
    print("Sample Dataset with Quality Issues:")
    print("-" * 30)

    # Generate data with various quality problems
    np.random.seed(42)
    n_records = 100

    # Clean data
    customer_ids = list(range(1, n_records + 1))
    ages = np.random.normal(35, 10, n_records)
    ages = np.clip(ages, 18, 80)  # Reasonable age range

    # Introduce quality issues
    data_with_issues = []
    for i in range(n_records):
        record = {
            "customer_id": customer_ids[i],
            "age": ages[i],
            "email": f"customer{i}@example.com",
            "purchase_amount": np.random.exponential(100),
        }

        # Introduce missing values (5% of records)
        if np.random.random() < 0.05:
            record["age"] = None

        # Introduce invalid ages (2% of records)
        if np.random.random() < 0.02:
            record["age"] = np.random.choice([-5, 150, 999])

        # Introduce invalid emails (3% of records)
        if np.random.random() < 0.03:
            record["email"] = np.random.choice(["invalid-email", "no@domain", ""])

        # Introduce negative purchase amounts (1% of records)
        if np.random.random() < 0.01:
            record["purchase_amount"] = -np.random.exponential(50)

        data_with_issues.append(record)

    df_issues = pd.DataFrame(data_with_issues)
    print(f"Dataset shape: {df_issues.shape}")
    print(df_issues.head(10))
    print()

    # Data Quality Assessment
    print("Data Quality Assessment:")
    print("-" * 30)

    # 1. Completeness
    completeness = df_issues.notna().mean() * 100
    print("Completeness (% non-null values):")
    for col, comp in completeness.items():
        print(f"  {col}: {comp:.1f}%")
    print()

    # 2. Validity
    print("Data Validity Checks:")

    # Age validation
    valid_ages = df_issues["age"].between(18, 80)
    age_validity = valid_ages.mean() * 100
    print(f"  Valid ages (18-80): {age_validity:.1f}%")

    # Email validation
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    valid_emails = df_issues["email"].str.match(email_pattern, na=False)
    email_validity = valid_emails.mean() * 100
    print(f"  Valid email format: {email_validity:.1f}%")

    # Purchase amount validation
    valid_purchases = df_issues["purchase_amount"] > 0
    purchase_validity = valid_purchases.mean() * 100
    print(f"  Valid purchase amounts (>0): {purchase_validity:.1f}%")
    print()

    # 3. Data Cleaning
    print("Data Cleaning Process:")
    print("-" * 20)

    # Create cleaned dataset
    df_clean = df_issues.copy()

    # Remove records with missing critical fields
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=["customer_id", "age"])

    # Remove records with invalid ages
    df_clean = df_clean[df_clean["age"].between(18, 80)]

    # Remove records with invalid purchase amounts
    df_clean = df_clean[df_clean["purchase_amount"] > 0]

    # Clean email addresses (remove invalid ones)
    df_clean = df_clean[df_clean["email"].str.match(email_pattern, na=False)]

    final_count = len(df_clean)
    removed_count = initial_count - final_count

    print(f"  Initial records: {initial_count}")
    print(f"  Records removed: {removed_count}")
    print(f"  Final records: {final_count}")
    print(f"  Data retention: {final_count/initial_count*100:.1f}%")
    print()

    # 4. Data Quality Metrics
    print("Final Data Quality Metrics:")
    print("-" * 25)

    # Completeness after cleaning
    completeness_clean = df_clean.notna().mean() * 100
    print("Completeness after cleaning:")
    for col, comp in completeness_clean.items():
        print(f"  {col}: {comp:.1f}%")

    # Data consistency
    age_std = df_clean["age"].std()
    purchase_std = df_clean["purchase_amount"].std()
    print(f"\nData consistency:")
    print(f"  Age standard deviation: {age_std:.2f}")
    print(f"  Purchase amount standard deviation: {purchase_std:.2f}")
    print()


def demonstrate_data_storage():
    """Demonstrate data storage and different file formats."""
    print("Data Storage and File Formats:")
    print("-" * 40)

    # Create sample data for storage demonstration
    sample_data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [85.5, 92.3, 78.9, 95.1, 88.7],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    }

    df = pd.DataFrame(sample_data)
    print("Sample Data for Storage Demonstration:")
    print(df)
    print()

    # 1. CSV Format
    print("1. CSV FORMAT:")
    print("-" * 15)

    csv_filename = "sample_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✅ Data saved to {csv_filename}")

    # Read CSV back
    df_csv = pd.read_csv(csv_filename)
    print(f"✅ Data loaded from {csv_filename}")
    print(f"   Shape: {df_csv.shape}")
    print()

    # 2. JSON Format
    print("2. JSON FORMAT:")
    print("-" * 15)

    json_filename = "sample_data.json"
    df.to_json(json_filename, orient="records", indent=2)
    print(f"✅ Data saved to {json_filename}")

    # Read JSON back
    df_json = pd.read_json(json_filename)
    print(f"✅ Data loaded from {json_filename}")
    print(f"   Shape: {df_json.shape}")
    print()

    # 3. Excel Format
    print("3. EXCEL FORMAT:")
    print("-" * 15)

    try:
        excel_filename = "sample_data.xlsx"
        df.to_excel(excel_filename, index=False, sheet_name="Data")
        print(f"✅ Data saved to {excel_filename}")

        # Read Excel back
        df_excel = pd.read_excel(excel_filename, sheet_name="Data")
        print(f"✅ Data loaded from {excel_filename}")
        print(f"   Shape: {df_excel.shape}")
    except ImportError:
        print("❌ Excel support not available (install openpyxl)")
    print()

    # 4. SQLite Database
    print("4. SQLITE DATABASE:")
    print("-" * 20)

    db_filename = "sample_data.db"
    conn = sqlite3.connect(db_filename)

    # Save to SQLite
    df.to_sql("sample_table", conn, if_exists="replace", index=False)
    print(f"✅ Data saved to SQLite database: {db_filename}")

    # Read from SQLite
    df_sql = pd.read_sql_query("SELECT * FROM sample_table", conn)
    print(f"✅ Data loaded from SQLite database")
    print(f"   Shape: {df_sql.shape}")

    # Close connection
    conn.close()
    print()

    # 5. Parquet Format (if available)
    print("5. PARQUET FORMAT:")
    print("-" * 18)

    try:
        parquet_filename = "sample_data.parquet"
        df.to_parquet(parquet_filename, index=False)
        print(f"✅ Data saved to {parquet_filename}")

        # Read Parquet back
        df_parquet = pd.read_parquet(parquet_filename)
        print(f"✅ Data loaded from {parquet_filename}")
        print(f"   Shape: {df_parquet.shape}")
    except ImportError:
        print("❌ Parquet support not available (install pyarrow)")
    print()

    # File size comparison
    print("File Size Comparison:")
    print("-" * 20)

    import os

    files_to_check = [csv_filename, json_filename, db_filename]

    for filename in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  {filename}: {size:,} bytes")

    # Clean up temporary files
    print("\nCleaning up temporary files...")
    for filename in [csv_filename, json_filename, db_filename]:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"  🧹 Removed {filename}")

    if "excel_filename" in locals() and os.path.exists(excel_filename):
        os.remove(excel_filename)
        print(f"  🧹 Removed {excel_filename}")

    if "parquet_filename" in locals() and os.path.exists(parquet_filename):
        os.remove(parquet_filename)
        print(f"  🧹 Removed {parquet_filename}")

    print()


if __name__ == "__main__":
    main()
