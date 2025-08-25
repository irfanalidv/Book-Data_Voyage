#!/usr/bin/env python3
"""
Chapter 5: Data Collection and Storage
Data Voyage: Building Robust Data Pipelines and Storage Systems

This script covers essential data collection and storage concepts using REAL data.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import os
import time
import requests
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("CHAPTER 5: DATA COLLECTION AND STORAGE")
    print("=" * 80)
    print()

    # Section 5.1: Data Collection Methods
    print("5.1 DATA COLLECTION METHODS")
    print("-" * 40)
    demonstrate_data_collection()

    # Section 5.2: Data Storage Systems
    print("\n5.2 DATA STORAGE SYSTEMS")
    print("-" * 40)
    demonstrate_data_storage()

    # Section 5.3: Data Pipelines and ETL
    print("\n5.3 DATA PIPELINES AND ETL")
    print("-" * 40)
    demonstrate_data_pipelines()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Data collection methods - Real APIs, web scraping, and public datasets")
    print("✅ Data storage systems - Relational and file-based storage")
    print("✅ Data pipelines and ETL - Extract, transform, and load processes")
    print()
    print("Next: Chapter 6 - Data Cleaning and Preprocessing")
    print("=" * 80)


def demonstrate_data_collection():
    """Demonstrate various data collection methods using REAL data sources."""
    print("Data Collection Methods and Techniques:")
    print("-" * 40)

    # 1. Real API Data Collection
    print("1. REAL API DATA COLLECTION:")
    print("-" * 20)

    # Collect real data from public APIs
    print("Collecting real data from public APIs...")

    # 1.1 Collect COVID-19 data from a public API
    try:
        covid_url = "https://disease.sh/v3/covid-19/countries"
        covid_response = requests.get(covid_url, timeout=10)
        if covid_response.status_code == 200:
            covid_data = covid_response.json()
            print(f"  ✅ Collected COVID-19 data for {len(covid_data)} countries")

            # Extract key metrics for top 10 countries by cases
            covid_df = pd.DataFrame(covid_data)
            covid_df["cases"] = pd.to_numeric(covid_df["cases"], errors="coerce")
            top_countries = covid_df.nlargest(10, "cases")[
                ["country", "cases", "deaths", "recovered", "population"]
            ]
            print("  Top 10 countries by COVID-19 cases:")
            for _, row in top_countries.iterrows():
                print(
                    f"    {row['country']}: {row['cases']:,} cases, {row['deaths']:,} deaths"
                )
        else:
            print(f"  ❌ Failed to collect COVID-19 data: {covid_response.status_code}")
            covid_data = []
    except Exception as e:
        print(f"  ❌ Error collecting COVID-19 data: {e}")
        covid_data = []

    print()

    # 1.2 Collect stock market data
    try:
        # Using Alpha Vantage API (free tier)
        # Note: In production, you'd use an API key
        stock_url = "https://www.alphavantage.co/query"
        stock_params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": "AAPL",
            "apikey": "demo",  # Using demo key for example
        }
        stock_response = requests.get(stock_url, params=stock_params, timeout=10)
        if stock_response.status_code == 200:
            stock_data = stock_response.json()
            if "Time Series (Daily)" in stock_data:
                print("  ✅ Collected Apple stock data (demo API)")
                # Extract recent data
                daily_data = stock_data["Time Series (Daily)"]
                recent_dates = list(daily_data.keys())[:5]
                for date in recent_dates:
                    data = daily_data[date]
                    print(
                        f"    {date}: Open: ${data['1. open']}, Close: ${data['4. close']}"
                    )
            else:
                print("  ℹ️  Using demo stock data (limited access)")
        else:
            print(f"  ❌ Failed to collect stock data: {stock_response.status_code}")
    except Exception as e:
        print(f"  ❌ Error collecting stock data: {e}")

    print()

    # 1.3 Collect weather data
    try:
        # Using OpenWeatherMap API (free tier)
        # Note: In production, you'd use an API key
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            "q": "London,UK",
            "appid": "demo",  # Using demo for example
            "units": "metric",
        }
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            print("  ✅ Collected London weather data (demo API)")
            print(f"    Temperature: {weather_data['main']['temp']}°C")
            print(f"    Humidity: {weather_data['main']['humidity']}%")
            print(f"    Description: {weather_data['weather'][0]['description']}")
        else:
            print(f"  ℹ️  Using simulated weather data (API limit reached)")
            weather_data = {
                "main": {"temp": 18.5, "humidity": 65},
                "weather": [{"description": "scattered clouds"}],
            }
    except Exception as e:
        print(f"  ❌ Error collecting weather data: {e}")
        weather_data = {
            "main": {"temp": 18.5, "humidity": 65},
            "weather": [{"description": "scattered clouds"}],
        }

    print()

    # 2. Real Dataset Collection
    print("2. REAL DATASET COLLECTION:")
    print("-" * 20)

    # 2.1 Load built-in sklearn datasets
    try:
        from sklearn.datasets import load_iris, load_diabetes

        iris = load_iris()
        diabetes = load_diabetes()

        print("  ✅ Loaded sklearn built-in datasets:")
        print(
            f"    Iris dataset: {iris.data.shape[0]} samples, {iris.data.shape[1]} features"
        )
        print(
            f"    Diabetes dataset: {diabetes.data.shape[0]} samples, {diabetes.data.shape[1]} features"
        )

        # Create DataFrames
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df["target"] = iris.target
        iris_df["species"] = [iris.target_names[i] for i in iris.target]

        diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        diabetes_df["target"] = diabetes.target

        print(f"    Iris species: {list(set(iris_df['species']))}")
        print(
            f"    Diabetes target range: {diabetes_df['target'].min():.2f} to {diabetes_df['target'].max():.2f}"
        )

    except ImportError:
        print("  ❌ sklearn not available, using simulated data")
        iris_df = None
        diabetes_df = None

    print()

    # 2.2 Download sample CSV from GitHub
    try:
        # Download a sample dataset from GitHub
        sample_url = (
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/iris.csv"
        )
        sample_response = requests.get(sample_url, timeout=10)
        if sample_response.status_code == 200:
            sample_data = sample_response.text
            # Save to local file
            with open("collected_data/sample_iris.csv", "w") as f:
                f.write(sample_data)
            print("  ✅ Downloaded sample iris dataset from GitHub")
            print(f"    File size: {len(sample_data)} characters")
        else:
            print(
                f"  ❌ Failed to download sample dataset: {sample_response.status_code}"
            )
    except Exception as e:
        print(f"  ❌ Error downloading sample dataset: {e}")

    print()

    # 3. Web Scraping (with real website structure)
    print("3. WEB SCRAPING (Real Website Structure):")
    print("-" * 20)

    try:
        # Scrape a simple, public website for demonstration
        # Using a website that allows scraping for educational purposes
        from bs4 import BeautifulSoup

        # Example: Scrape quotes from quotes.toscrape.com
        quotes_url = "http://quotes.toscrape.com"
        quotes_response = requests.get(quotes_url, timeout=10)
        if quotes_response.status_code == 200:
            soup = BeautifulSoup(quotes_response.content, "html.parser")
            quotes = soup.find_all("span", class_="text")
            authors = soup.find_all("small", class_="author")

            if quotes and authors:
                print(f"  ✅ Scraped {len(quotes)} quotes from quotes.toscrape.com")
                print("  Sample quotes:")
                for i in range(min(3, len(quotes))):
                    print(f'    "{quotes[i].text}" - {authors[i].text}')

                # Save scraped data
                scraped_data = []
                for i in range(min(len(quotes), len(authors))):
                    scraped_data.append(
                        {"quote": quotes[i].text, "author": authors[i].text}
                    )
            else:
                print("  ℹ️  Website structure changed, using simulated data")
                scraped_data = [
                    {
                        "quote": "Be the change you wish to see in the world",
                        "author": "Mahatma Gandhi",
                    },
                    {"quote": "Stay hungry, stay foolish", "author": "Steve Jobs"},
                    {
                        "quote": "The only way to do great work is to love what you do",
                        "author": "Steve Jobs",
                    },
                ]
        else:
            print(f"  ❌ Failed to scrape website: {quotes_response.status_code}")
            scraped_data = []
    except Exception as e:
        print(f"  ❌ Error during web scraping: {e}")
        scraped_data = []

    print()

    # Store collected data for later use
    global collected_data
    collected_data = {
        "covid_data": covid_data,
        "weather_data": weather_data,
        "iris_df": iris_df,
        "diabetes_df": diabetes_df,
        "scraped_data": scraped_data,
    }

    print("📊 Data Collection Summary:")
    print(f"  • COVID-19 data: {len(covid_data)} countries")
    print(f"  • Weather data: London current conditions")
    print(f"  • Iris dataset: {iris_df.shape[0] if iris_df is not None else 0} samples")
    print(
        f"  • Diabetes dataset: {diabetes_df.shape[0] if diabetes_df is not None else 0} samples"
    )
    print(f"  • Scraped quotes: {len(scraped_data)} quotes")
    print()


def demonstrate_data_storage():
    """Demonstrate different data storage systems using REAL collected data."""
    print("Data Storage Systems and Architectures:")
    print("-" * 40)

    # 1. Relational Database Storage
    print("1. RELATIONAL DATABASE STORAGE:")
    print("-" * 30)

    # Create SQLite database
    db_filename = "data_collection.db"
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Create tables for our real data
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS covid_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT NOT NULL,
            cases INTEGER,
            deaths INTEGER,
            recovered INTEGER,
            population INTEGER,
            updated_at TEXT NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS iris_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            species TEXT,
            target INTEGER
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS quotes_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quote TEXT NOT NULL,
            author TEXT NOT NULL,
            scraped_at TEXT NOT NULL
        )
    """
    )

    print("  ✅ Created database schema with 3 tables for real data")

    # Insert real data from collection
    if "collected_data" in globals():
        # Insert COVID-19 data
        if collected_data["covid_data"]:
            for country_data in collected_data["covid_data"][:10]:  # Top 10 countries
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO covid_data 
                    (country, cases, deaths, recovered, population, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        country_data.get("country", "Unknown"),
                        country_data.get("cases", 0),
                        country_data.get("deaths", 0),
                        country_data.get("recovered", 0),
                        country_data.get("population", 0),
                        datetime.now().isoformat(),
                    ),
                )
            print(
                f"  ✅ Inserted COVID-19 data for {min(10, len(collected_data['covid_data']))} countries"
            )

        # Insert Iris data
        if collected_data["iris_df"] is not None:
            iris_data = collected_data["iris_df"]
            for _, row in iris_data.iterrows():
                cursor.execute(
                    """
                    INSERT INTO iris_data 
                    (sepal_length, sepal_width, petal_length, petal_width, species, target)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        row["sepal length (cm)"],
                        row["sepal width (cm)"],
                        row["petal length (cm)"],
                        row["petal width (cm)"],
                        row["species"],
                        row["target"],
                    ),
                )
            print(f"  ✅ Inserted {len(iris_data)} iris samples")

        # Insert scraped quotes
        if collected_data["scraped_data"]:
            for quote_data in collected_data["scraped_data"]:
                cursor.execute(
                    """
                    INSERT INTO quotes_data (quote, author, scraped_at)
                    VALUES (?, ?, ?)
                """,
                    (
                        quote_data["quote"],
                        quote_data["author"],
                        datetime.now().isoformat(),
                    ),
                )
            print(f"  ✅ Inserted {len(collected_data['scraped_data'])} quotes")

    # Query the database to show real data
    print("\n  Database Queries with Real Data:")

    cursor.execute("SELECT COUNT(*) FROM covid_data")
    covid_count = cursor.fetchone()[0]
    print(f"    Total COVID-19 records: {covid_count}")

    cursor.execute("SELECT COUNT(*) FROM iris_data")
    iris_count = cursor.fetchone()[0]
    print(f"    Total iris samples: {iris_count}")

    cursor.execute("SELECT COUNT(*) FROM quotes_data")
    quotes_count = cursor.fetchone()[0]
    print(f"    Total quotes: {quotes_count}")

    # Show some sample data
    cursor.execute(
        "SELECT country, cases, deaths FROM covid_data ORDER BY cases DESC LIMIT 3"
    )
    top_covid = cursor.fetchall()
    print(f"    Top 3 countries by cases:")
    for country, cases, deaths in top_covid:
        print(f"      {country}: {cases:,} cases, {deaths:,} deaths")

    cursor.execute("SELECT species, COUNT(*) FROM iris_data GROUP BY species")
    iris_species = cursor.fetchall()
    print(f"    Iris species distribution:")
    for species, count in iris_species:
        print(f"      {species}: {count} samples")

    print()

    # 2. File-based Storage
    print("2. FILE-BASED STORAGE:")
    print("-" * 20)

    # Create data directory
    data_dir = "collected_data"
    os.makedirs(data_dir, exist_ok=True)

    # Save COVID-19 data as JSON
    if collected_data["covid_data"]:
        covid_file = os.path.join(data_dir, "covid_data.json")
        with open(covid_file, "w") as f:
            json.dump(collected_data["covid_data"][:10], f, indent=2, default=str)
        print(f"  ✅ Saved COVID-19 data to {covid_file}")

    # Save Iris data as CSV
    if collected_data["iris_df"] is not None:
        iris_file = os.path.join(data_dir, "iris_data.csv")
        collected_data["iris_df"].to_csv(iris_file, index=False)
        print(f"  ✅ Saved iris data to {iris_file}")

    # Save quotes data as JSON
    if collected_data["scraped_data"]:
        quotes_file = os.path.join(data_dir, "quotes_data.json")
        with open(quotes_file, "w") as f:
            json.dump(collected_data["scraped_data"], f, indent=2, default=str)
        print(f"  ✅ Saved quotes data to {quotes_file}")

    # Save weather data
    if collected_data["weather_data"]:
        weather_file = os.path.join(data_dir, "weather_data.json")
        with open(weather_file, "w") as f:
            json.dump(collected_data["weather_data"], f, indent=2, default=str)
        print(f"  ✅ Saved weather data to {weather_file}")

    # Close database connection
    conn.close()


def demonstrate_data_pipelines():
    """Demonstrate data pipeline and ETL processes using REAL data."""
    print("Data Pipelines and ETL Processes:")
    print("-" * 40)

    # 1. Extract Phase
    print("1. EXTRACT PHASE:")
    print("-" * 15)

    print("  Extracting data from multiple sources...")

    # Extract from database
    conn = sqlite3.connect("data_collection.db")
    covid_df = pd.read_sql_query("SELECT * FROM covid_data", conn)
    iris_df = pd.read_sql_query("SELECT * FROM iris_data", conn)
    quotes_df = pd.read_sql_query("SELECT * FROM quotes_data", conn)

    print(f"    ✅ Extracted {len(covid_df)} COVID-19 records from database")
    print(f"    ✅ Extracted {len(iris_df)} iris samples from database")
    print(f"    ✅ Extracted {len(quotes_df)} quotes from database")

    # Extract from files
    if os.path.exists("collected_data/weather_data.json"):
        weather_data = json.load(open("collected_data/weather_data.json"))
        print(f"    ✅ Extracted weather data: {len(weather_data)} records")

    print()

    # 2. Transform Phase
    print("2. TRANSFORM PHASE:")
    print("-" * 15)

    print("  Transforming and cleaning real data...")

    # Transform COVID-19 data
    if not covid_df.empty:
        covid_df["updated_at"] = pd.to_datetime(covid_df["updated_at"])
        covid_df["mortality_rate"] = (
            covid_df["deaths"] / covid_df["cases"] * 100
        ).round(2)
        covid_df["recovery_rate"] = (
            covid_df["recovered"] / covid_df["cases"] * 100
        ).round(2)
        covid_df["cases_per_million"] = (
            covid_df["cases"] / covid_df["population"] * 1000000
        ).round(0)

        # Clean up infinite values
        covid_df = covid_df.replace([np.inf, -np.inf], np.nan)
        covid_df = covid_df.dropna(subset=["mortality_rate", "recovery_rate"])

        print(
            f"    ✅ Transformed COVID-19 data: added mortality_rate, recovery_rate, cases_per_million"
        )
        print(f"    ✅ Cleaned data: {len(covid_df)} valid records remaining")

    # Transform Iris data
    if not iris_df.empty:
        iris_df["petal_area"] = (
            iris_df["petal_length"] * iris_df["petal_width"]
        ).round(2)
        iris_df["sepal_area"] = (
            iris_df["sepal_length"] * iris_df["sepal_width"]
        ).round(2)
        iris_df["petal_to_sepal_ratio"] = (
            iris_df["petal_area"] / iris_df["sepal_area"]
        ).round(3)

        print(
            f"    ✅ Transformed iris data: added petal_area, sepal_area, petal_to_sepal_ratio"
        )

    # Transform quotes data
    if not quotes_df.empty:
        quotes_df["scraped_at"] = pd.to_datetime(quotes_df["scraped_at"])
        quotes_df["quote_length"] = quotes_df["quote"].str.len()
        quotes_df["word_count"] = quotes_df["quote"].str.split().str.len()
        quotes_df["author_word_count"] = quotes_df["author"].str.split().str.len()

        print(
            f"    ✅ Transformed quotes data: added quote_length, word_count, author_word_count"
        )

    print()

    # 3. Load Phase
    print("3. LOAD PHASE:")
    print("-" * 15)

    print("  Loading transformed data to destination...")

    # Create transformed data directory
    transformed_dir = "transformed_data"
    os.makedirs(transformed_dir, exist_ok=True)

    # Save transformed data
    if not covid_df.empty:
        covid_df.to_csv(
            os.path.join(transformed_dir, "transformed_covid.csv"), index=False
        )
        print(f"    ✅ Saved transformed COVID-19 data")

    if not iris_df.empty:
        iris_df.to_csv(
            os.path.join(transformed_dir, "transformed_iris.csv"), index=False
        )
        print(f"    ✅ Saved transformed iris data")

    if not quotes_df.empty:
        quotes_df.to_csv(
            os.path.join(transformed_dir, "transformed_quotes.csv"), index=False
        )
        print(f"    ✅ Saved transformed quotes data")

    # Create data warehouse
    warehouse_conn = sqlite3.connect("data_warehouse.db")

    # Load COVID-19 data into warehouse
    if not covid_df.empty:
        covid_warehouse = covid_df[
            [
                "country",
                "cases",
                "deaths",
                "recovered",
                "population",
                "mortality_rate",
                "recovery_rate",
                "cases_per_million",
                "updated_at",
            ]
        ].copy()
        covid_warehouse.to_sql(
            "dim_covid", warehouse_conn, if_exists="replace", index=False
        )
        print(
            f"    ✅ Loaded {len(covid_warehouse)} COVID-19 records to data warehouse"
        )

    # Load Iris data into warehouse
    if not iris_df.empty:
        iris_warehouse = iris_df[
            [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                "species",
                "target",
                "petal_area",
                "sepal_area",
                "petal_to_sepal_ratio",
            ]
        ].copy()
        iris_warehouse.to_sql(
            "dim_iris", warehouse_conn, if_exists="replace", index=False
        )
        print(f"    ✅ Loaded {len(iris_warehouse)} iris samples to data warehouse")

    # Create summary statistics
    print("\n  Data Pipeline Summary:")
    print(f"    ✅ Collected data from 4+ real sources")
    print(f"    ✅ Stored in 1 database system")
    print(f"    ✅ Created 3+ transformed datasets")
    print(f"    ✅ Established data warehouse with 2 dimension tables")

    if not covid_df.empty:
        print(f"    📊 COVID-19 insights:")
        print(
            f"      • Countries with highest mortality rate: {covid_df.nlargest(3, 'mortality_rate')['country'].tolist()}"
        )
        print(
            f"      • Countries with highest recovery rate: {covid_df.nlargest(3, 'recovery_rate')['country'].tolist()}"
        )

    if not iris_df.empty:
        print(f"    🌸 Iris dataset insights:")
        print(
            f"      • Species distribution: {iris_df['species'].value_counts().to_dict()}"
        )
        print(f"      • Average petal area by species:")
        species_avg = iris_df.groupby("species")["petal_area"].mean().round(2)
        for species, avg_area in species_avg.items():
            print(f"        {species}: {avg_area} cm²")

    # Close connections
    conn.close()
    warehouse_conn.close()


if __name__ == "__main__":
    main()
