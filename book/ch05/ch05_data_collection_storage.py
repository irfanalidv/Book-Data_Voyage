#!/usr/bin/env python3
"""
Chapter 5: Data Collection and Storage
Data Voyage: Building Robust Data Pipelines and Storage Systems

This script covers essential data collection and storage concepts.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import os
import time
from datetime import datetime, timedelta


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
    print("✅ Data collection methods - APIs, web scraping, and sensors")
    print("✅ Data storage systems - Relational and file-based storage")
    print("✅ Data pipelines and ETL - Extract, transform, and load processes")
    print()
    print("Next: Chapter 6 - Data Cleaning and Preprocessing")
    print("=" * 80)


def demonstrate_data_collection():
    """Demonstrate various data collection methods."""
    print("Data Collection Methods and Techniques:")
    print("-" * 40)

    # 1. API Data Collection
    print("1. API DATA COLLECTION:")
    print("-" * 20)

    # Simulate API calls
    print("Simulating API data collection...")

    # Mock API responses
    api_responses = {
        "users": {
            "status": "success",
            "data": [
                {
                    "id": 1,
                    "name": "Alice",
                    "email": "alice@example.com",
                    "created_at": "2024-01-01",
                },
                {
                    "id": 2,
                    "name": "Bob",
                    "email": "bob@example.com",
                    "created_at": "2024-01-02",
                },
                {
                    "id": 3,
                    "name": "Charlie",
                    "email": "charlie@example.com",
                    "created_at": "2024-01-03",
                },
            ],
            "total": 3,
        },
        "products": {
            "status": "success",
            "data": [
                {"id": 1, "name": "Laptop", "price": 999.99, "category": "Electronics"},
                {"id": 2, "name": "Phone", "price": 699.99, "category": "Electronics"},
                {"id": 3, "name": "Book", "price": 29.99, "category": "Education"},
            ],
            "total": 3,
        },
    }

    for endpoint, response in api_responses.items():
        print(f"  ✅ Collected data from {endpoint}: {len(response['data'])} records")

    print()

    # 2. Sensor/IoT Data Collection
    print("2. SENSOR/IOT DATA COLLECTION:")
    print("-" * 20)

    # Generate sensor data
    np.random.seed(42)
    n_sensors = 3
    n_readings = 24

    sensor_data = []
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for sensor_id in range(1, n_sensors + 1):
        if sensor_id == 1:  # Temperature
            base_value, noise = 70, 5
        elif sensor_id == 2:  # Humidity
            base_value, noise = 50, 10
        else:  # Pressure
            base_value, noise = 1013, 2

        for hour in range(n_readings):
            timestamp = base_time + timedelta(hours=hour)
            reading = base_value + np.random.normal(0, noise)
            reading = np.clip(reading, base_value - 3 * noise, base_value + 3 * noise)

            sensor_data.append(
                {
                    "sensor_id": sensor_id,
                    "timestamp": timestamp,
                    "reading": round(reading, 2),
                    "sensor_type": ["temperature", "humidity", "pressure"][
                        sensor_id - 1
                    ],
                }
            )

    print(f"  ✅ Collected {len(sensor_data)} sensor readings from {n_sensors} sensors")
    print()

    # Store collected data for later use
    global collected_data
    collected_data = {"api_data": api_responses, "sensor_data": sensor_data}


def demonstrate_data_storage():
    """Demonstrate different data storage systems."""
    print("Data Storage Systems and Architectures:")
    print("-" * 40)

    # 1. Relational Database Storage
    print("1. RELATIONAL DATABASE STORAGE:")
    print("-" * 30)

    # Create SQLite database
    db_filename = "data_collection.db"
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            reading REAL NOT NULL,
            sensor_type TEXT NOT NULL
        )
    """
    )

    print("  ✅ Created database schema with 3 tables")

    # Insert data from collection
    if "collected_data" in globals():
        # Insert users
        for user in collected_data["api_data"]["users"]["data"]:
            cursor.execute(
                """
                INSERT OR REPLACE INTO users (id, name, email, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (user["id"], user["name"], user["email"], user["created_at"]),
            )

        # Insert products
        for product in collected_data["api_data"]["products"]["data"]:
            cursor.execute(
                """
                INSERT OR REPLACE INTO products (id, name, price, category)
                VALUES (?, ?, ?, ?)
            """,
                (product["id"], product["name"], product["price"], product["category"]),
            )

        # Insert sensor readings
        for reading in collected_data["sensor_data"]:
            cursor.execute(
                """
                INSERT INTO sensor_readings (sensor_id, timestamp, reading, sensor_type)
                VALUES (?, ?, ?, ?)
            """,
                (
                    reading["sensor_id"],
                    reading["timestamp"].isoformat(),
                    reading["reading"],
                    reading["sensor_type"],
                ),
            )

        conn.commit()
        print(f"  ✅ Inserted {len(collected_data['api_data']['users']['data'])} users")
        print(
            f"  ✅ Inserted {len(collected_data['api_data']['products']['data'])} products"
        )
        print(f"  ✅ Inserted {len(collected_data['sensor_data'])} sensor readings")

    # Query the database
    print("\n  Database Queries:")

    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    print(f"    Total users: {user_count}")

    cursor.execute("SELECT AVG(price) FROM products")
    avg_price = cursor.fetchone()[0]
    print(f"    Average product price: ${avg_price:.2f}")

    cursor.execute(
        "SELECT sensor_type, COUNT(*), AVG(reading) FROM sensor_readings GROUP BY sensor_type"
    )
    sensor_summary = cursor.fetchall()
    print(f"    Sensor readings by type:")
    for sensor_type, count, avg_reading in sensor_summary:
        print(f"      {sensor_type}: {count} readings, avg: {avg_reading:.2f}")

    print()

    # 2. File-based Storage
    print("2. FILE-BASED STORAGE:")
    print("-" * 20)

    # Create data directory
    data_dir = "collected_data"
    os.makedirs(data_dir, exist_ok=True)

    # Save API data as JSON
    api_file = os.path.join(data_dir, "api_data.json")
    with open(api_file, "w") as f:
        json.dump(collected_data["api_data"], f, indent=2, default=str)
    print(f"  ✅ Saved API data to {api_file}")

    # Save sensor data as CSV
    sensor_file = os.path.join(data_dir, "sensor_data.csv")
    df_sensor = pd.DataFrame(collected_data["sensor_data"])
    df_sensor.to_csv(sensor_file, index=False)
    print(f"  ✅ Saved sensor data to {sensor_file}")

    # Close database connection
    conn.close()


def demonstrate_data_pipelines():
    """Demonstrate data pipeline and ETL processes."""
    print("Data Pipelines and ETL Processes:")
    print("-" * 40)

    # 1. Extract Phase
    print("1. EXTRACT PHASE:")
    print("-" * 15)

    print("  Extracting data from multiple sources...")

    # Extract from database
    conn = sqlite3.connect("data_collection.db")
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    products_df = pd.read_sql_query("SELECT * FROM products", conn)
    sensor_df = pd.read_sql_query("SELECT * FROM sensor_readings", conn)

    print(f"    ✅ Extracted {len(users_df)} users from database")
    print(f"    ✅ Extracted {len(products_df)} products from database")
    print(f"    ✅ Extracted {len(sensor_df)} sensor readings from database")

    # Extract from files
    api_data = json.load(open("collected_data/api_data.json"))
    print(f"    ✅ Extracted API data: {len(api_data)} endpoints")

    print()

    # 2. Transform Phase
    print("2. TRANSFORM PHASE:")
    print("-" * 15)

    print("  Transforming and cleaning data...")

    # Transform users data
    users_df["created_at"] = pd.to_datetime(users_df["created_at"])
    users_df["name_length"] = users_df["name"].str.len()
    users_df["email_domain"] = users_df["email"].str.split("@").str[1]

    # Transform products data
    products_df["price_category"] = pd.cut(
        products_df["price"],
        bins=[0, 100, 500, 1000, float("inf")],
        labels=["Budget", "Mid-range", "Premium", "Luxury"],
    )

    # Transform sensor data
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    sensor_df["hour"] = sensor_df["timestamp"].dt.hour

    # Create aggregated sensor data
    sensor_agg = (
        sensor_df.groupby(["sensor_type", "hour"])
        .agg({"reading": ["mean", "std"]})
        .round(2)
    )
    sensor_agg.columns = ["_".join(col).strip() for col in sensor_agg.columns]
    sensor_agg = sensor_agg.reset_index()

    print(f"    ✅ Transformed users data: added name_length, email_domain")
    print(f"    ✅ Transformed products data: added price_category")
    print(f"    ✅ Transformed sensor data: added hour")
    print(f"    ✅ Created aggregated sensor data: {len(sensor_agg)} summary records")
    print()

    # 3. Load Phase
    print("3. LOAD PHASE:")
    print("-" * 15)

    print("  Loading transformed data to destination...")

    # Create transformed data directory
    transformed_dir = "transformed_data"
    os.makedirs(transformed_dir, exist_ok=True)

    # Save transformed data
    users_df.to_csv(os.path.join(transformed_dir, "transformed_users.csv"), index=False)
    products_df.to_csv(
        os.path.join(transformed_dir, "transformed_products.csv"), index=False
    )
    sensor_df.to_csv(
        os.path.join(transformed_dir, "transformed_sensor.csv"), index=False
    )
    sensor_agg.to_csv(
        os.path.join(transformed_dir, "sensor_aggregated.csv"), index=False
    )

    print(f"    ✅ Loaded transformed data to {transformed_dir}/")

    # Create data warehouse
    warehouse_conn = sqlite3.connect("data_warehouse.db")

    # Load data into warehouse
    users_df[["id", "name", "email_domain", "name_length", "created_at"]].to_sql(
        "dim_users", warehouse_conn, if_exists="replace", index=False
    )

    products_df[["id", "name", "category", "price_category", "price"]].to_sql(
        "dim_products", warehouse_conn, if_exists="replace", index=False
    )

    print(f"    ✅ Created data warehouse with 2 dimension tables")
    print(f"    ✅ Loaded {len(users_df)} users to dimension table")
    print(f"    ✅ Loaded {len(products_df)} products to dimension table")

    # Close connections
    conn.close()
    warehouse_conn.close()

    print("\n  Data Pipeline Summary:")
    print(f"    ✅ Collected data from 2 sources")
    print(f"    ✅ Stored in 1 database system")
    print(f"    ✅ Created 4 transformed datasets")
    print(f"    ✅ Established data warehouse")


if __name__ == "__main__":
    main()
