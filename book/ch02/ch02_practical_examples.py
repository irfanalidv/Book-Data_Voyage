#!/usr/bin/env python3
"""
Chapter 2 Practical Examples: Python for Data Science
Real-world applications and exercises to reinforce Python fundamentals
"""

import random
import time
from datetime import datetime, timedelta
import json
import csv

def main():
    print("🚀 CHAPTER 2 PRACTICAL EXAMPLES: PYTHON FOR DATA SCIENCE")
    print("=" * 80)
    print("This script demonstrates practical Python applications for data science.")
    print("=" * 80)
    
    # Example 1: Data Processing Pipeline
    print("\n1. DATA PROCESSING PIPELINE")
    print("-" * 50)
    demonstrate_data_processing_pipeline()
    
    # Example 2: Data Validation and Cleaning
    print("\n2. DATA VALIDATION AND CLEANING")
    print("-" * 50)
    demonstrate_data_validation_and_cleaning()
    
    # Example 3: Data Aggregation and Analysis
    print("\n3. DATA AGGREGATION AND ANALYSIS")
    print("-" * 50)
    demonstrate_data_aggregation_and_analysis()
    
    # Example 4: File Processing and Data Export
    print("\n4. FILE PROCESSING AND DATA EXPORT")
    print("-" * 50)
    demonstrate_file_processing_and_export()
    
    # Example 5: Performance Optimization
    print("\n5. PERFORMANCE OPTIMIZATION")
    print("-" * 50)
    demonstrate_performance_optimization()
    
    print("\n" + "=" * 80)
    print("🎉 PRACTICAL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("Key skills demonstrated:")
    print("• Data processing and transformation")
    print("• Data validation and error handling")
    print("• Aggregation and statistical analysis")
    print("• File I/O and data export")
    print("• Performance optimization techniques")
    print("\nPractice these concepts with your own datasets!")

def demonstrate_data_processing_pipeline():
    """Demonstrate a complete data processing pipeline."""
    print("Processing sales data through multiple stages...")
    
    # Stage 1: Generate raw data
    raw_data = generate_sample_sales_data(100)
    print(f"✅ Generated {len(raw_data)} raw sales records")
    
    # Stage 2: Clean and validate data
    cleaned_data = clean_sales_data(raw_data)
    print(f"✅ Cleaned data: {len(cleaned_data)} valid records")
    
    # Stage 3: Transform and enrich data
    enriched_data = enrich_sales_data(cleaned_data)
    print(f"✅ Enriched data with calculated fields")
    
    # Stage 4: Aggregate and summarize
    summary = aggregate_sales_data(enriched_data)
    print(f"✅ Generated summary statistics")
    
    # Display results
    print(f"\nData Processing Results:")
    print(f"  Raw records: {len(raw_data)}")
    print(f"  Valid records: {len(cleaned_data)}")
    print(f"  Invalid records: {len(raw_data) - len(cleaned_data)}")
    print(f"  Total revenue: ${summary['total_revenue']:,.2f}")
    print(f"  Average order value: ${summary['avg_order_value']:,.2f}")
    print(f"  Top product: {summary['top_product']}")

def generate_sample_sales_data(count):
    """Generate sample sales data for demonstration."""
    products = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    
    data = []
    for i in range(count):
        # Simulate some data quality issues
        if random.random() < 0.1:  # 10% missing product names
            product = None
        else:
            product = random.choice(products)
        
        if random.random() < 0.05:  # 5% negative prices
            price = random.uniform(-100, 100)
        else:
            price = random.uniform(50, 2000)
        
        if random.random() < 0.08:  # 8% missing dates
            date = None
        else:
            date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        record = {
            "id": i + 1,
            "product": product,
            "price": round(price, 2),
            "quantity": random.randint(1, 5),
            "city": random.choice(cities),
            "date": date,
            "customer_id": f"CUST_{random.randint(1000, 9999)}"
        }
        data.append(record)
    
    return data

def clean_sales_data(data):
    """Clean and validate sales data."""
    cleaned = []
    
    for record in data:
        # Check for required fields
        if record["product"] is None:
            continue
        
        if record["price"] <= 0:
            continue
        
        if record["date"] is None:
            continue
        
        # Validate data types and ranges
        if not isinstance(record["quantity"], int) or record["quantity"] <= 0:
            continue
        
        if not isinstance(record["price"], (int, float)):
            continue
        
        # Add validation flag
        record["is_valid"] = True
        cleaned.append(record)
    
    return cleaned

def enrich_sales_data(data):
    """Add calculated fields to sales data."""
    for record in data:
        # Calculate total amount
        record["total_amount"] = record["price"] * record["quantity"]
        
        # Add category based on product
        if record["product"] in ["Laptop", "Phone", "Tablet"]:
            record["category"] = "Electronics"
        elif record["product"] in ["Monitor", "Keyboard", "Mouse"]:
            record["category"] = "Accessories"
        else:
            record["category"] = "Other"
        
        # Add season based on date
        month = record["date"].month
        if month in [12, 1, 2]:
            record["season"] = "Winter"
        elif month in [3, 4, 5]:
            record["season"] = "Spring"
        elif month in [6, 7, 8]:
            record["season"] = "Summer"
        else:
            record["season"] = "Fall"
    
    return data

def aggregate_sales_data(data):
    """Aggregate and summarize sales data."""
    if not data:
        return {}
    
    # Basic statistics
    total_revenue = sum(record["total_amount"] for record in data)
    avg_order_value = total_revenue / len(data)
    
    # Product analysis
    product_sales = {}
    for record in data:
        product = record["product"]
        if product not in product_sales:
            product_sales[product] = 0
        product_sales[product] += record["total_amount"]
    
    top_product = max(product_sales, key=product_sales.get)
    
    # Category analysis
    category_sales = {}
    for record in data:
        category = record["category"]
        if category not in category_sales:
            category_sales[category] = 0
        category_sales[category] += record["total_amount"]
    
    # Season analysis
    season_sales = {}
    for record in data:
        season = record["season"]
        if season not in season_sales:
            season_sales[season] = 0
        season_sales[season] += record["total_amount"]
    
    return {
        "total_revenue": total_revenue,
        "avg_order_value": avg_order_value,
        "total_orders": len(data),
        "top_product": top_product,
        "product_sales": product_sales,
        "category_sales": category_sales,
        "season_sales": season_sales
    }

def demonstrate_data_validation_and_cleaning():
    """Demonstrate data validation and cleaning techniques."""
    print("Creating and validating a customer dataset...")
    
    # Create sample customer data with various issues
    customers = [
        {"id": 1, "name": "John Doe", "email": "john@example.com", "age": 30, "city": "New York"},
        {"id": 2, "name": "", "email": "invalid-email", "age": -5, "city": "Boston"},
        {"id": 3, "name": "Jane Smith", "email": "jane@example.com", "age": 25, "city": "Chicago"},
        {"id": 4, "name": "Bob Johnson", "email": "bob@example.com", "age": 150, "city": ""},
        {"id": 5, "name": "Alice Brown", "email": "alice@example.com", "age": 35, "city": "Los Angeles"}
    ]
    
    print(f"Original data: {len(customers)} customers")
    
    # Validation rules
    validation_rules = {
        "name": lambda x: isinstance(x, str) and len(x.strip()) > 0,
        "email": lambda x: isinstance(x, str) and "@" in x and "." in x,
        "age": lambda x: isinstance(x, (int, float)) and 0 < x < 120,
        "city": lambda x: isinstance(x, str) and len(x.strip()) > 0
    }
    
    # Validate and clean data
    valid_customers = []
    invalid_customers = []
    
    for customer in customers:
        is_valid = True
        errors = []
        
        for field, rule in validation_rules.items():
            if field in customer and not rule(customer[field]):
                is_valid = False
                errors.append(f"Invalid {field}: {customer[field]}")
        
        if is_valid:
            # Clean the data
            cleaned_customer = customer.copy()
            cleaned_customer["name"] = customer["name"].strip()
            cleaned_customer["city"] = customer["city"].strip()
            valid_customers.append(cleaned_customer)
        else:
            customer["errors"] = errors
            invalid_customers.append(customer)
    
    print(f"✅ Valid customers: {len(valid_customers)}")
    print(f"❌ Invalid customers: {len(invalid_customers)}")
    
    if invalid_customers:
        print("\nValidation errors found:")
        for customer in invalid_customers:
            print(f"  Customer {customer['id']}: {', '.join(customer['errors'])}")

def demonstrate_data_aggregation_and_analysis():
    """Demonstrate data aggregation and analysis techniques."""
    print("Analyzing sales performance by various dimensions...")
    
    # Generate sample data
    sales_data = generate_sample_sales_data(50)
    cleaned_data = clean_sales_data(sales_data)
    enriched_data = enrich_sales_data(cleaned_data)
    
    if not enriched_data:
        print("No valid data to analyze")
        return
    
    # 1. Time-based analysis
    print("\n1. Time-based Analysis:")
    monthly_sales = {}
    for record in enriched_data:
        month_key = record["date"].strftime("%Y-%m")
        if month_key not in monthly_sales:
            monthly_sales[month_key] = 0
        monthly_sales[month_key] += record["total_amount"]
    
    print("Monthly sales:")
    for month, sales in sorted(monthly_sales.items()):
        print(f"  {month}: ${sales:,.2f}")
    
    # 2. Geographic analysis
    print("\n2. Geographic Analysis:")
    city_sales = {}
    for record in enriched_data:
        city = record["city"]
        if city not in city_sales:
            city_sales[city] = {"revenue": 0, "orders": 0}
        city_sales[city]["revenue"] += record["total_amount"]
        city_sales[city]["orders"] += 1
    
    print("Sales by city:")
    for city, data in sorted(city_sales.items(), key=lambda x: x[1]["revenue"], reverse=True):
        avg_order = data["revenue"] / data["orders"]
        print(f"  {city}: ${data['revenue']:,.2f} ({data['orders']} orders, avg: ${avg_order:.2f})")
    
    # 3. Product performance analysis
    print("\n3. Product Performance Analysis:")
    product_stats = {}
    for record in enriched_data:
        product = record["product"]
        if product not in product_stats:
            product_stats[product] = {"revenue": 0, "quantity": 0, "orders": 0}
        product_stats[product]["revenue"] += record["total_amount"]
        product_stats[product]["quantity"] += record["quantity"]
        product_stats[product]["orders"] += 1
    
    print("Product performance:")
    for product, stats in sorted(product_stats.items(), key=lambda x: x[1]["revenue"], reverse=True):
        avg_price = stats["revenue"] / stats["quantity"]
        print(f"  {product}: ${stats['revenue']:,.2f} ({stats['orders']} orders, avg price: ${avg_price:.2f})")

def demonstrate_file_processing_and_export():
    """Demonstrate file processing and data export capabilities."""
    print("Processing data and exporting to multiple formats...")
    
    # Generate and process data
    sales_data = generate_sample_sales_data(25)
    cleaned_data = clean_sales_data(sales_data)
    enriched_data = enrich_sales_data(cleaned_data)
    
    if not enriched_data:
        print("No valid data to export")
        return
    
    # Export to JSON
    json_filename = "sales_data.json"
    with open(json_filename, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        export_data = []
        for record in enriched_data:
            export_record = record.copy()
            export_record["date"] = record["date"].isoformat()
            export_data.append(export_record)
        
        json.dump(export_data, f, indent=2)
    print(f"✅ Exported to {json_filename}")
    
    # Export to CSV
    csv_filename = "sales_data.csv"
    with open(csv_filename, 'w', newline='') as f:
        if export_data:
            writer = csv.DictWriter(f, fieldnames=export_data[0].keys())
            writer.writeheader()
            writer.writerows(export_data)
    print(f"✅ Exported to {csv_filename}")
    
    # Export summary report
    summary = aggregate_sales_data(enriched_data)
    report_filename = "sales_summary.txt"
    with open(report_filename, 'w') as f:
        f.write("SALES SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total orders: {summary['total_orders']}\n")
        f.write(f"Total revenue: ${summary['total_revenue']:,.2f}\n")
        f.write(f"Average order value: ${summary['avg_order_value']:,.2f}\n")
        f.write(f"Top product: {summary['top_product']}\n\n")
        
        f.write("Category breakdown:\n")
        for category, sales in summary['category_sales'].items():
            f.write(f"  {category}: ${sales:,.2f}\n")
        
        f.write("\nSeasonal breakdown:\n")
        for season, sales in summary['season_sales'].items():
            f.write(f"  {season}: ${sales:,.2f}\n")
    
    print(f"✅ Exported summary report to {report_filename}")
    
    # Clean up temporary files
    import os
    for filename in [json_filename, csv_filename, report_filename]:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🧹 Cleaned up {filename}")

def demonstrate_performance_optimization():
    """Demonstrate performance optimization techniques."""
    print("Comparing different approaches for data processing...")
    
    # Generate large dataset
    large_dataset = generate_sample_sales_data(10000)
    print(f"Generated {len(large_dataset)} records for performance testing")
    
    # Method 1: Traditional loop
    start_time = time.time()
    total_revenue_loop = 0
    for record in large_dataset:
        if record.get("price", 0) > 0:
            total_revenue_loop += record.get("price", 0) * record.get("quantity", 0)
    loop_time = time.time() - start_time
    
    # Method 2: List comprehension
    start_time = time.time()
    valid_records = [record for record in large_dataset if record.get("price", 0) > 0]
    total_revenue_comp = sum(record.get("price", 0) * record.get("quantity", 0) for record in valid_records)
    comp_time = time.time() - start_time
    
    # Method 3: Generator expression
    start_time = time.time()
    total_revenue_gen = sum(
        record.get("price", 0) * record.get("quantity", 0) 
        for record in large_dataset 
        if record.get("price", 0) > 0
    )
    gen_time = time.time() - start_time
    
    print(f"\nPerformance Results:")
    print(f"  Traditional loop: {loop_time:.4f} seconds")
    print(f"  List comprehension: {comp_time:.4f} seconds")
    print(f"  Generator expression: {gen_time:.4f} seconds")
    
    print(f"\nResults verification:")
    print(f"  Loop method: ${total_revenue_loop:,.2f}")
    print(f"  Comprehension method: ${total_revenue_comp:,.2f}")
    print(f"  Generator method: ${total_revenue_gen:,.2f}")
    
    # Memory usage demonstration
    print(f"\nMemory efficiency:")
    print(f"  Original dataset: {len(large_dataset)} records")
    print(f"  Filtered records: {len(valid_records)} records")
    print(f"  Memory saved with generators: ~{(len(large_dataset) - len(valid_records)) / len(large_dataset) * 100:.1f}%")

if __name__ == "__main__":
    main()
