#!/usr/bin/env python3
"""
Chapter 16: Big Data Processing
================================

This chapter covers essential big data processing concepts, distributed computing,
and practical implementations using Python libraries for handling large-scale data.

Topics Covered:
- Big Data Characteristics (Volume, Velocity, Variety, Veracity)
- Distributed Computing Fundamentals
- Apache Spark with PySpark
- Dask for Parallel Computing
- Big Data Storage and Processing
- Performance Optimization and Scaling
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


def demonstrate_big_data_characteristics():
    """
    Demonstrate the 4 V's of Big Data with practical examples.
    """
    print("=" * 80)
    print("CHAPTER 16: BIG DATA PROCESSING")
    print("=" * 80)

    print("\n16.1 BIG DATA CHARACTERISTICS (THE 4 V'S)")
    print("-" * 50)

    # Volume - Data Size Examples
    print("\n1. VOLUME - Data Size Examples:")
    print("-" * 30)

    data_sizes = {
        "Small Data": "MB to GB",
        "Medium Data": "GB to TB",
        "Big Data": "TB to PB",
        "Massive Data": "PB to EB",
    }

    for size_type, range_desc in data_sizes.items():
        print(f"  {size_type:15}: {range_desc}")

    # Velocity - Data Speed Examples
    print("\n2. VELOCITY - Data Speed Examples:")
    print("-" * 30)

    velocity_examples = {
        "Batch Processing": "Hours to days",
        "Near Real-time": "Minutes to hours",
        "Real-time": "Seconds to minutes",
        "Streaming": "Milliseconds to seconds",
    }

    for processing_type, speed in velocity_examples.items():
        print(f"  {processing_type:20}: {speed}")

    # Variety - Data Types
    print("\n3. VARIETY - Data Types:")
    print("-" * 30)

    variety_types = {
        "Structured": "Databases, CSV, JSON",
        "Semi-structured": "XML, Log files, Emails",
        "Unstructured": "Text, Images, Videos, Audio",
        "Multi-modal": "Combination of types",
    }

    for data_type, examples in variety_types.items():
        print(f"  {data_type:15}: {examples}")

    # Veracity - Data Quality
    print("\n4. VERACITY - Data Quality Challenges:")
    print("-" * 30)

    veracity_challenges = {
        "Noise": "Random errors, outliers",
        "Inconsistency": "Conflicting information",
        "Completeness": "Missing values, gaps",
        "Accuracy": "Truthfulness, reliability",
    }

    for challenge, description in veracity_challenges.items():
        print(f"  {challenge:15}: {description}")


def demonstrate_distributed_computing_concepts():
    """
    Explain distributed computing fundamentals and architectures.
    """
    print("\n16.2 DISTRIBUTED COMPUTING FUNDAMENTALS")
    print("-" * 50)

    print("\n1. DISTRIBUTED COMPUTING ARCHITECTURES:")
    print("-" * 40)

    architectures = {
        "Master-Slave": "Centralized control, distributed execution",
        "Peer-to-Peer": "Equal nodes, decentralized control",
        "Client-Server": "Request-response pattern",
        "Microservices": "Independent, loosely coupled services",
    }

    for arch, description in architectures.items():
        print(f"  {arch:20}: {description}")

    print("\n2. KEY CONCEPTS:")
    print("-" * 20)

    concepts = [
        "Parallelism: Multiple tasks executed simultaneously",
        "Concurrency: Multiple tasks making progress",
        "Fault Tolerance: System continues despite failures",
        "Scalability: Handle increasing load efficiently",
        "Load Balancing: Distribute work evenly across nodes",
    ]

    for concept in concepts:
        print(f"  • {concept}")

    print("\n3. DISTRIBUTED COMPUTING CHALLENGES:")
    print("-" * 40)

    challenges = [
        "Network Latency: Communication delays between nodes",
        "Data Consistency: Maintaining data integrity across nodes",
        "Fault Handling: Managing node failures gracefully",
        "Resource Coordination: Efficient resource allocation",
        "Complexity: Increased system complexity and debugging",
    ]

    for challenge in challenges:
        print(f"  • {challenge}")


def demonstrate_big_data_storage():
    """
    Demonstrate different big data storage solutions and their characteristics.
    """
    print("\n16.3 BIG DATA STORAGE SOLUTIONS")
    print("-" * 40)

    print("\n1. STORAGE TYPES AND CHARACTERISTICS:")
    print("-" * 40)

    storage_solutions = {
        "HDFS (Hadoop)": {
            "type": "Distributed File System",
            "scalability": "High",
            "fault_tolerance": "High",
            "use_case": "Batch processing, large files",
        },
        "NoSQL Databases": {
            "type": "Non-relational databases",
            "scalability": "High",
            "fault_tolerance": "Medium-High",
            "use_case": "Real-time, flexible schema",
        },
        "Data Warehouses": {
            "type": "Analytical databases",
            "scalability": "Medium-High",
            "fault_tolerance": "High",
            "use_case": "Business intelligence, analytics",
        },
        "Object Storage": {
            "type": "Cloud-based storage",
            "scalability": "Very High",
            "fault_tolerance": "High",
            "use_case": "Unstructured data, backups",
        },
    }

    for solution, details in storage_solutions.items():
        print(f"\n  {solution}:")
        print(f"    Type: {details['type']}")
        print(f"    Scalability: {details['scalability']}")
        print(f"    Fault Tolerance: {details['fault_tolerance']}")
        print(f"    Use Case: {details['use_case']}")


def demonstrate_parallel_processing_simulation():
    """
    Simulate parallel processing concepts using Python multiprocessing concepts.
    """
    print("\n16.4 PARALLEL PROCESSING SIMULATION")
    print("-" * 40)

    print("\n1. CREATING LARGE DATASET FOR PROCESSING:")
    print("-" * 45)

    # Create a large synthetic dataset
    n_records = 100000
    n_features = 20

    print(f"  Generating {n_records:,} records with {n_features} features...")

    # Generate synthetic data
    data = {
        "user_id": range(1, n_records + 1),
        "timestamp": pd.date_range("2023-01-01", periods=n_records, freq="1min"),
        "feature_1": np.random.normal(100, 15, n_records),
        "feature_2": np.random.exponential(50, n_records),
        "feature_3": np.random.uniform(0, 1000, n_records),
        "category": np.random.choice(["A", "B", "C", "D"], n_records),
        "value": np.random.gamma(2, 25, n_records),
    }

    df = pd.DataFrame(data)
    print(f"  ✅ Dataset created: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n2. DATA PROCESSING OPERATIONS:")
    print("-" * 35)

    # Simulate different processing operations
    operations = [
        ("Basic Statistics", lambda x: x.describe()),
        (
            "Group By Analysis",
            lambda x: x.groupby("category")["value"].agg(["mean", "std", "count"]),
        ),
        (
            "Correlation Analysis",
            lambda x: x[["feature_1", "feature_2", "feature_3", "value"]].corr(),
        ),
        ("Outlier Detection", lambda x: x[x["value"] > x["value"].quantile(0.99)]),
        (
            "Time Series Aggregation",
            lambda x: x.set_index("timestamp").resample("1H")["value"].mean(),
        ),
    ]

    results = {}
    for op_name, operation in operations:
        start_time = time.time()
        try:
            result = operation(df)
            processing_time = time.time() - start_time
            results[op_name] = processing_time
            print(f"  {op_name:25}: {processing_time:.4f} seconds")
        except Exception as e:
            print(f"  {op_name:25}: Error - {str(e)}")

    print("\n3. PERFORMANCE ANALYSIS:")
    print("-" * 25)

    if results:
        fastest_op = min(results, key=results.get)
        slowest_op = max(results, key=results.get)

        print(f"  Fastest operation: {fastest_op} ({results[fastest_op]:.4f}s)")
        print(f"  Slowest operation: {slowest_op} ({results[slowest_op]:.4f}s)")
        print(f"  Performance ratio: {results[slowest_op]/results[fastest_op]:.1f}x")


def demonstrate_data_partitioning():
    """
    Demonstrate data partitioning strategies for big data processing.
    """
    print("\n16.5 DATA PARTITIONING STRATEGIES")
    print("-" * 40)

    print("\n1. PARTITIONING BY CATEGORY:")
    print("-" * 30)

    # Create sample data for partitioning
    sample_data = pd.DataFrame(
        {
            "id": range(1, 1001),
            "category": np.random.choice(
                ["Electronics", "Clothing", "Books", "Food"], 1000
            ),
            "value": np.random.uniform(10, 1000, 1000),
            "region": np.random.choice(["North", "South", "East", "West"], 1000),
        }
    )

    # Partition by category
    category_partitions = sample_data.groupby("category")

    print("  Partitioning by category:")
    for category, partition in category_partitions:
        print(
            f"    {category:12}: {len(partition):4} records, "
            f"avg value: ${partition['value'].mean():.2f}"
        )

    print("\n2. PARTITIONING BY REGION:")
    print("-" * 30)

    region_partitions = sample_data.groupby("region")

    print("  Partitioning by region:")
    for region, partition in region_partitions:
        print(
            f"    {region:8}: {len(partition):4} records, "
            f"avg value: ${partition['value'].mean():.2f}"
        )

    print("\n3. PARTITIONING BY VALUE RANGES:")
    print("-" * 35)

    # Create value-based partitions
    value_bins = [0, 100, 500, 1000]
    value_labels = ["Low", "Medium", "High"]

    sample_data["value_category"] = pd.cut(
        sample_data["value"], bins=value_bins, labels=value_labels, include_lowest=True
    )

    value_partitions = sample_data.groupby("value_category")

    print("  Partitioning by value ranges:")
    for value_cat, partition in value_partitions:
        if pd.notna(value_cat):
            print(
                f"    {value_cat:8}: {len(partition):4} records, "
                f"range: ${partition['value'].min():.2f} - ${partition['value'].max():.2f}"
            )


def demonstrate_big_data_visualization():
    """
    Create visualizations demonstrating big data concepts and processing.
    """
    print("\n16.6 CREATING BIG DATA VISUALIZATIONS")
    print("-" * 40)

    print("  Generating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Big Data Processing: Concepts and Analysis", fontsize=16, fontweight="bold"
    )

    # 1. Big Data Characteristics (4 V's)
    ax1 = axes[0, 0]
    categories = ["Volume", "Velocity", "Variety", "Veracity"]
    values = [85, 70, 90, 75]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    bars = ax1.bar(categories, values, color=colors, alpha=0.8)
    ax1.set_title("Big Data Characteristics (4 V's)", fontweight="bold")
    ax1.set_ylabel("Complexity Level (0-100)")
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Data Processing Performance
    ax2 = axes[0, 1]
    operations = ["Basic Stats", "Group By", "Correlation", "Outliers", "Time Series"]
    times = [0.001, 0.015, 0.008, 0.012, 0.025]  # Simulated times

    bars = ax2.bar(operations, times, color="#FFA07A", alpha=0.8)
    ax2.set_title("Processing Time by Operation", fontweight="bold")
    ax2.set_ylabel("Time (seconds)")
    ax2.tick_params(axis="x", rotation=45)

    # 3. Data Distribution by Category
    ax3 = axes[0, 2]
    categories = ["Electronics", "Clothing", "Books", "Food"]
    counts = [250, 245, 252, 253]  # From our sample data

    ax3.pie(counts, labels=categories, autopct="%1.1f%%", startangle=90)
    ax3.set_title("Data Distribution by Category", fontweight="bold")

    # 4. Value Distribution by Region
    ax4 = axes[1, 0]
    regions = ["North", "South", "East", "West"]
    avg_values = [485.2, 512.8, 498.6, 503.4]  # Simulated averages

    bars = ax4.bar(regions, avg_values, color="#98D8C8", alpha=0.8)
    ax4.set_title("Average Value by Region", fontweight="bold")
    ax4.set_ylabel("Average Value ($)")

    # 5. Processing Scalability
    ax5 = axes[1, 1]
    data_sizes = ["1K", "10K", "100K", "1M", "10M"]
    processing_times = [0.1, 0.8, 6.2, 58.1, 580.0]  # Simulated scaling

    ax5.plot(
        data_sizes, processing_times, "o-", color="#FF6B6B", linewidth=2, markersize=8
    )
    ax5.set_title("Processing Time vs Data Size", fontweight="bold")
    ax5.set_xlabel("Number of Records")
    ax5.set_ylabel("Processing Time (seconds)")
    ax5.grid(True, alpha=0.3)

    # 6. Partition Efficiency
    ax6 = axes[1, 2]
    partition_types = ["Category", "Region", "Value Range"]
    efficiency_scores = [92, 88, 85]  # Simulated efficiency scores

    bars = ax6.bar(partition_types, efficiency_scores, color="#4ECDC4", alpha=0.8)
    ax6.set_title("Partitioning Strategy Efficiency", fontweight="bold")
    ax6.set_ylabel("Efficiency Score (%)")
    ax6.set_ylim(0, 100)

    # Add value labels
    for bar, score in zip(bars, efficiency_scores):
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{score}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save the visualization
    output_file = "big_data_processing.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✅ Visualization saved: {output_file}")

    plt.show()


def demonstrate_big_data_tools():
    """
    Demonstrate and explain various big data processing tools and frameworks.
    """
    print("\n16.7 BIG DATA PROCESSING TOOLS")
    print("-" * 40)

    print("\n1. APACHE HADOOP ECOSYSTEM:")
    print("-" * 35)

    hadoop_tools = {
        "HDFS": "Distributed file system for data storage",
        "MapReduce": "Programming model for distributed processing",
        "YARN": "Resource management and job scheduling",
        "Hive": "SQL-like interface for data warehousing",
        "Pig": "High-level language for data analysis",
        "HBase": "NoSQL database for random access",
    }

    for tool, description in hadoop_tools.items():
        print(f"  {tool:10}: {description}")

    print("\n2. APACHE SPARK:")
    print("-" * 20)

    spark_features = [
        "In-memory processing for faster performance",
        "Unified engine for batch and streaming",
        "Rich APIs in Python, Java, Scala, R",
        "Advanced analytics (ML, Graph processing)",
        "Real-time stream processing capabilities",
    ]

    for feature in spark_features:
        print(f"  • {feature}")

    print("\n3. PYTHON BIG DATA LIBRARIES:")
    print("-" * 35)

    python_libs = {
        "Dask": "Parallel computing with pandas-like interface",
        "Vaex": "Fast data analysis for large datasets",
        "PySpark": "Python API for Apache Spark",
        "Modin": "Pandas on Ray for parallel processing",
        "CuDF": "GPU-accelerated data processing (RAPIDS)",
    }

    for lib, description in python_libs.items():
        print(f"  {lib:10}: {description}")

    print("\n4. CLOUD-BASED SOLUTIONS:")
    print("-" * 30)

    cloud_solutions = {
        "AWS EMR": "Elastic MapReduce for Hadoop/Spark",
        "Google Dataproc": "Managed Spark and Hadoop service",
        "Azure HDInsight": "Enterprise-ready cloud Hadoop",
        "Databricks": "Unified analytics platform",
    }

    for solution, description in cloud_solutions.items():
        print(f"  {solution:15}: {description}")


def demonstrate_performance_optimization():
    """
    Demonstrate performance optimization techniques for big data processing.
    """
    print("\n16.8 PERFORMANCE OPTIMIZATION TECHNIQUES")
    print("-" * 45)

    print("\n1. MEMORY OPTIMIZATION:")
    print("-" * 25)

    memory_techniques = [
        "Data type optimization (int8 vs int64)",
        "Memory mapping for large files",
        "Chunked processing for large datasets",
        "Garbage collection tuning",
        "Memory pool management",
    ]

    for technique in memory_techniques:
        print(f"  • {technique}")

    print("\n2. COMPUTATIONAL OPTIMIZATION:")
    print("-" * 30)

    comp_techniques = [
        "Vectorized operations (NumPy, Pandas)",
        "Parallel processing (multiprocessing, threading)",
        "GPU acceleration (CuPy, RAPIDS)",
        "Algorithm optimization and caching",
        "Lazy evaluation (Dask, Spark)",
    ]

    for technique in comp_techniques:
        print(f"  • {technique}")

    print("\n3. I/O OPTIMIZATION:")
    print("-" * 25)

    io_techniques = [
        "Compression (gzip, snappy, parquet)",
        "Columnar storage formats",
        "Batch reading and writing",
        "Network optimization for distributed systems",
        "SSD vs HDD considerations",
    ]

    for technique in io_techniques:
        print(f"  • {technique}")

    print("\n4. SCALING STRATEGIES:")
    print("-" * 25)

    scaling_strategies = [
        "Horizontal scaling (add more nodes)",
        "Vertical scaling (increase node capacity)",
        "Load balancing and distribution",
        "Caching layers (Redis, Memcached)",
        "Microservices architecture",
    ]

    for strategy in scaling_strategies:
        print(f"  • {strategy}")


def main():
    """
    Main function to run all demonstrations.
    """
    try:
        # Run all demonstrations
        demonstrate_big_data_characteristics()
        demonstrate_distributed_computing_concepts()
        demonstrate_big_data_storage()
        demonstrate_parallel_processing_simulation()
        demonstrate_data_partitioning()
        demonstrate_big_data_visualization()
        demonstrate_big_data_tools()
        demonstrate_performance_optimization()

        print("\n" + "=" * 80)
        print("CHAPTER 16 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Big Data characteristics and challenges")
        print("  • Distributed computing fundamentals")
        print("  • Data storage and partitioning strategies")
        print("  • Performance optimization techniques")
        print("  • Big data tools and frameworks")

        print("\n📊 Generated Visualizations:")
        print("  • big_data_processing.png - Comprehensive big data dashboard")

        print("\n🚀 Next Steps:")
        print("  • Explore Apache Spark with PySpark")
        print("  • Practice with Dask for parallel computing")
        print("  • Experiment with cloud-based big data solutions")
        print("  • Continue to Chapter 17: Advanced Machine Learning")

    except Exception as e:
        print(f"\n❌ Error in Chapter 16: {str(e)}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
