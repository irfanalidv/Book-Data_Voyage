#!/usr/bin/env python3
"""
Simple script to add image displays to CHAPTERX_SUMMARY.md files.
"""

import os
import re


def update_summary_files():
    """Update all CHAPTERX_SUMMARY.md files with image displays."""

    # Define chapters and their image files
    chapters_with_images = {
        "ch01": [
            "data_science_venn_diagram.png",
            "ethics_framework.png",
            "correlation_heatmap.png",
            "daily_sales_trend.png",
            "data_science_workflow.png",
            "industry_applications.png",
            "sales_distribution.png",
            "skills_radar_chart.png",
        ],
        "ch03": [
            "ab_testing.png",
            "correlation_regression.png",
            "data_distribution.png",
            "financial_analysis.png",
            "hypothesis_testing.png",
            "linear_algebra.png",
            "probability_distributions.png",
            "quality_control.png",
        ],
        "ch06": ["data_preprocessing.png"],
        "ch07": ["univariate_analysis.png", "bivariate_analysis.png"],
        "ch08": ["sampling_distributions.png", "hypothesis_testing.png"],
        "ch09": ["model_evaluation.png"],
        "ch10": ["feature_engineering_selection.png"],
        "ch11": [
            "unsupervised_datasets.png",
            "dimensionality_reduction.png",
            "clustering_results.png",
        ],
        "ch12": [
            "activation_functions.png",
            "neural_network_results.png",
            "training_optimization.png",
        ],
        "ch13": [
            "text_preprocessing.png",
            "text_representation.png",
            "nlp_applications.png",
        ],
        "ch14": [
            "image_processing.png",
            "feature_extraction.png",
            "cv_applications.png",
        ],
        "ch15": [
            "time_series_components.png",
            "seasonal_decomposition.png",
            "stationarity_analysis.png",
            "time_series_forecasting.png",
            "forecast_residuals.png",
        ],
        "ch16": ["big_data_processing.png"],
        "ch17": ["advanced_machine_learning.png"],
        "ch18": ["model_deployment_mlops.png"],
        "ch19": ["real_world_case_studies.png"],
        "ch20": ["data_science_ethics.png"],
        "ch21": ["communication_storytelling.png"],
        "ch22": ["portfolio_development.png"],
        "ch23": ["career_development.png"],
        "ch24": ["advanced_career_specializations.png"],
        "ch25": ["python_library_development.png"],
    }

    for chapter, image_files in chapters_with_images.items():
        # Extract chapter number
        chapter_num = chapter[2:]  # Remove 'ch' prefix
        summary_path = f"book/{chapter}/CHAPTER{chapter_num}_SUMMARY.md"

        if not os.path.exists(summary_path):
            print(f"⚠️  Summary file {summary_path} not found, skipping...")
            continue

        print(f"📁 Processing {chapter} summary...")
        update_single_summary(summary_path, image_files, chapter)


def update_single_summary(summary_path, image_files, chapter):
    """Update a single summary file with image displays."""

    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if images section already exists
    if "![Image" in content or "![Visualization" in content:
        print(f"    ✅ Already has image displays")
        return

    # Find the Generated Outputs section
    if "## Generated Outputs" in content:
        # Add images after the existing content
        new_section = create_images_section(image_files, chapter)

        # Insert after Generated Outputs section
        content = re.sub(
            r"(## Generated Outputs\n.*?)(\n## )",
            r"\1" + new_section + r"\2",
            content,
            flags=re.DOTALL,
        )

        # Write updated content
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"    ✅ Updated with image displays")
    else:
        print(f"    ⚠️  Could not find Generated Outputs section")


def create_images_section(image_files, chapter):
    """Create the images section content."""

    if len(image_files) == 1:
        # Single image
        image_file = image_files[0]
        title = get_image_title(image_file)
        return f"""

### {title}

![{title}]({image_file})

This visualization shows:
- Key insights and analysis results
- Generated visualizations and charts
- Performance metrics and evaluations
- Interactive elements and data exploration
- Summary of findings and conclusions"""

    else:
        # Multiple images
        content = f"\n### Generated Visualizations\n\n"
        content += "This chapter generates multiple visualizations:\n\n"

        for i, image_file in enumerate(image_files, 1):
            image_title = get_image_title(image_file)
            content += f"#### {i}. {image_title}\n\n"
            content += f"![{image_title}]({image_file})\n\n"
            content += f"- {get_image_description(image_file)}\n\n"

        return content


def get_image_title(image_file):
    """Get a human-readable title for the image file."""
    title = image_file.replace(".png", "").replace("_", " ").title()

    title_mapping = {
        "cv_applications": "Computer Vision Applications",
        "nlp_applications": "NLP Applications",
        "mlops": "MLOps",
        "eda": "Exploratory Data Analysis",
    }

    return title_mapping.get(title.lower(), title)


def get_image_description(image_file):
    """Get a description for the image file."""
    descriptions = {
        "data_science_venn_diagram": "Data science components and their relationships",
        "ethics_framework": "Ethical decision-making framework for data science",
        "correlation_heatmap": "Feature correlation analysis and visualization",
        "daily_sales_trend": "Time series analysis of sales data",
        "data_science_workflow": "Complete data science project workflow",
        "industry_applications": "Data science applications across industries",
        "sales_distribution": "Statistical distribution of sales data",
        "skills_radar_chart": "Data science skills assessment and development",
        "data_preprocessing": "Data cleaning and preprocessing pipeline",
        "univariate_analysis": "Single variable analysis and distributions",
        "bivariate_analysis": "Two variable relationship analysis",
        "sampling_distributions": "Statistical sampling and distribution analysis",
        "hypothesis_testing": "Hypothesis testing procedures and results",
        "model_evaluation": "Machine learning model performance evaluation",
        "feature_engineering_selection": "Feature engineering and selection techniques",
        "unsupervised_datasets": "Unsupervised learning dataset analysis",
        "dimensionality_reduction": "Dimensionality reduction techniques and results",
        "clustering_results": "Clustering algorithm results and analysis",
        "activation_functions": "Neural network activation functions",
        "neural_network_results": "Deep learning model training and results",
        "training_optimization": "Neural network training optimization",
        "text_preprocessing": "Natural language text preprocessing pipeline",
        "text_representation": "Text representation and vectorization",
        "nlp_applications": "Natural language processing applications",
        "image_processing": "Computer vision image processing techniques",
        "feature_extraction": "Image feature extraction and analysis",
        "cv_applications": "Computer vision applications and results",
        "time_series_components": "Time series decomposition and components",
        "seasonal_decomposition": "Seasonal pattern analysis and decomposition",
        "stationarity_analysis": "Time series stationarity testing",
        "time_series_forecasting": "Time series forecasting models and predictions",
        "forecast_residuals": "Forecasting model residual analysis",
        "big_data_processing": "Big data processing and analysis results",
        "advanced_machine_learning": "Advanced ML techniques and ensemble methods",
        "model_deployment_mlops": "Model deployment and MLOps pipeline",
        "real_world_case_studies": "Real-world data science case studies",
        "data_science_ethics": "Data science ethics and privacy protection",
        "communication_storytelling": "Data communication and storytelling techniques",
        "portfolio_development": "Data science portfolio development strategies",
        "career_development": "Career development and job search strategies",
        "advanced_career_specializations": "Advanced career specialization paths",
        "python_library_development": "Python library development and packaging",
    }

    base_name = image_file.replace(".png", "")
    return descriptions.get(
        base_name, f'Visualization of {base_name.replace("_", " ")}'
    )


def main():
    """Main function."""
    print("🚀 Starting summary file image updates...")
    print("=" * 50)

    update_summary_files()

    print("=" * 50)
    print("✅ Summary file image updates completed!")


if __name__ == "__main__":
    main()
