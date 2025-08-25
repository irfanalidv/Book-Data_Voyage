#!/usr/bin/env python3
"""
Improved script to add image displays to all markdown files that have generated visualizations.
This version works with the actual file structure and PNG files found in the chapters.
"""

import os
import re
import glob


def add_image_displays_to_markdown():
    """Add image displays to all markdown files that have generated visualizations."""

    # Define the chapters and their actual image files based on what exists
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

    # Determine the correct path to the book directory
    if os.path.exists("../book"):
        book_path = "../book"
    elif os.path.exists("book"):
        book_path = "book"
    else:
        print("❌ Could not find 'book' directory!")
        print("Please run this script from the root directory or scripts directory.")
        return

    # Process each chapter
    for chapter, image_files in chapters_with_images.items():
        chapter_path = os.path.join(book_path, chapter)

        # Check if chapter directory exists
        if not os.path.exists(chapter_path):
            print(f"⚠️  Chapter directory {chapter_path} not found, skipping...")
            continue

        print(f"📁 Processing {chapter}...")

        # Update README.md
        readme_path = os.path.join(chapter_path, "README.md")
        if os.path.exists(readme_path):
            update_readme_with_images(readme_path, image_files, chapter)

        # Update CHAPTERX_SUMMARY.md
        summary_path = os.path.join(chapter_path, f"{chapter.upper()}_SUMMARY.md")
        if os.path.exists(summary_path):
            update_summary_with_images(summary_path, image_files, chapter)


def update_readme_with_images(readme_path, image_files, chapter):
    """Update README.md file to include image displays."""

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if images section already exists
    if "## Generated Visualizations" in content:
        print(f"    ✅ README.md already has images section")
        return

    # Find the Generated Outputs section or create one
    if "## Generated Outputs" in content:
        # Add Generated Visualizations section after Generated Outputs
        new_section = create_visualizations_section(image_files, chapter)

        # Insert after Generated Outputs section
        content = re.sub(
            r"(## Generated Outputs\n.*?\n)",
            r"\1" + new_section,
            content,
            flags=re.DOTALL,
        )
    else:
        # Create a new Generated Outputs section with visualizations
        new_section = f"""## Generated Outputs

### Main Script
- `{chapter}_*.py` - Complete chapter implementation

### Generated Visualizations

{create_visualizations_section(image_files, chapter)}
"""

        # Add before the "Running the Code" section or at the end
        if "## Running the Code" in content:
            content = re.sub(r"(## Running the Code)", new_section + "\n\\1", content)
        else:
            content += "\n" + new_section

    # Write updated content
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"    ✅ README.md updated with image displays")


def update_summary_with_images(summary_path, image_files, chapter):
    """Update CHAPTERX_SUMMARY.md file to include image displays."""

    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if images section already exists
    if "## Generated Visualizations" in content:
        print(f"    ✅ {os.path.basename(summary_path)} already has images section")
        return

    # Find the Generated Outputs section or add one
    if "## Generated Outputs and Visualizations" in content:
        # Add image displays to existing section
        new_section = create_visualizations_section(
            image_files, chapter, is_summary=True
        )

        # Replace the existing section
        content = re.sub(
            r"(## Generated Outputs and Visualizations\n\n)(.*?)(\n\n## Key Concepts)",
            r"\1" + new_section + r"\3",
            content,
            flags=re.DOTALL,
        )
    else:
        # Add new section before Key Concepts
        new_section = f"""## Generated Outputs and Visualizations

{create_visualizations_section(image_files, chapter, is_summary=True)}

"""

        if "## Key Concepts" in content:
            content = re.sub(r"(## Key Concepts)", new_section + "\\1", content)
        else:
            content += "\n" + new_section

    # Write updated content
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"    ✅ {os.path.basename(summary_path)} updated with image displays")


def create_visualizations_section(image_files, chapter, is_summary=False):
    """Create the visualizations section content."""

    if len(image_files) == 1:
        # Single image
        image_file = image_files[0]
        title = get_chapter_title(chapter)
        return f"""### {title} Dashboard

![{title} Dashboard]({image_file})

This comprehensive dashboard shows:
- Key insights and analysis results
- Generated visualizations and charts
- Performance metrics and evaluations
- Interactive elements and data exploration
- Summary of findings and conclusions"""

    else:
        # Multiple images
        content = f"### {get_chapter_title(chapter)} Visualizations\n\n"
        content += "This chapter generates multiple visualizations showing:\n\n"

        for i, image_file in enumerate(image_files, 1):
            image_title = get_image_title(image_file)
            content += f"#### {i}. {image_title}\n\n"
            content += f"![{image_title}]({image_file})\n\n"
            content += f"- {get_image_description(image_file)}\n\n"

        return content


def get_image_title(image_file):
    """Get a human-readable title for the image file."""
    # Remove .png extension and convert underscores to spaces
    title = image_file.replace(".png", "").replace("_", " ").title()

    # Special cases
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


def get_chapter_title(chapter):
    """Get a human-readable title for the chapter."""
    titles = {
        "ch01": "Data Science Fundamentals",
        "ch03": "Mathematics and Statistics",
        "ch06": "Data Cleaning and Preprocessing",
        "ch07": "Exploratory Data Analysis",
        "ch08": "Statistical Inference and Hypothesis Testing",
        "ch09": "Machine Learning Fundamentals",
        "ch10": "Feature Engineering and Selection",
        "ch11": "Unsupervised Learning",
        "ch12": "Deep Learning Fundamentals",
        "ch13": "Natural Language Processing",
        "ch14": "Computer Vision Fundamentals",
        "ch15": "Time Series Analysis",
        "ch16": "Big Data Processing",
        "ch17": "Advanced Machine Learning",
        "ch18": "Model Deployment and MLOps",
        "ch19": "Real-World Case Studies",
        "ch20": "Data Science Ethics",
        "ch21": "Communication and Storytelling",
        "ch22": "Portfolio Development",
        "ch23": "Career Development",
        "ch24": "Advanced Career Specializations",
        "ch25": "Python Library Development",
    }
    return titles.get(chapter, chapter.upper())


def main():
    """Main function to run the markdown update process."""
    print("🚀 Starting improved markdown image display updates...")
    print("=" * 70)

    add_image_displays_to_markdown()

    print("=" * 70)
    print("✅ Markdown image display updates completed!")
    print("\n📝 Summary of changes:")
    print("- Added image displays to README.md files")
    print("- Added image displays to CHAPTERX_SUMMARY.md files")
    print("- All visualizations now appear directly in the documentation")
    print("\n🎯 Next steps:")
    print("- Review the updated markdown files")
    print("- Ensure all images are displaying correctly")
    print("- Test the documentation in your markdown viewer")


if __name__ == "__main__":
    main()
