#!/usr/bin/env python3
"""
Script to add image displays to all markdown files that have generated visualizations.
This will make the documentation much more visual and engaging.
"""

import os
import re
import glob


def add_image_displays_to_markdown():
    """Add image displays to all markdown files that have generated visualizations."""

    # Define the chapters and their image files
    chapters_with_images = {
        "ch05": "data_collection_storage.png",
        "ch06": "data_cleaning_preprocessing.png",
        "ch07": "exploratory_data_analysis.png",
        "ch08": "statistical_inference_hypothesis_testing.png",
        "ch09": "machine_learning_fundamentals.png",
        "ch10": "feature_engineering_selection.png",
        "ch11": "unsupervised_learning.png",
        "ch12": "deep_learning_fundamentals.png",
        "ch13": "natural_language_processing.png",
        "ch14": "computer_vision_fundamentals.png",
        "ch15": "time_series_analysis.png",
        "ch16": "big_data_processing.png",
        "ch17": "advanced_machine_learning.png",
        "ch18": "model_deployment_mlops.png",
        "ch19": "real_world_case_studies.png",
        "ch20": "data_science_ethics.png",
        "ch21": "communication_storytelling.png",
        "ch22": "portfolio_development.png",
        "ch23": "career_development.png",
        "ch24": "advanced_career_specializations.png",
    }

    # Process each chapter
    for chapter, image_file in chapters_with_images.items():
        chapter_path = f"book/{chapter}"

        # Check if chapter directory exists
        if not os.path.exists(chapter_path):
            print(f"⚠️  Chapter directory {chapter_path} not found, skipping...")
            continue

        # Check if image file exists
        image_path = os.path.join(chapter_path, image_file)
        if not os.path.exists(image_path):
            print(f"⚠️  Image file {image_path} not found, skipping...")
            continue

        print(f"📁 Processing {chapter}...")

        # Update README.md
        readme_path = os.path.join(chapter_path, "README.md")
        if os.path.exists(readme_path):
            update_readme_with_images(readme_path, image_file, chapter)

        # Update CHAPTERX_SUMMARY.md
        summary_path = os.path.join(chapter_path, f"{chapter.upper()}_SUMMARY.md")
        if os.path.exists(summary_path):
            update_summary_with_images(summary_path, image_file, chapter)


def update_readme_with_images(readme_path, image_file, chapter):
    """Update README.md file to include image displays."""

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if images section already exists
    if "## Generated Visualizations" in content:
        print(f"    ✅ README.md already has images section")
        return

    # Find the Generated Outputs section
    outputs_pattern = r"(## Generated Outputs\n\n### Main Script\n.*?\n\n### Generated Outputs\n.*?\n)"

    if re.search(outputs_pattern, content, re.DOTALL):
        # Add Generated Visualizations section after Generated Outputs
        new_section = f"""## Generated Visualizations

### {get_chapter_title(chapter)} Dashboard
![{get_chapter_title(chapter)} Dashboard]({image_file})

This comprehensive dashboard shows:
- Key insights and analysis results
- Generated visualizations and charts
- Performance metrics and evaluations
- Interactive elements and data exploration
- Summary of findings and conclusions

"""

        # Insert after Generated Outputs section
        content = re.sub(
            r"(## Generated Outputs\n\n### Main Script\n.*?\n\n### Generated Outputs\n.*?\n)",
            r"\1" + new_section,
            content,
            flags=re.DOTALL,
        )

        # Write updated content
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"    ✅ README.md updated with image displays")
    else:
        print(f"    ⚠️  Could not find Generated Outputs section in README.md")


def update_summary_with_images(summary_path, image_file, chapter):
    """Update CHAPTERX_SUMMARY.md file to include image displays."""

    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if images section already exists
    if "## Generated Visualizations" in content:
        print(f"    ✅ {os.path.basename(summary_path)} already has images section")
        return

    # Find the Generated Outputs and Visualizations section
    outputs_pattern = r"(## Generated Outputs and Visualizations\n\n### 1\. .*?Dashboard\nThe script generates comprehensive visualizations showing:.*?\n\n### 2\. Console Output Examples)"

    if re.search(outputs_pattern, content, re.DOTALL):
        # Add image display after the description
        new_section = f"""### 1. {get_chapter_title(chapter)} Dashboard

![{get_chapter_title(chapter)} Dashboard]({image_file})

The script generates comprehensive visualizations showing:"""

        # Replace the existing section
        content = re.sub(
            r"(## Generated Outputs and Visualizations\n\n### 1\. .*?Dashboard\n)The script generates comprehensive visualizations showing:",
            new_section,
            content,
            flags=re.DOTALL,
        )

        # Write updated content
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"    ✅ {os.path.basename(summary_path)} updated with image displays")
    else:
        print(
            f"    ⚠️  Could not find Generated Outputs section in {os.path.basename(summary_path)}"
        )


def get_chapter_title(chapter):
    """Get a human-readable title for the chapter."""
    titles = {
        "ch05": "Data Collection and Storage",
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
    }
    return titles.get(chapter, chapter.upper())


def main():
    """Main function to run the markdown update process."""
    print("🚀 Starting markdown image display updates...")
    print("=" * 60)

    add_image_displays_to_markdown()

    print("=" * 60)
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
