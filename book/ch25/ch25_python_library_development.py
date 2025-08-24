#!/usr/bin/env python3
"""
Chapter 25: Building and Publishing Python Libraries to PyPI
============================================================

This bonus chapter covers the complete process of building and publishing
Python libraries to PyPI. You'll learn library design, packaging,
testing, documentation, and distribution to contribute to the Python ecosystem.
"""

import os
import sys
import time
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Set up for demonstration
warnings.filterwarnings("ignore")


class LibraryDesigner:
    """Demonstrate library design and architecture principles."""

    def __init__(self):
        self.library_structure = {}

    def design_library_structure(self):
        """Design a professional Python library structure."""
        print("1. LIBRARY DESIGN AND ARCHITECTURE")
        print("=" * 60)

        print("\n1.1 DESIGNING LIBRARY STRUCTURE:")
        print("-" * 45)

        # Define library structure
        library_structure = {
            "package_name": "datascience_toolkit",
            "version": "0.1.0",
            "description": "A comprehensive toolkit for data science workflows",
            "author": "Data Scientist",
            "email": "author@example.com",
            "license": "MIT",
            "python_requires": ">=3.8",
            "classifiers": [
                "Development Status :: 3 - Alpha",
                "Intended Audience :: Developers",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11",
                "Topic :: Scientific/Engineering :: Information Analysis",
                "Topic :: Software Development :: Libraries :: Python Modules",
            ],
            "dependencies": [
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "matplotlib>=3.4.0",
                "scikit-learn>=1.0.0",
                "scipy>=1.7.0",
            ],
            "dev_dependencies": [
                "pytest>=6.0.0",
                "pytest-cov>=2.12.0",
                "black>=21.0.0",
                "flake8>=3.9.0",
                "mypy>=0.910",
                "sphinx>=4.0.0",
            ],
        }

        self.library_structure = library_structure

        print("  ✅ Library structure designed:")
        print(f"    Package Name: {library_structure['package_name']}")
        print(f"    Version: {library_structure['version']}")
        print(f"    Description: {library_structure['description']}")
        print(f"    Python Version: {library_structure['python_requires']}")
        print(f"    Dependencies: {len(library_structure['dependencies'])} packages")
        print(
            f"    Dev Dependencies: {len(library_structure['dependencies'])} packages"
        )

        return self.library_structure

    def create_directory_structure(self):
        """Create the standard Python package directory structure."""
        print("\n1.2 CREATING DIRECTORY STRUCTURE:")
        print("-" * 45)

        # Define directory structure
        directories = [
            "datascience_toolkit/",
            "datascience_toolkit/core/",
            "datascience_toolkit/utils/",
            "datascience_toolkit/visualization/",
            "datascience_toolkit/ml/",
            "tests/",
            "tests/unit/",
            "tests/integration/",
            "docs/",
            "docs/source/",
            "examples/",
            "scripts/",
        ]

        print("  📁 Standard Python package directories:")
        for directory in directories:
            print(f"    {directory}")

        # Create files structure
        files = [
            "setup.py",
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "MANIFEST.in",
            "requirements.txt",
            "requirements-dev.txt",
            "tox.ini",
            ".gitignore",
            "CHANGELOG.md",
            "datascience_toolkit/__init__.py",
            "datascience_toolkit/core/__init__.py",
            "datascience_toolkit/utils/__init__.py",
            "datascience_toolkit/visualization/__init__.py",
            "datascience_toolkit/ml/__init__.py",
        ]

        print("\n  📄 Essential package files:")
        for file in files:
            print(f"    {file}")

        return directories, files


class LibraryBuilder:
    """Demonstrate the actual building and publishing process."""

    def __init__(self):
        self.package_name = "datascience_toolkit"
        self.version = "0.1.0"

    def create_setup_py(self):
        """Create a complete setup.py file."""
        print("\n2. CREATING PACKAGE CONFIGURATION FILES")
        print("=" * 60)

        print("\n2.1 SETUP.PY CONFIGURATION:")
        print("-" * 45)

        setup_py_content = f'''#!/usr/bin/env python3
"""
Setup script for {self.package_name}
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="{self.package_name}",
    version="{self.version}",
    author="Data Scientist",
    author_email="author@example.com",
    description="A comprehensive toolkit for data science workflows",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/{self.package_name}",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={{
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0"
        ]
    }},
    include_package_data=True,
    zip_safe=False,
)
'''

        print("  ✅ setup.py created with:")
        print("    • Package metadata and classifiers")
        print("    • Dependency management")
        print("    • Development dependencies")
        print("    • Package discovery")

        return setup_py_content

    def create_pyproject_toml(self):
        """Create a modern pyproject.toml file."""
        print("\n2.2 PYPROJECT.TOML CONFIGURATION:")
        print("-" * 45)

        pyproject_content = f"""[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "{self.package_name}"
version = "{self.version}"
description = "A comprehensive toolkit for data science workflows"
readme = "README.md"
license = {{text = "MIT"}}
authors = [
    {{name = "Data Scientist", email = "author@example.com"}}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries :: Python Modules"
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "pytest-cov>=2.12.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "sphinx>=4.0.0"
]

[project.urls]
Homepage = "https://github.com/username/{self.package_name}"
Repository = "https://github.com/username/{self.package_name}.git"
Documentation = "https://{self.package_name}.readthedocs.io"
"Bug Tracker" = "https://github.com/username/{self.package_name}/issues"

[tool.setuptools]
packages = ["{self.package_name}"]

[tool.setuptools.package-data]
"{self.package_name}" = ["*.txt", "*.md", "*.yml", "*.yaml"]

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov={self.package_name} --cov-report=html --cov-report=term-missing"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
"""

        print("  ✅ pyproject.toml created with:")
        print("    • Modern build system configuration")
        print("    • Project metadata")
        print("    • Tool configurations (black, pytest, mypy)")
        print("    • Package data inclusion")

        return pyproject_content

    def create_manifest_in(self):
        """Create MANIFEST.in for including non-Python files."""
        print("\n2.3 MANIFEST.IN CONFIGURATION:")
        print("-" * 45)

        manifest_content = """# Include README and license
include README.md
include LICENSE
include CHANGELOG.md
include requirements.txt
include requirements-dev.txt

# Include documentation
recursive-include docs *.md *.rst *.txt *.html *.css *.js

# Include test files
recursive-include tests *.py *.txt *.csv *.json

# Include examples
recursive-include examples *.py *.ipynb *.md *.txt

# Include configuration files
include *.toml *.ini *.cfg *.yml *.yaml

# Exclude common unwanted files
global-exclude *.pyc
global-exclude *.pyo
global-exclude *.pyd
global-exclude .git*
global-exclude .DS_Store
global-exclude __pycache__
global-exclude *.egg-info
global-exclude .pytest_cache
global-exclude .coverage
global-exclude htmlcov
"""

        print("  ✅ MANIFEST.in created with:")
        print("    • File inclusion patterns")
        print("    • Documentation inclusion")
        print("    • Test file inclusion")
        print("    • Unwanted file exclusions")

        return manifest_content

    def create_package_init_files(self):
        """Create __init__.py files for the package."""
        print("\n2.4 PACKAGE INITIALIZATION FILES:")
        print("-" * 45)

        # Main package __init__.py
        main_init = f'''"""
{self.package_name} - A comprehensive toolkit for data science workflows

A Python library providing tools and utilities for data science,
machine learning, and statistical analysis workflows.
"""

__version__ = "{self.version}"
__author__ = "Data Scientist"
__email__ = "author@example.com"

# Import main modules
from .core import *
from .utils import *
from .visualization import *
from .ml import *

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
'''

        # Core module __init__.py
        core_init = '''"""
Core functionality for data science operations.
"""

from .data_processor import DataProcessor
from .statistical_analyzer import StatisticalAnalyzer

__all__ = [
    "DataProcessor",
    "StatisticalAnalyzer",
]
'''

        print("  ✅ Package __init__.py files created:")
        print("    • Main package initialization")
        print("    • Module imports and exports")
        print("    • Version and author information")

        return main_init, core_init

    def demonstrate_build_process(self):
        """Demonstrate the complete build and publish process."""
        print("\n3. BUILDING AND PUBLISHING PROCESS")
        print("=" * 60)

        print("\n3.1 PREPARATION STEPS:")
        print("-" * 45)

        preparation_steps = [
            "1. Install build tools: pip install build twine",
            "2. Ensure all dependencies are in requirements.txt",
            "3. Update version in __init__.py and setup.py",
            "4. Run tests: python -m pytest",
            "5. Check code quality: black . && flake8",
            "6. Update CHANGELOG.md with new version",
            "7. Commit and tag the release: git tag v0.1.0",
        ]

        for step in preparation_steps:
            print(f"    {step}")

        print("\n3.2 BUILDING THE PACKAGE:")
        print("-" * 45)

        build_steps = [
            "1. Clean previous builds: rm -rf dist/ build/ *.egg-info/",
            "2. Build source distribution: python -m build --sdist",
            "3. Build wheel distribution: python -m build --wheel",
            "4. Verify package contents: tar -tzf dist/*.tar.gz",
            "5. Check wheel contents: unzip -l dist/*.whl",
        ]

        for step in build_steps:
            print(f"    {step}")

        print("\n3.3 TESTING THE PACKAGE:")
        print("-" * 45)

        test_steps = [
            "1. Install in test environment: pip install dist/*.whl",
            "2. Run import test: python -c 'import datascience_toolkit'",
            "3. Run basic functionality tests",
            "4. Uninstall test installation: pip uninstall datascience_toolkit",
        ]

        for step in test_steps:
            print(f"    {step}")

        print("\n3.4 PUBLISHING TO PYPI:")
        print("-" * 45)

        publish_steps = [
            "1. Test upload to TestPyPI: twine upload --repository testpypi dist/*",
            "2. Verify on TestPyPI: https://test.pypi.org/project/datascience_toolkit/",
            "3. Install from TestPyPI: pip install -i https://test.pypi.org/simple/ datascience_toolkit",
            "4. Upload to PyPI: twine upload dist/*",
            "5. Verify on PyPI: https://pypi.org/project/datascience_toolkit/",
        ]

        for step in publish_steps:
            print(f"    {step}")

        print("\n3.5 AUTOMATION WITH GITHUB ACTIONS:")
        print("-" * 45)

        automation_steps = [
            "1. Create .github/workflows/publish.yml",
            "2. Configure PyPI API tokens in repository secrets",
            "3. Set up automatic testing on pull requests",
            "4. Configure automatic publishing on tags",
            "5. Set up documentation building and deployment",
        ]

        for step in automation_steps:
            print(f"    {step}")

        return (
            preparation_steps,
            build_steps,
            test_steps,
            publish_steps,
            automation_steps,
        )

    def create_github_workflow(self):
        """Create a GitHub Actions workflow for CI/CD."""
        print("\n3.6 GITHUB ACTIONS WORKFLOW:")
        print("-" * 45)

        workflow_content = """name: Build and Publish Python Package

on:
  push:
    tags:
      - 'v*'
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Format with black
      run: |
        black --check .
    
    - name: Type check with mypy
      run: |
        mypy datascience_toolkit/
    
    - name: Test with pytest
      run: |
        pytest --cov=datascience_toolkit --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  publish:
    needs: test
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
"""

        print("  ✅ GitHub Actions workflow created with:")
        print("    • Multi-Python version testing")
        print("    • Code quality checks (linting, formatting, type checking)")
        print("    • Test coverage reporting")
        print("    • Automatic PyPI publishing on tags")

        return workflow_content


def create_library_development_visualizations():
    """Create visualizations for Python library development process."""
    print("\n4. CREATING LIBRARY DEVELOPMENT VISUALIZATIONS:")
    print("-" * 45)

    print("  Generating visualizations...")

    # Create figure with subplots
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Python Library Development: Complete Package Creation and Distribution Pipeline",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Package Structure Overview
    ax1 = axes[0, 0]
    structure_elements = ["Core", "Utils", "ML", "Viz", "Tests", "Docs"]
    complexity_scores = [0.9, 0.7, 0.8, 0.6, 0.5, 0.4]  # Simulated scores

    bars = ax1.bar(
        structure_elements,
        complexity_scores,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax1.set_title("Package Structure Complexity", fontweight="bold")
    ax1.set_ylabel("Complexity Score")
    ax1.set_ylim(0, 1)

    # Add value labels
    for bar, score in zip(bars, complexity_scores):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Development Time Distribution
    ax2 = axes[0, 1]
    development_phases = ["Design", "Coding", "Testing", "Docs", "Build", "Publish"]
    time_percentages = [15, 40, 25, 10, 5, 5]  # Simulated percentages

    wedges, texts, autotexts = ax2.pie(
        time_percentages,
        labels=development_phases,
        autopct="%1.0f%%",
        colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
    )
    ax2.set_title("Development Time Distribution", fontweight="bold")

    # 3. Quality Metrics
    ax3 = axes[1, 0]
    quality_metrics = [
        "Code Coverage",
        "Linting Score",
        "Type Coverage",
        "Documentation",
        "Test Quality",
    ]
    metric_scores = [92, 95, 88, 85, 90]  # Simulated scores

    bars = ax3.barh(
        quality_metrics,
        metric_scores,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"],
        alpha=0.8,
    )
    ax3.set_title("Quality Metrics", fontweight="bold")
    ax3.set_xlabel("Score (%)")
    ax3.set_xlim(0, 100)

    # Add value labels
    for bar, score in zip(bars, metric_scores):
        width = bar.get_width()
        ax3.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score}%",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # 4. Build Process Flow
    ax4 = axes[1, 1]
    build_steps = ["Clean", "Validate", "Build", "Test", "Package", "Upload"]
    success_rates = [100, 95, 90, 85, 95, 90]  # Simulated rates

    bars = ax4.bar(
        build_steps,
        success_rates,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A", "#98D8C8"],
        alpha=0.8,
    )
    ax4.set_title("Build Process Success Rates", fontweight="bold")
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_ylim(0, 100)
    ax4.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{rate}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save visualization
    output_file = "python_library_development.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✅ Visualization saved: {output_file}")

    plt.show()


def main():
    """Main function to run Python library development demonstrations."""
    try:
        print("=" * 80)
        print("CHAPTER 25: BUILDING AND PUBLISHING PYTHON LIBRARIES TO PYPI")
        print("=" * 80)

        # Initialize library development demonstrations
        designer = LibraryDesigner()
        builder = LibraryBuilder()

        # Run library design demonstrations
        print("\n" + "=" * 80)
        library_structure = designer.design_library_structure()
        directories, files = designer.create_directory_structure()

        # Run library building demonstrations
        print("\n" + "=" * 80)
        setup_py = builder.create_setup_py()
        pyproject_toml = builder.create_pyproject_toml()
        manifest_in = builder.create_manifest_in()
        main_init, core_init = builder.create_package_init_files()

        # Demonstrate build and publish process
        preparation, build_steps, test_steps, publish_steps, automation_steps = (
            builder.demonstrate_build_process()
        )
        github_workflow = builder.create_github_workflow()

        # Create visualizations
        create_library_development_visualizations()

        print("\n" + "=" * 80)
        print("CHAPTER 25 - PYTHON LIBRARY DEVELOPMENT COMPLETED!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Complete Python library design and architecture")
        print("  • Professional package structure and organization")
        print("  • Configuration files (setup.py, pyproject.toml, MANIFEST.in)")
        print("  • Step-by-step build and publish process")
        print("  • GitHub Actions CI/CD automation")
        print("  • Library development visualization and analysis")

        print("\n📊 Generated Visualizations:")
        print(
            "  • python_library_development.png - Complete library development pipeline dashboard"
        )

        print("\n🚀 Next Steps:")
        print("  • Design and build your own Python library")
        print("  • Implement comprehensive testing and documentation")
        print("  • Package and publish to PyPI")
        print("  • Set up automated CI/CD pipelines")
        print(
            "  • 🎉 CONGRATULATIONS! You've completed the comprehensive Data Science Book + Python Library Development!"
        )
        print(
            "  • You now have complete mastery of data science AND professional Python development!"
        )

    except Exception as e:
        print(f"\n❌ Error in Chapter 25: {str(e)}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
