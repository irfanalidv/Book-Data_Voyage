#!/usr/bin/env python3
"""
Setup script for datascience_toolkit

A comprehensive toolkit for data science workflows including data processing,
statistical analysis, machine learning, and visualization tools.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    """Read README.md file for long description."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    """Read requirements.txt file for dependencies."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read development requirements
def read_dev_requirements():
    """Read requirements-dev.txt file for development dependencies."""
    with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="datascience_toolkit",
    version="0.1.0",
    author="Data Scientist",
    author_email="author@example.com",
    description="A comprehensive toolkit for data science workflows",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/datascience_toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/username/datascience_toolkit/issues",
        "Documentation": "https://datascience_toolkit.readthedocs.io",
        "Source Code": "https://github.com/username/datascience_toolkit",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": read_dev_requirements(),
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
        "all": read_dev_requirements(),
    },
    include_package_data=True,
    package_data={
        "datascience_toolkit": [
            "*.txt",
            "*.md",
            "*.yml",
            "*.yaml",
            "data/*",
            "config/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "datascience-toolkit=datascience_toolkit.cli:main",
        ],
    },
    zip_safe=False,
    keywords=[
        "data-science",
        "machine-learning",
        "statistics",
        "data-analysis",
        "visualization",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
    ],
    platforms=["any"],
    license="MIT",
    download_url="https://github.com/username/datascience_toolkit/archive/v0.1.0.tar.gz",
)
