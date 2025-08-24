"""
datascience_toolkit - A comprehensive toolkit for data science workflows

A Python library providing tools and utilities for data science,
machine learning, and statistical analysis workflows.

Features:
- Data processing and cleaning
- Statistical analysis and testing
- Machine learning utilities
- Data visualization tools
- Performance optimization
- Quality assurance tools

Author: Data Scientist
Email: author@example.com
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Data Scientist"
__email__ = "author@example.com"
__license__ = "MIT"
__url__ = "https://github.com/username/datascience_toolkit"

# Import main modules
try:
    from .core import DataProcessor, StatisticalAnalyzer
    from .utils import DataValidator, PerformanceOptimizer
    from .ml import MLPipeline, ModelEvaluator
    from .visualization import DataVisualizer, ChartBuilder
    
    __all__ = [
        "__version__",
        "__author__",
        "__email__",
        "__license__",
        "__url__",
        "DataProcessor",
        "StatisticalAnalyzer", 
        "DataValidator",
        "PerformanceOptimizer",
        "MLPipeline",
        "ModelEvaluator",
        "DataVisualizer",
        "ChartBuilder",
    ]
    
except ImportError as e:
    # Handle import errors gracefully for development
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")
    
    __all__ = [
        "__version__",
        "__author__",
        "__email__",
        "__license__",
        "__url__",
    ]

# Package metadata
__package_info__ = {
    "name": "datascience_toolkit",
    "version": __version__,
    "description": "A comprehensive toolkit for data science workflows",
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "url": __url__,
    "python_requires": ">=3.8",
    "keywords": [
        "data-science",
        "machine-learning",
        "statistics", 
        "data-analysis",
        "visualization",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib"
    ],
    "classifiers": [
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
        "Topic :: Software Development :: Libraries :: Application Frameworks"
    ]
}

def get_package_info():
    """Get comprehensive package information."""
    return __package_info__.copy()

def get_version():
    """Get package version."""
    return __version__

def get_author():
    """Get package author information."""
    return {
        "name": __author__,
        "email": __email__
    }

def get_license():
    """Get package license information."""
    return __license__

def get_url():
    """Get package URL."""
    return __url__

# Version compatibility check
def check_compatibility():
    """Check if the current Python version is compatible."""
    import sys
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        raise RuntimeError(
            f"datascience_toolkit requires Python {min_version[0]}.{min_version[1]} or higher. "
            f"Current version: {current_version[0]}.{current_version[1]}"
        )
    return True

# Run compatibility check on import
check_compatibility()

# Package initialization message (only in development)
if __version__.endswith('dev') or __version__.endswith('alpha'):
    import warnings
    warnings.warn(
        f"datascience_toolkit {__version__} is a development version. "
        "For production use, please install a stable release.",
        UserWarning,
        stacklevel=2
    )
