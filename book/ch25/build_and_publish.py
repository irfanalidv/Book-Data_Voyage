#!/usr/bin/env python3
"""
Build and Publish Script for datascience_toolkit

This script demonstrates the complete process of building and publishing
a Python package to PyPI. It includes all the steps from preparation
to final publication.
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

class PackageBuilder:
    """Handles the complete package building and publishing process."""
    
    def __init__(self, package_name: str = "datascience_toolkit"):
        self.package_name = package_name
        self.current_dir = Path.cwd()
        self.dist_dir = self.current_dir / "dist"
        self.build_dir = self.current_dir / "build"
        
    def check_prerequisites(self) -> bool:
        """Check if all required tools are installed."""
        print("🔍 Checking prerequisites...")
        
        required_tools = ["python", "pip", "build", "twine"]
        missing_tools = []
        
        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"], 
                             capture_output=True, check=True)
                print(f"  ✅ {tool} is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
                print(f"  ❌ {tool} is missing")
        
        if missing_tools:
            print(f"\n❌ Missing tools: {', '.join(missing_tools)}")
            print("Install missing tools with:")
            print("  pip install build twine")
            return False
        
        print("✅ All prerequisites are satisfied!")
        return True
    
    def clean_previous_builds(self) -> None:
        """Remove previous build artifacts."""
        print("\n🧹 Cleaning previous builds...")
        
        dirs_to_clean = [self.dist_dir, self.build_dir]
        files_to_clean = list(self.current_dir.glob("*.egg-info"))
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  🗑️  Removed {dir_path}")
        
        for file_path in files_to_clean:
            if file_path.exists():
                shutil.rmtree(file_path)
                print(f"  🗑️  Removed {file_path}")
        
        print("✅ Build artifacts cleaned!")
    
    def run_tests(self) -> bool:
        """Run the test suite to ensure quality."""
        print("\n🧪 Running tests...")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                "python", "-m", "pytest", 
                "--cov", self.package_name,
                "--cov-report", "term-missing",
                "-v"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed!")
                return True
            else:
                print("❌ Tests failed!")
                print("Test output:")
                print(result.stdout)
                print("Test errors:")
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("⚠️  pytest not found, skipping tests")
            return True
    
    def check_code_quality(self) -> bool:
        """Run code quality checks."""
        print("\n🔍 Running code quality checks...")
        
        quality_checks = [
            ("black", ["black", "--check", "."], "Code formatting"),
            ("flake8", ["flake8", "."], "Linting"),
            ("mypy", ["mypy", self.package_name], "Type checking")
        ]
        
        all_passed = True
        
        for tool_name, command, description in quality_checks:
            try:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✅ {description} passed")
                else:
                    print(f"  ❌ {description} failed")
                    print(f"    {result.stderr}")
                    all_passed = False
            except FileNotFoundError:
                print(f"  ⚠️  {tool_name} not found, skipping {description}")
        
        return all_passed
    
    def build_package(self) -> bool:
        """Build the package distributions."""
        print("\n🔨 Building package...")
        
        try:
            # Build source distribution
            print("  📦 Building source distribution...")
            result = subprocess.run([
                "python", "-m", "build", "--sdist"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Source distribution build failed: {result.stderr}")
                return False
            
            # Build wheel distribution
            print("  🎡 Building wheel distribution...")
            result = subprocess.run([
                "python", "-m", "build", "--wheel"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Wheel distribution build failed: {result.stderr}")
                return False
            
            print("✅ Package built successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Build failed with error: {e}")
            return False
    
    def verify_package(self) -> bool:
        """Verify the built package."""
        print("\n🔍 Verifying package...")
        
        if not self.dist_dir.exists():
            print("❌ Distribution directory not found")
            return False
        
        dist_files = list(self.dist_dir.glob("*"))
        if not dist_files:
            print("❌ No distribution files found")
            return False
        
        print(f"  📁 Found {len(dist_files)} distribution files:")
        for file_path in dist_files:
            print(f"    • {file_path.name}")
        
        # Check package with twine
        try:
            result = subprocess.run([
                "twine", "check", "dist/*"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Package verification passed!")
                return True
            else:
                print("❌ Package verification failed:")
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("⚠️  twine not found, skipping verification")
            return True
    
    def test_installation(self) -> bool:
        """Test package installation."""
        print("\n🧪 Testing package installation...")
        
        try:
            # Find wheel file
            wheel_files = list(self.dist_dir.glob("*.whl"))
            if not wheel_files:
                print("❌ No wheel files found for testing")
                return False
            
            wheel_file = wheel_files[0]
            print(f"  📦 Testing installation of {wheel_file.name}")
            
            # Install package
            result = subprocess.run([
                "pip", "install", str(wheel_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Installation failed: {result.stderr}")
                return False
            
            # Test import
            result = subprocess.run([
                "python", "-c", f"import {self.package_name}; print('Import successful')"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Import test failed: {result.stderr}")
                return False
            
            print("✅ Package installation test passed!")
            
            # Uninstall test package
            subprocess.run([
                "pip", "uninstall", "-y", self.package_name
            ], capture_output=True)
            
            return True
            
        except Exception as e:
            print(f"❌ Installation test failed: {e}")
            return False
    
    def publish_to_testpypi(self) -> bool:
        """Publish package to TestPyPI."""
        print("\n🚀 Publishing to TestPyPI...")
        
        try:
            result = subprocess.run([
                "twine", "upload", "--repository", "testpypi", "dist/*"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Package published to TestPyPI successfully!")
                print("  🔗 TestPyPI URL: https://test.pypi.org/project/datascience_toolkit/")
                return True
            else:
                print("❌ TestPyPI publication failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ TestPyPI publication failed: {e}")
            return False
    
    def publish_to_pypi(self) -> bool:
        """Publish package to PyPI."""
        print("\n🚀 Publishing to PyPI...")
        
        try:
            result = subprocess.run([
                "twine", "upload", "dist/*"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Package published to PyPI successfully!")
                print("  🔗 PyPI URL: https://pypi.org/project/datascience_toolkit/")
                return True
            else:
                print("❌ PyPI publication failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ PyPI publication failed: {e}")
            return False
    
    def show_build_summary(self) -> None:
        """Show a summary of the build process."""
        print("\n" + "="*60)
        print("📊 BUILD SUMMARY")
        print("="*60)
        
        if self.dist_dir.exists():
            dist_files = list(self.dist_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in dist_files if f.is_file())
            
            print(f"📦 Distribution files created: {len(dist_files)}")
            print(f"💾 Total size: {total_size / 1024:.1f} KB")
            
            for file_path in dist_files:
                file_size = file_path.stat().st_size
                print(f"  • {file_path.name} ({file_size / 1024:.1f} KB)")
        
        print("\n🎯 Next steps:")
        print("  1. Verify package on PyPI")
        print("  2. Test installation: pip install datascience_toolkit")
        print("  3. Monitor for issues and feedback")
        print("  4. Plan next release")
    
    def run_complete_build(self, skip_tests: bool = False, 
                          skip_quality: bool = False,
                          test_pypi_only: bool = False) -> bool:
        """Run the complete build and publish process."""
        print("🚀 Starting complete package build and publish process...")
        print("="*60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Clean previous builds
        self.clean_previous_builds()
        
        # Run tests (unless skipped)
        if not skip_tests:
            if not self.run_tests():
                print("❌ Build process stopped due to test failures")
                return False
        
        # Run quality checks (unless skipped)
        if not skip_quality:
            if not self.check_code_quality():
                print("❌ Build process stopped due to quality check failures")
                return False
        
        # Build package
        if not self.build_package():
            return False
        
        # Verify package
        if not self.verify_package():
            return False
        
        # Test installation
        if not self.test_installation():
            return False
        
        # Publish to TestPyPI
        if not self.publish_to_testpypi():
            return False
        
        # Publish to PyPI (unless TestPyPI only)
        if not test_pypi_only:
            if not self.publish_to_pypi():
                return False
        
        # Show summary
        self.show_build_summary()
        
        print("\n🎉 Package build and publish process completed successfully!")
        return True


def main():
    """Main function to run the build and publish process."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build and publish datascience_toolkit package"
    )
    parser.add_argument(
        "--skip-tests", 
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--skip-quality", 
        action="store_true",
        help="Skip code quality checks"
    )
    parser.add_argument(
        "--test-pypi-only", 
        action="store_true",
        help="Only publish to TestPyPI, not PyPI"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual changes will be made")
        print("="*60)
        print("This would run the following steps:")
        print("1. Check prerequisites")
        print("2. Clean previous builds")
        if not args.skip_tests:
            print("3. Run tests")
        if not args.skip_quality:
            print("4. Run code quality checks")
        print("5. Build package")
        print("6. Verify package")
        print("7. Test installation")
        print("8. Publish to TestPyPI")
        if not args.test_pypi_only:
            print("9. Publish to PyPI")
        print("10. Show build summary")
        return
    
    # Initialize builder
    builder = PackageBuilder()
    
    # Run complete build process
    success = builder.run_complete_build(
        skip_tests=args.skip_tests,
        skip_quality=args.skip_quality,
        test_pypi_only=args.test_pypi_only
    )
    
    if success:
        print("\n🎉 SUCCESS: Package built and published successfully!")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Package build and publish process failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
