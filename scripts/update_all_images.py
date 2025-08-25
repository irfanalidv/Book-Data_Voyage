#!/usr/bin/env python3
"""
Launcher script to update all markdown files with image displays.
This script runs both the README and summary file updates from the scripts directory.
"""

import os
import sys
import subprocess


def main():
    """Main function to run all image update scripts."""
    print("🚀 Data Voyage - Image Display Update Launcher")
    print("=" * 60)

    # Check if we're in the scripts directory
    if not os.path.exists("update_markdown_images_v2.py"):
        print("❌ This script must be run from the scripts directory!")
        print("Please navigate to the scripts folder and run this script.")
        return

    print("📁 Running from scripts directory")

    # Run README updates
    print("\n📝 Step 1: Updating README.md files with image displays...")
    try:
        result = subprocess.run(
            [sys.executable, "update_markdown_images_v2.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running README updates: {e}")
        print(f"Error output: {e.stderr}")
        return

    # Run summary file updates
    print("\n📝 Step 2: Updating CHAPTERX_SUMMARY.md files with image displays...")
    try:
        result = subprocess.run(
            [sys.executable, "update_summary_images.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running summary updates: {e}")
        print(f"Error output: {e.stderr}")
        return

    print("\n" + "=" * 60)
    print("✅ All image display updates completed successfully!")
    print("\n📝 Summary of what was updated:")
    print("- README.md files in all chapters")
    print("- CHAPTERX_SUMMARY.md files in all chapters")
    print("- All visualizations now appear directly in the documentation")

    print("\n🎯 Next steps:")
    print("1. Review the updated markdown files")
    print("2. Ensure all images are displaying correctly")
    print("3. Test the documentation in your markdown viewer")
    print("4. Commit changes to version control")

    print("\n📁 All scripts are now located in this 'scripts/' folder")
    print("📚 Documentation for the scripts is in 'README.md'")


if __name__ == "__main__":
    main()
