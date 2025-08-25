#!/usr/bin/env python3
"""
Launcher script to update all markdown files with image displays.
This script runs both the README and summary file updates from the root directory.
"""

import os
import sys
import subprocess


def main():
    """Main function to run all image update scripts."""
    print("🚀 Data Voyage - Image Display Update Launcher")
    print("=" * 60)

    # Check if scripts directory exists
    scripts_dir = "scripts"
    if not os.path.exists(scripts_dir):
        print(f"❌ Scripts directory '{scripts_dir}' not found!")
        print("Please ensure the scripts folder exists with the update scripts.")
        return

    # Store current directory
    current_dir = os.getcwd()

    # Change to scripts directory
    os.chdir(scripts_dir)
    print(f"📁 Changed to {scripts_dir} directory")

    # Run README updates
    print("\n📝 Step 1: Updating README.md files with image displays...")
    try:
        # Set environment variable to indicate we're running from scripts directory
        env = os.environ.copy()
        env["RUNNING_FROM_SCRIPTS"] = "1"

        result = subprocess.run(
            [sys.executable, "update_markdown_images_v2.py"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
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
            env=env,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running summary updates: {e}")
        print(f"Error output: {e.stderr}")
        return

    # Return to original directory
    os.chdir(current_dir)

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

    print("\n📁 Scripts are located in the 'scripts/' folder")
    print("📚 Documentation for the scripts is in 'scripts/README.md'")


if __name__ == "__main__":
    main()
