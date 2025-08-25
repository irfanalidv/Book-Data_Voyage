#!/usr/bin/env python3
"""
Simple launcher script for Data Voyage image updates.
This script directs users to the scripts folder where all utility scripts are located.
"""

import os
import sys


def main():
    """Main function to guide users to the scripts folder."""
    print("🚀 Data Voyage - Image Display Updates")
    print("=" * 50)

    scripts_dir = "scripts"

    if not os.path.exists(scripts_dir):
        print(f"❌ Scripts directory '{scripts_dir}' not found!")
        print("Please ensure the scripts folder exists.")
        return

    print(
        "📁 All utility scripts have been moved to the 'scripts/' folder for better organization."
    )
    print("\n🎯 To update all markdown files with image displays:")
    print(f"   cd {scripts_dir}")
    print("   python update_all_images.py")

    print("\n📚 Available scripts in the scripts folder:")
    print("   - update_all_images.py          # Main launcher (recommended)")
    print("   - update_markdown_images_v2.py  # Update README files")
    print("   - update_summary_images.py      # Update summary files")
    print("   - README.md                     # Complete documentation")

    print("\n💡 For detailed usage instructions and documentation:")
    print(f"   See: {scripts_dir}/README.md")

    print("\n🔧 To run the main launcher directly:")
    print(f"   python {scripts_dir}/update_all_images.py")


if __name__ == "__main__":
    main()
