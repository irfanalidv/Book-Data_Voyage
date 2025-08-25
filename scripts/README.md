# Utility Scripts

This folder contains utility scripts for maintaining and updating the Data Voyage book documentation.

## Scripts Overview

### 1. `update_markdown_images.py` (Original Version)
- **Purpose**: Initial attempt to add image displays to markdown files
- **Status**: Deprecated - replaced by improved version
- **Use Case**: Historical reference only

### 2. `update_markdown_images_v2.py` (Improved Version)
- **Purpose**: Add image displays to all README.md files
- **Features**: 
  - Works with actual PNG files found in chapters
  - Handles multiple images per chapter
  - Creates proper Generated Outputs sections
  - Updates both README.md and CHAPTERX_SUMMARY.md files
- **Use Case**: Primary script for updating README files with image displays

### 3. `update_summary_images.py` (Summary Files Version)
- **Purpose**: Add image displays to CHAPTERX_SUMMARY.md files
- **Features**:
  - Specifically targets summary files
  - Handles different file naming conventions
  - Adds images to existing Generated Outputs sections
- **Use Case**: Secondary script for updating summary files with image displays

## Usage

### Updating README Files with Images
```bash
cd scripts
python update_markdown_images_v2.py
```

### Updating Summary Files with Images
```bash
cd scripts
python update_summary_images.py
```

### Running Both Updates
```bash
cd scripts
python update_markdown_images_v2.py
python update_summary_images.py
```

## What These Scripts Do

1. **Scan Chapters**: Automatically detect all chapters with PNG image files
2. **Update README.md**: Add "Generated Visualizations" sections with actual image displays
3. **Update CHAPTERX_SUMMARY.md**: Add image displays to summary files
4. **Smart Detection**: Avoid duplicate updates and handle existing content gracefully

## Image Display Format

The scripts add markdown image syntax like this:
```markdown
![Image Title](image_filename.png)
```

This means:
- Images display automatically in GitHub, GitLab, and other markdown viewers
- Images are responsive and scale properly
- Alt text provides accessibility
- Images are clickable and can be viewed in full size

## Chapter Coverage

The scripts cover all chapters that have generated PNG files:
- **Chapters 1, 3, 6-25**: All have image displays in README.md
- **Chapters 12-24**: All have image displays in CHAPTERX_SUMMARY.md
- **Chapters 18, 20**: Already had image displays from manual updates

## Maintenance

- **Run After Code Updates**: Execute these scripts after generating new PNG files
- **Backup First**: Always backup markdown files before running updates
- **Review Changes**: Check the updated files to ensure proper formatting
- **Version Control**: Commit changes to git after verification

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)
- Access to the book directory structure
- PNG files must exist in chapter directories

## Troubleshooting

### Common Issues
1. **Missing PNG Files**: Scripts will skip chapters without images
2. **File Permission Errors**: Ensure write access to markdown files
3. **Encoding Issues**: Scripts use UTF-8 encoding for international characters

### Debug Mode
Add print statements or modify the scripts to see detailed processing information.

## Future Enhancements

Potential improvements for these scripts:
- Support for other image formats (JPG, SVG, etc.)
- Batch processing of multiple book projects
- Configuration file for custom image descriptions
- Integration with CI/CD pipelines
- Support for different markdown formats
