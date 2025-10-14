#!/usr/bin/env python3
"""
Validation script to check if UI-Venus Ground model setup is correct.
Checks for annotation files, images, and proper format.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes for pretty output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check_annotation_file(file_path: str, image_base_dir: str = None) -> Tuple[bool, List[str], Dict]:
    """
    Check if an annotation file is valid and properly formatted.
    Returns (is_valid, issues, stats)
    """
    issues = []
    stats = {
        'total_annotations': 0,
        'valid_annotations': 0,
        'images_found': 0,
        'images_missing': 0,
        'bbox_valid': 0,
        'bbox_invalid': 0
    }

    if not os.path.exists(file_path):
        issues.append(f"File does not exist: {file_path}")
        return False, issues, stats

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON format: {e}")
        return False, issues, stats

    if not isinstance(data, list):
        issues.append("JSON should be a list of annotations")
        return False, issues, stats

    stats['total_annotations'] = len(data)

    # Required fields for each annotation
    required_fields = ["img_filename", "bbox", "instruction"]

    for idx, annotation in enumerate(data):
        is_valid_annotation = True

        # Check required fields
        for field in required_fields:
            if field not in annotation:
                issues.append(f"Annotation {idx}: missing required field '{field}'")
                is_valid_annotation = False

        # Check bbox format
        if "bbox" in annotation:
            bbox = annotation["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                issues.append(f"Annotation {idx}: invalid bbox format (should be [x1, y1, x2, y2])")
                stats['bbox_invalid'] += 1
                is_valid_annotation = False
            else:
                # Check if bbox values are numbers
                if not all(isinstance(x, (int, float)) for x in bbox):
                    issues.append(f"Annotation {idx}: bbox values must be numbers")
                    stats['bbox_invalid'] += 1
                    is_valid_annotation = False
                else:
                    stats['bbox_valid'] += 1

        # Check if image file exists (if image_base_dir provided)
        if image_base_dir and "img_filename" in annotation:
            img_path = os.path.join(image_base_dir, annotation["img_filename"])
            if os.path.exists(img_path):
                stats['images_found'] += 1
            else:
                stats['images_missing'] += 1
                if idx < 5:  # Only report first 5 missing images
                    issues.append(f"Annotation {idx}: image not found: {img_path}")

        if is_valid_annotation:
            stats['valid_annotations'] += 1

    return len(issues) == 0, issues, stats


def validate_directory_structure():
    """
    Check if the expected directory structure exists.
    """
    print(f"\n{BOLD}Checking Directory Structure:{RESET}")
    print("=" * 60)

    expected_dirs = [
        ("ScreenSpot-v2-variants", "Primary annotation directory"),
        ("Screenspot-pro/annotations", "ScreenSpot Pro annotations"),
        ("data/osworld_meta", "OSWorld annotations"),
        ("data/ui_vision/ui-vision/annotations", "UI Vision annotations"),
        ("CAGUI/CAGUI_grounding/json_files", "CAGUI annotations"),
    ]

    dirs_status = []
    for dir_path, description in expected_dirs:
        exists = os.path.exists(dir_path)
        status_icon = f"{GREEN}✓{RESET}" if exists else f"{RED}✗{RESET}"
        json_count = 0

        if exists:
            # Count JSON files
            json_files = list(Path(dir_path).glob("*.json"))
            json_count = len(json_files)

        dirs_status.append((dir_path, exists, json_count, description))
        print(f"{status_icon} {dir_path:50} {description}")
        if exists and json_count > 0:
            print(f"  {BLUE}→ Found {json_count} JSON file(s){RESET}")

    return dirs_status


def validate_annotations():
    """
    Validate all annotation files found.
    """
    print(f"\n{BOLD}Validating Annotation Files:{RESET}")
    print("=" * 60)

    # Directories to check for annotations
    annotation_configs = [
        ("ScreenSpot-v2-variants", "extracted_screenspot_v2"),
        ("Screenspot-pro/annotations", "Screenspot-pro/images"),
        ("data/osworld_meta", "data/osworld"),
        ("data/ui_vision/ui-vision/annotations/element_grounding",
         "data/ui_vision/ui-vision/images"),
        ("CAGUI/CAGUI_grounding/json_files", "CAGUI/CAGUI_grounding/images"),
    ]

    total_valid = 0
    total_files = 0

    for ann_dir, img_dir in annotation_configs:
        if not os.path.exists(ann_dir):
            continue

        json_files = list(Path(ann_dir).glob("*.json"))
        json_files = [f for f in json_files if f.name != "dataset_info.json"]

        if not json_files:
            continue

        print(f"\n{BLUE}{ann_dir}:{RESET}")

        for json_file in json_files:
            total_files += 1
            is_valid, issues, stats = check_annotation_file(str(json_file), img_dir)

            status_icon = f"{GREEN}✓{RESET}" if is_valid else f"{YELLOW}⚠{RESET}"
            print(f"  {status_icon} {json_file.name}")
            print(f"     Annotations: {stats['total_annotations']}, Valid: {stats['valid_annotations']}")

            if stats['total_annotations'] > 0:
                if img_dir and os.path.exists(img_dir):
                    print(f"     Images: Found {stats['images_found']}, Missing {stats['images_missing']}")

            if not is_valid and issues:
                print(f"     {RED}Issues:{RESET}")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"       - {issue}")
                if len(issues) > 3:
                    print(f"       ... and {len(issues) - 3} more issues")

            if is_valid:
                total_valid += 1

    return total_valid, total_files


def test_model_loading():
    """
    Test if we can load tasks with the current setup.
    """
    print(f"\n{BOLD}Testing Task Loading:{RESET}")
    print("=" * 60)

    # Test directories
    test_configs = [
        ("ScreenSpot-v2-variants", "all"),
        ("Screenspot-pro/annotations", "all"),
    ]

    for test_dir, task in test_configs:
        if not os.path.exists(test_dir):
            print(f"{RED}✗{RESET} {test_dir} does not exist")
            continue

        json_files = list(Path(test_dir).glob("*.json"))
        json_files = [f for f in json_files if f.name != "dataset_info.json"]

        if json_files:
            print(f"{GREEN}✓{RESET} {test_dir} has {len(json_files)} JSON file(s)")

            # Simulate what the model script does
            total_tasks = 0
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_tasks += len(data)
                except:
                    pass

            print(f"  → Would load {total_tasks} task(s) from this directory")
        else:
            print(f"{YELLOW}⚠{RESET} {test_dir} has no annotation files")


def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}UI-Venus Ground Model Setup Validation{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Check directory structure
    dirs_status = validate_directory_structure()

    # Validate annotation files
    valid_files, total_files = validate_annotations()

    # Test model loading
    test_model_loading()

    # Summary
    print(f"\n{BOLD}Summary:{RESET}")
    print("=" * 60)

    dirs_exist = sum(1 for _, exists, _, _ in dirs_status if exists)
    dirs_with_json = sum(1 for _, exists, json_count, _ in dirs_status if exists and json_count > 0)

    print(f"Directories: {dirs_exist}/{len(dirs_status)} exist")
    print(f"Directories with JSON: {dirs_with_json}/{len(dirs_status)}")
    print(f"Valid annotation files: {valid_files}/{total_files}")

    if dirs_with_json > 0 and valid_files > 0:
        print(f"\n{GREEN}{BOLD}✓ Setup appears to be working!{RESET}")
        print(f"\nYou can now test the model with:")
        print(f"  {BLUE}bash scripts/run_gd_7b.sh{RESET}")
    else:
        print(f"\n{YELLOW}{BOLD}⚠ Setup incomplete!{RESET}")
        print(f"\nIssues to fix:")
        if dirs_with_json == 0:
            print(f"  - No directories contain annotation JSON files")
        if valid_files == 0:
            print(f"  - No valid annotation files found")
        print(f"\nRun the following to set up sample data:")
        print(f"  {BLUE}python3 setup_annotations.py --setup-dirs --create-sample{RESET}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()