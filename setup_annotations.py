#!/usr/bin/env python3
"""
Script to set up proper annotation files for UI-Venus Ground model evaluation.
This creates the necessary JSON annotation files in the expected format.
"""

import json
import os
import sys
from pathlib import Path
import argparse
from typing import Dict, List, Any

def create_sample_annotations(output_dir: str, image_dir: str, num_samples: int = 5):
    """
    Create sample annotation files for testing the model.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find available images
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []

    if os.path.exists(image_dir):
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    rel_path = os.path.relpath(os.path.join(root, file), image_dir)
                    image_files.append(rel_path)
                    if len(image_files) >= num_samples:
                        break
            if len(image_files) >= num_samples:
                break

    if not image_files:
        print(f"Warning: No images found in {image_dir}")
        return []

    # Sample UI instructions and bounding boxes
    sample_instructions = [
        "click the button",
        "close this window",
        "open settings",
        "tap on the icon",
        "select the menu"
    ]

    # Create sample annotations
    annotations = []
    for idx, img_file in enumerate(image_files[:num_samples]):
        annotation = {
            "img_filename": img_file,
            "bbox": [100 + idx * 10, 100 + idx * 10, 200 + idx * 10, 150 + idx * 10],  # Sample bbox
            "instruction": sample_instructions[idx % len(sample_instructions)],
            "application": "sample_app",
            "id": f"sample_{idx}",
            "action": f"Sample action for {sample_instructions[idx % len(sample_instructions)]}",
            "description": "A sample UI element",
            "ui_type": "button",
            "platform": "desktop",
            "img_size": [1920, 1080]  # Default size, should be updated with actual
        }
        annotations.append(annotation)

    return annotations


def convert_manifest_to_annotations(manifest_path: str, output_dir: str, dataset_name: str = "screenspot"):
    """
    Convert manifest.jsonl to proper annotation JSON files.
    """
    if not os.path.exists(manifest_path):
        print(f"Manifest file not found: {manifest_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    annotations_by_split = {}

    # Read manifest file
    with open(manifest_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                split = entry.get('split', 'train')

                if split not in annotations_by_split:
                    annotations_by_split[split] = []

                # Create annotation in expected format
                annotation = {
                    "img_filename": entry.get('output_path', ''),
                    "bbox": [100, 100, 200, 150],  # Default bbox, needs actual data
                    "instruction": "click on element",  # Default instruction, needs actual data
                    "application": dataset_name,
                    "id": f"{split}_{entry.get('index', 0)}",
                    "platform": "desktop" if entry.get('image_size', [0, 0])[0] > 1000 else "mobile",
                    "img_size": entry.get('image_size', [1920, 1080])
                }

                annotations_by_split[split].append(annotation)

            except json.JSONDecodeError:
                continue

    # Write annotation files for each split
    files_created = []
    for split, annotations in annotations_by_split.items():
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        files_created.append(output_file)
        print(f"Created {output_file} with {len(annotations)} annotations")

    return files_created


def download_proper_datasets():
    """
    Provide instructions for downloading the proper annotated datasets.
    """
    print("\n" + "="*60)
    print("IMPORTANT: Dataset Download Instructions")
    print("="*60)

    datasets = {
        "ScreenSpot-v2": {
            "url": "https://huggingface.co/datasets/likaixin/ScreenSpot-v2-variants",
            "note": "Contains UI screenshots with grounding annotations"
        },
        "ScreenSpot-Pro": {
            "url": "https://huggingface.co/datasets/likaixin/ScreenSpot-pro",
            "note": "Professional UI dataset with detailed annotations"
        },
        "OSWorld": {
            "url": "https://github.com/xlang-ai/OSWorld",
            "note": "Operating system UI interactions dataset"
        }
    }

    print("\nTo get the full annotated datasets, you need to:")
    print("\n1. Download the datasets with annotations:")
    for name, info in datasets.items():
        print(f"\n   {name}:")
        print(f"   - URL: {info['url']}")
        print(f"   - Note: {info['note']}")

    print("\n2. Use the HuggingFace datasets library:")
    print("""
   from datasets import load_dataset

   # For ScreenSpot datasets
   ds = load_dataset("likaixin/ScreenSpot-v2-variants")
   ds.save_to_disk("./ScreenSpot-v2-full")

   # Extract both images AND annotations
   # The annotations should include: instruction, bbox, etc.
   """)

    print("\n3. Check if the dataset has annotation columns:")
    print("""
   # Check dataset structure
   print(ds.column_names)  # Should show columns beyond just 'image'
   """)


def setup_directory_structure():
    """
    Create the expected directory structure for annotations.
    """
    directories = [
        "ScreenSpot-v2-variants",
        "Screenspot-pro/annotations",
        "Screenspot-pro/images",
        "data/osworld",
        "data/osworld_meta",
        "data/ui_vision/ui-vision/images",
        "data/ui_vision/ui-vision/annotations/element_grounding",
        "CAGUI/CAGUI_grounding/images",
        "CAGUI/CAGUI_grounding/json_files"
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

    return directories


def main():
    parser = argparse.ArgumentParser(description="Set up annotations for UI-Venus Ground model")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample annotations for testing")
    parser.add_argument("--convert-manifest", type=str,
                       help="Path to manifest.jsonl to convert")
    parser.add_argument("--setup-dirs", action="store_true",
                       help="Create expected directory structure")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Base output directory")

    args = parser.parse_args()

    if args.setup_dirs:
        print("\nSetting up directory structure...")
        setup_directory_structure()

    if args.convert_manifest:
        print(f"\nConverting manifest: {args.convert_manifest}")

        # Convert for ScreenSpot-v2
        if "screenspot_v2" in args.convert_manifest.lower():
            output_dir = "ScreenSpot-v2-variants"
            image_dir = "extracted_screenspot_v2"
        else:
            output_dir = "Screenspot-pro/annotations"
            image_dir = "extracted_screenspot_pro"

        files = convert_manifest_to_annotations(args.convert_manifest, output_dir)

        if not files:
            print("No annotation files created. Creating sample annotations...")
            args.create_sample = True

    if args.create_sample:
        print("\nCreating sample annotations for testing...")

        # Create sample annotations for each expected directory
        test_configs = [
            ("ScreenSpot-v2-variants", "extracted_screenspot_v2/train/unlabeled"),
            ("Screenspot-pro/annotations", "Screenspot-pro/images"),
            ("data/osworld_meta", "data/osworld"),
            ("data/ui_vision/ui-vision/annotations/element_grounding",
             "data/ui_vision/ui-vision/images"),
            ("CAGUI/CAGUI_grounding/json_files", "CAGUI/CAGUI_grounding/images")
        ]

        for ann_dir, img_dir in test_configs:
            print(f"\nCreating annotations for {ann_dir}...")
            annotations = create_sample_annotations(ann_dir, img_dir, num_samples=10)

            if annotations:
                # Save as test.json
                output_file = os.path.join(ann_dir, "test.json")
                with open(output_file, 'w') as f:
                    json.dump(annotations, f, indent=2)
                print(f"✓ Created {output_file} with {len(annotations)} sample annotations")

    # Always show download instructions
    download_proper_datasets()

    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. If using sample data: Run 'bash scripts/run_gd_7b.sh' to test")
    print("2. For real data: Download the full annotated datasets")
    print("3. Verify with: find . -name '*.json' -path '*/annotations/*' | wc -l")


if __name__ == "__main__":
    main()