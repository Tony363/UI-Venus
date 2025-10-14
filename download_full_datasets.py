#!/usr/bin/env python3
"""
Complete dataset downloader for UI-Venus Ground model.
Downloads HuggingFace datasets WITH full annotations (bbox, instructions, etc.)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

try:
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Error: Required libraries not installed. {e}")
    print("Install with: pip install datasets pillow numpy")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm for progress bars")
    # Fallback tqdm
    def tqdm(iterable, desc=None, total=None, **kwargs):
        return iterable


class ScreenSpotDownloader:
    """Downloads and processes ScreenSpot datasets with complete annotations."""

    DATASETS = {
        "screenspot-pro": {
            "name": "likaixin/ScreenSpot-Pro",
            "expected_samples": 1585,
            "bbox_format": "xyxy",  # [x1, y1, x2, y2]
            "has_chinese": True,
        },
        "screenspot-v2": {
            "name": "OS-Copilot/ScreenSpot-v2",
            "alternative": "likaixin/ScreenSpot-v2-variants",
            "expected_samples": 755,  # variants version
            "bbox_format": "xywh",  # [x, y, width, height]
            "has_chinese": False,
        }
    }

    def __init__(self, output_dir: str = ".", cache_dir: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir

    def convert_bbox_format(self, bbox: List[float], from_format: str, to_format: str) -> List[float]:
        """Convert between bbox formats."""
        if from_format == to_format:
            return bbox

        if from_format == "xywh" and to_format == "xyxy":
            x, y, w, h = bbox
            return [x, y, x + w, y + h]
        elif from_format == "xyxy" and to_format == "xywh":
            x1, y1, x2, y2 = bbox
            return [x1, y1, x2 - x1, y2 - y1]
        else:
            raise ValueError(f"Unknown conversion: {from_format} to {to_format}")

    def download_screenspot_pro(self, save_images: bool = True) -> Dict:
        """Download ScreenSpot-Pro dataset with full annotations."""
        print("\n" + "="*60)
        print("Downloading ScreenSpot-Pro Dataset")
        print("="*60)

        dataset_info = self.DATASETS["screenspot-pro"]
        output_path = self.output_dir / "Screenspot-pro"

        # Create directories
        (output_path / "images").mkdir(parents=True, exist_ok=True)
        (output_path / "annotations").mkdir(parents=True, exist_ok=True)

        try:
            # Load dataset
            print(f"Loading dataset: {dataset_info['name']}")
            dataset = load_dataset(
                dataset_info['name'],
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            stats = {
                "total_samples": 0,
                "images_saved": 0,
                "annotations_saved": 0,
                "errors": 0,
                "applications": set(),
                "platforms": set()
            }

            # Process each split
            for split_name in dataset.keys():
                print(f"\nProcessing split: {split_name}")
                split_data = dataset[split_name]

                annotations = []
                manifest_entries = []

                for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                    try:
                        # Save image if requested
                        img_filename = f"{split_name}_{idx:06d}.png"
                        img_rel_path = f"images/{img_filename}"

                        if save_images and 'image' in sample:
                            img_path = output_path / "images" / img_filename
                            if not img_path.exists():  # Skip if already downloaded
                                img = sample['image']
                                if hasattr(img, 'save'):
                                    img.save(img_path, 'PNG')
                                    stats['images_saved'] += 1

                        # Extract ALL annotation data
                        annotation = {
                            "img_filename": img_rel_path,
                            "bbox": sample.get('bbox', [0, 0, 100, 100]),
                            "instruction": sample.get('instruction', ''),
                            "application": sample.get('application', 'unknown'),
                            "platform": sample.get('platform', 'unknown'),
                            "ui_type": sample.get('ui_type', 'unknown'),
                            "group": sample.get('group', ''),
                            "ui_id": sample.get('ui_id', f"{split_name}_{idx}"),
                            "id": f"{split_name}_{idx}",
                            "label": sample.get('label', -1),
                        }

                        # Add Chinese instruction if available
                        if 'instruction_cn' in sample:
                            annotation['instruction_cn'] = sample['instruction_cn']

                        # Get image size if available
                        if 'image' in sample and hasattr(sample['image'], 'size'):
                            annotation['img_size'] = list(sample['image'].size)

                        annotations.append(annotation)

                        # Create manifest entry
                        manifest_entry = {
                            "split": split_name,
                            "index": idx,
                            "output_path": img_rel_path,
                            **annotation  # Include all annotation data
                        }
                        manifest_entries.append(manifest_entry)

                        # Update stats
                        stats['applications'].add(annotation['application'])
                        stats['platforms'].add(annotation['platform'])
                        stats['total_samples'] += 1
                        stats['annotations_saved'] += 1

                    except Exception as e:
                        print(f"\nError processing sample {idx}: {e}")
                        stats['errors'] += 1
                        continue

                # Save annotations for this split
                ann_file = output_path / "annotations" / f"{split_name}.json"
                with open(ann_file, 'w') as f:
                    json.dump(annotations, f, indent=2)
                print(f"Saved {len(annotations)} annotations to {ann_file}")

                # Save manifest
                manifest_file = output_path / f"manifest_{split_name}.jsonl"
                with open(manifest_file, 'w') as f:
                    for entry in manifest_entries:
                        f.write(json.dumps(entry) + '\n')
                print(f"Saved manifest to {manifest_file}")

            # Save dataset statistics
            stats_file = output_path / "dataset_stats.json"
            stats['applications'] = sorted(list(stats['applications']))
            stats['platforms'] = sorted(list(stats['platforms']))
            stats['download_time'] = datetime.now().isoformat()

            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"\n✅ ScreenSpot-Pro download complete!")
            print(f"   Total samples: {stats['total_samples']}")
            print(f"   Images saved: {stats['images_saved']}")
            print(f"   Annotations saved: {stats['annotations_saved']}")
            print(f"   Applications: {', '.join(stats['applications'][:5])}...")
            print(f"   Platforms: {', '.join(stats['platforms'])}")

            return stats

        except Exception as e:
            print(f"❌ Error downloading ScreenSpot-Pro: {e}")
            return {}

    def download_screenspot_v2(self, save_images: bool = True) -> Dict:
        """Download ScreenSpot-v2 dataset with full annotations."""
        print("\n" + "="*60)
        print("Downloading ScreenSpot-v2 Dataset")
        print("="*60)

        dataset_info = self.DATASETS["screenspot-v2"]
        output_path = self.output_dir / "ScreenSpot-v2-variants"

        # Create directories
        (output_path / "images").mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        # Try primary dataset first, then alternative
        dataset_names = [dataset_info['name']]
        if 'alternative' in dataset_info:
            dataset_names.append(dataset_info['alternative'])

        dataset = None
        used_name = None

        for ds_name in dataset_names:
            try:
                print(f"Trying to load: {ds_name}")
                dataset = load_dataset(
                    ds_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                used_name = ds_name
                print(f"✓ Successfully loaded {ds_name}")
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue

        if dataset is None:
            print("❌ Could not load ScreenSpot-v2 dataset from any source")
            return {}

        stats = {
            "total_samples": 0,
            "images_saved": 0,
            "annotations_saved": 0,
            "errors": 0,
            "data_sources": set(),
            "data_types": set()
        }

        # Process each split
        for split_name in dataset.keys():
            print(f"\nProcessing split: {split_name}")
            split_data = dataset[split_name]

            annotations = []
            manifest_entries = []

            for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                try:
                    # Handle different possible field names
                    img_filename_orig = sample.get('img_filename',
                                                  sample.get('image_id',
                                                           f"{split_name}_{idx:06d}.png"))

                    # Save image if requested
                    if save_images and 'image' in sample:
                        img_filename = f"{split_name}_{idx:06d}.png"
                        img_path = output_path / "images" / img_filename
                        img_rel_path = f"images/{img_filename}"

                        if not img_path.exists():
                            img = sample['image']
                            if hasattr(img, 'save'):
                                img.save(img_path, 'PNG')
                                stats['images_saved'] += 1
                    else:
                        img_rel_path = img_filename_orig

                    # Get bbox - handle both formats
                    bbox_raw = sample.get('bbox', [0, 0, 100, 100])

                    # Convert bbox format if needed (v2 uses xywh, eval expects xyxy)
                    if dataset_info['bbox_format'] == 'xywh' and len(bbox_raw) == 4:
                        bbox = self.convert_bbox_format(bbox_raw, 'xywh', 'xyxy')
                    else:
                        bbox = bbox_raw

                    # Extract annotation data
                    annotation = {
                        "img_filename": img_rel_path,
                        "bbox": bbox,
                        "instruction": sample.get('instruction', ''),
                        "data_type": sample.get('data_type',
                                               sample.get('ui_type', 'unknown')),
                        "data_source": sample.get('data_source',
                                                 sample.get('platform', 'unknown')),
                        "id": f"{split_name}_{idx}",
                    }

                    # Add additional instruction variants if available
                    if 'action' in sample:
                        annotation['action'] = sample['action']
                    if 'description' in sample:
                        annotation['description'] = sample['description']
                    if 'negative_instruction' in sample:
                        annotation['negative_instruction'] = sample['negative_instruction']

                    # Get image size if available
                    if 'image' in sample and hasattr(sample['image'], 'size'):
                        annotation['img_size'] = list(sample['image'].size)
                    elif 'img_size' in sample:
                        annotation['img_size'] = sample['img_size']

                    # Map to expected format for eval script
                    annotation['platform'] = annotation['data_source']
                    annotation['ui_type'] = annotation['data_type']
                    annotation['application'] = annotation.get('data_source', 'unknown')

                    annotations.append(annotation)

                    # Create manifest entry
                    manifest_entry = {
                        "split": split_name,
                        "index": idx,
                        "output_path": img_rel_path,
                        **annotation
                    }
                    manifest_entries.append(manifest_entry)

                    # Update stats
                    stats['data_sources'].add(annotation['data_source'])
                    stats['data_types'].add(annotation['data_type'])
                    stats['total_samples'] += 1
                    stats['annotations_saved'] += 1

                except Exception as e:
                    print(f"\nError processing sample {idx}: {e}")
                    stats['errors'] += 1
                    continue

            # Save annotations for this split
            ann_file = output_path / f"{split_name}.json"
            with open(ann_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"Saved {len(annotations)} annotations to {ann_file}")

            # Save manifest
            manifest_file = output_path / f"manifest_{split_name}.jsonl"
            with open(manifest_file, 'w') as f:
                for entry in manifest_entries:
                    f.write(json.dumps(entry) + '\n')
            print(f"Saved manifest to {manifest_file}")

        # Save dataset statistics
        stats_file = output_path / "dataset_stats.json"
        stats['data_sources'] = sorted(list(stats['data_sources']))
        stats['data_types'] = sorted(list(stats['data_types']))
        stats['download_time'] = datetime.now().isoformat()
        stats['dataset_name'] = used_name

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✅ ScreenSpot-v2 download complete!")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Images saved: {stats['images_saved']}")
        print(f"   Annotations saved: {stats['annotations_saved']}")
        print(f"   Data sources: {', '.join(stats['data_sources'])}")
        print(f"   Data types: {', '.join(stats['data_types'])}")

        return stats

    def validate_annotations(self, dataset_name: str) -> bool:
        """Validate that annotations are complete and properly formatted."""
        if dataset_name == "screenspot-pro":
            ann_dir = self.output_dir / "Screenspot-pro" / "annotations"
        else:
            ann_dir = self.output_dir / "ScreenSpot-v2-variants"

        if not ann_dir.exists():
            print(f"❌ Annotation directory does not exist: {ann_dir}")
            return False

        json_files = list(ann_dir.glob("*.json"))
        if not json_files:
            print(f"❌ No annotation files found in {ann_dir}")
            return False

        total_valid = 0
        total_invalid = 0

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    annotations = json.load(f)

                if not isinstance(annotations, list):
                    print(f"❌ {json_file.name}: Not a list")
                    total_invalid += 1
                    continue

                required_fields = ['img_filename', 'bbox', 'instruction']

                for ann in annotations:
                    if all(field in ann for field in required_fields):
                        if isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
                            total_valid += 1
                        else:
                            total_invalid += 1
                    else:
                        total_invalid += 1

                print(f"✓ {json_file.name}: {len(annotations)} annotations")

            except Exception as e:
                print(f"❌ Error reading {json_file.name}: {e}")
                total_invalid += 1

        print(f"\nValidation Summary:")
        print(f"  Valid annotations: {total_valid}")
        print(f"  Invalid annotations: {total_invalid}")

        return total_invalid == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download UI-Venus datasets with complete annotations from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for downloaded datasets"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["screenspot-pro", "screenspot-v2", "both"],
        default="both",
        help="Which dataset(s) to download"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip downloading images (only get annotations)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate annotations after download"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = ScreenSpotDownloader(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )

    save_images = not args.no_images

    # Download requested datasets
    if args.dataset in ["screenspot-pro", "both"]:
        stats_pro = downloader.download_screenspot_pro(save_images=save_images)
        if args.validate and stats_pro:
            downloader.validate_annotations("screenspot-pro")

    if args.dataset in ["screenspot-v2", "both"]:
        stats_v2 = downloader.download_screenspot_v2(save_images=save_images)
        if args.validate and stats_v2:
            downloader.validate_annotations("screenspot-v2")

    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the annotation files in:")
    print(f"   - {args.output_dir}/Screenspot-pro/annotations/")
    print(f"   - {args.output_dir}/ScreenSpot-v2-variants/")
    print("\n2. Run the model with:")
    print("   bash scripts/run_gd_7b.sh")
    print("\n3. The model will now find all tasks with complete annotations!")


if __name__ == "__main__":
    main()