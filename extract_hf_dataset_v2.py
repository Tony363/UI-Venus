#!/usr/bin/env python3
"""
Enhanced HuggingFace dataset extraction with resume capability and robust error handling.
Optimized for extracting large image datasets with RGBA support.
"""

import argparse
import json
import os
import sys
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import re
import time

try:
    from datasets import load_from_disk, Dataset, DatasetDict
    from datasets import Image as HFImage
except ImportError:
    print("Error: datasets library not installed. Install with: pip install datasets")
    sys.exit(1)

try:
    from PIL import Image, ImageFile
    # Allow loading truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    print("Error: Pillow library not installed. Install with: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm for progress bars")
    tqdm = lambda x, **k: x


def load_dataset_from_path(path: str):
    """
    Load dataset from either a directory (saved with save_to_disk) or .arrow file.
    """
    path = Path(path)

    # Try loading as a saved dataset directory first
    if path.is_dir():
        try:
            print(f"Loading dataset from directory: {path}")
            ds = load_from_disk(str(path))
            return ds
        except Exception as e:
            print(f"Could not load as saved dataset directory: {e}")

            # Try to find arrow files in the directory
            arrow_files = list(path.rglob("*.arrow"))
            if arrow_files:
                arrow_file = arrow_files[0]
                print(f"Found arrow file: {arrow_file}")
                try:
                    return Dataset.from_file(str(arrow_file))
                except Exception as e2:
                    print(f"Could not load arrow file: {e2}")

    # Try loading as a single arrow file
    elif str(path).endswith(".arrow"):
        try:
            print(f"Loading dataset from arrow file: {path}")
            return Dataset.from_file(str(path))
        except Exception as e:
            print(f"Could not load arrow file: {e}")

    raise ValueError(f"Could not load dataset from: {path}")


def extract_image_data(example: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """
    Extract image data and original path from dataset example.
    Returns (image_object, original_path)
    """
    # Try getting image from 'image' column
    if "image" in example and example["image"] is not None:
        img = example["image"]

        # Handle PIL Image directly
        if hasattr(img, 'save'):  # PIL Image
            return img, None

        # Handle dict with path/bytes
        if isinstance(img, dict):
            # Try to get PIL image from bytes
            if "bytes" in img and img["bytes"]:
                try:
                    from io import BytesIO
                    return Image.open(BytesIO(img["bytes"])), img.get("path")
                except Exception:
                    pass

            # Try to load from path
            if "path" in img and img["path"]:
                path = img["path"]
                if os.path.exists(path):
                    try:
                        return Image.open(path), path
                    except Exception:
                        pass

    # Try alternative image column names
    for col in ["img", "picture", "photo"]:
        if col in example and example[col] is not None:
            if hasattr(example[col], 'save'):
                return example[col], None

    # Try path-based columns
    for col in ["image_path", "filepath", "file_path", "filename", "path"]:
        if col in example and example[col]:
            path = example[col]
            if isinstance(path, str) and os.path.exists(path):
                try:
                    return Image.open(path), path
                except Exception:
                    pass

    return None, None


def get_label_info(example: Dict[str, Any], features) -> Tuple[Optional[int], Optional[str]]:
    """
    Extract label ID and name from example.
    Returns (label_id, label_name)
    """
    if "label" not in example:
        return None, None

    label = example["label"]

    # Handle ClassLabel feature type
    if features and "label" in features:
        label_feature = features["label"]
        if hasattr(label_feature, "names"):
            names = label_feature.names
            if isinstance(label, int) and 0 <= label < len(names):
                return label, names[label]
            elif isinstance(label, str) and label in names:
                return names.index(label), label

    # Handle string label directly
    if isinstance(label, str):
        return None, label

    # Handle numeric label without names
    if isinstance(label, (int, float)):
        return int(label), f"label_{int(label)}"

    return None, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for filesystem safety.
    """
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    return name + ext


def derive_output_path(
    example: Dict[str, Any],
    idx: int,
    split_name: str,
    label_name: Optional[str],
    original_path: Optional[str],
    png_only: bool = False
) -> Path:
    """
    Derive output path for the extracted file.
    """
    # Start with split directory
    parts = [split_name]

    # Add label-based subdirectory if available
    if label_name:
        parts.append(sanitize_filename(label_name))
    else:
        parts.append("unlabeled")

    # Determine filename
    if original_path:
        # Try to preserve original filename
        orig_path = Path(original_path)
        filename = sanitize_filename(orig_path.name)
    else:
        filename = f"image_{idx:06d}"

    # Ensure proper extension
    if png_only:
        # Always use .png
        name, _ = os.path.splitext(filename)
        filename = name + ".png"
    else:
        # Ensure a valid image extension
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']):
            filename = filename + ".png"  # Default to PNG for safety

    parts.append(filename)
    return Path(*parts)


def extract_dataset_split(
    dataset: Dataset,
    output_dir: Path,
    split_name: str,
    manifest_file,
    resume: bool = False,
    png_only: bool = False,
    batch_size: Optional[int] = None
) -> Tuple[int, int, int]:
    """
    Extract all images from a dataset split with resume capability.
    Returns (success_count, skip_count, error_count)
    """
    success_count = 0
    skip_count = 0
    error_count = 0

    print(f"\nProcessing split '{split_name}' with {len(dataset)} examples...")
    if resume:
        print("Resume mode: Skipping existing files...")
    if png_only:
        print("PNG-only mode: All images will be saved as PNG...")

    # Get features for label mapping
    features = dataset.features if hasattr(dataset, 'features') else None

    # Process in batches if specified
    total_examples = len(dataset)

    for idx in tqdm(range(total_examples), desc=f"Extracting {split_name}", unit="img"):
        try:
            example = dataset[idx]

            # Extract image data
            img, original_path = extract_image_data(example)

            if img is None:
                skip_count += 1
                continue

            # Get label information
            label_id, label_name = get_label_info(example, features)

            # Derive output path
            rel_path = derive_output_path(
                example, idx, split_name, label_name,
                original_path, png_only=png_only
            )

            output_path = output_dir / rel_path

            # In resume mode, check if file already exists
            if resume and output_path.exists():
                # Still add to manifest for completeness
                metadata = {
                    "split": split_name,
                    "index": idx,
                    "output_path": str(rel_path),
                    "label_id": label_id,
                    "label_name": label_name,
                    "original_path": original_path,
                    "resumed": True
                }

                # Add other metadata fields (excluding image data)
                for key, value in example.items():
                    if key not in ["image", "img", "picture", "photo", "label"]:
                        try:
                            json.dumps(value)
                            metadata[key] = value
                        except Exception:
                            pass

                manifest_file.write(json.dumps(metadata) + "\n")
                success_count += 1
                continue

            # Create directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save image with atomic write (use temp file)
            if hasattr(img, 'save'):
                temp_path = output_path.with_suffix('.tmp')

                try:
                    if png_only or img.mode == 'RGBA':
                        # Save as PNG
                        if img.mode not in ['RGBA', 'RGB', 'L', 'LA', 'P']:
                            img = img.convert('RGBA')
                        img.save(temp_path, 'PNG', optimize=True)
                        # Rename temp to final
                        temp_path.replace(output_path)
                    else:
                        # Handle other modes
                        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                            # Convert to RGB for JPEG
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(temp_path, 'JPEG', quality=95)
                        else:
                            # Save as PNG by default
                            img.save(temp_path, 'PNG', optimize=True)
                        temp_path.replace(output_path)

                    # Clean up image object to free memory
                    if hasattr(img, 'close'):
                        img.close()

                except Exception as save_error:
                    # Clean up temp file if it exists
                    if temp_path.exists():
                        temp_path.unlink()
                    raise save_error
            else:
                skip_count += 1
                continue

            # Write metadata to manifest
            metadata = {
                "split": split_name,
                "index": idx,
                "output_path": str(rel_path),
                "label_id": label_id,
                "label_name": label_name,
                "original_path": original_path,
                "image_size": img.size if hasattr(img, 'size') else None,
                "image_mode": img.mode if hasattr(img, 'mode') else None
            }

            # Add other metadata fields (excluding image data)
            for key, value in example.items():
                if key not in ["image", "img", "picture", "photo", "label"]:
                    try:
                        # Only include JSON-serializable data
                        json.dumps(value)
                        metadata[key] = value
                    except Exception:
                        pass

            manifest_file.write(json.dumps(metadata) + "\n")
            manifest_file.flush()  # Ensure data is written
            success_count += 1

            # Periodic garbage collection for memory management
            if batch_size and idx > 0 and idx % batch_size == 0:
                gc.collect()
                time.sleep(0.1)  # Brief pause to avoid overwhelming system

        except KeyboardInterrupt:
            print("\nExtraction interrupted by user. Progress saved.")
            break
        except Exception as e:
            error_count += 1
            print(f"\nError processing index {idx}: {e}", file=sys.stderr)
            continue

    return success_count, skip_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract images and metadata from local HuggingFace datasets with resume support"
    )
    parser.add_argument(
        "input_path",
        help="Path to dataset directory or .arrow file"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save extracted images and metadata"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume extraction, skip existing files"
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Force all images to be saved as PNG"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process images in batches with garbage collection (helps with memory)"
    )
    parser.add_argument(
        "--manifest",
        default="manifest.jsonl",
        help="Name of the metadata manifest file (default: manifest.jsonl)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    try:
        dataset = load_dataset_from_path(args.input_path)
        print(f"Successfully loaded dataset from {args.input_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Open manifest file (append mode if resuming, write mode otherwise)
    manifest_path = output_dir / args.manifest
    manifest_mode = 'a' if args.resume and manifest_path.exists() else 'w'

    if manifest_mode == 'a':
        print(f"Appending to existing manifest: {manifest_path}")
    else:
        print(f"Creating new manifest: {manifest_path}")

    total_success = 0
    total_skip = 0
    total_error = 0

    with open(manifest_path, manifest_mode, encoding='utf-8') as manifest_file:
        # Handle different dataset types
        if isinstance(dataset, DatasetDict):
            # Multiple splits
            print(f"Dataset contains {len(dataset)} splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                success, skip, error = extract_dataset_split(
                    split_data, output_dir, split_name,
                    manifest_file,
                    resume=args.resume,
                    png_only=args.png_only,
                    batch_size=args.batch_size
                )
                total_success += success
                total_skip += skip
                total_error += error

        elif isinstance(dataset, Dataset):
            # Single split
            success, skip, error = extract_dataset_split(
                dataset, output_dir, "train",
                manifest_file,
                resume=args.resume,
                png_only=args.png_only,
                batch_size=args.batch_size
            )
            total_success += success
            total_skip += skip
            total_error += error

        else:
            print(f"Unsupported dataset type: {type(dataset)}", file=sys.stderr)
            sys.exit(1)

    # Print summary
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"✅ Successfully extracted: {total_success} images")
    print(f"⏭️  Skipped (no image data): {total_skip} examples")
    print(f"❌ Errors: {total_error} examples")
    print(f"📄 Manifest saved to: {manifest_path}")
    print(f"📁 Images saved to: {output_dir}")

    # Verify extraction completeness
    if isinstance(dataset, DatasetDict):
        total_expected = sum(len(split) for split in dataset.values())
    else:
        total_expected = len(dataset)

    total_processed = total_success + total_skip + total_error

    if total_processed < total_expected:
        print(f"\n⚠️  Warning: Only processed {total_processed}/{total_expected} examples")
        print("   Run with --resume to continue extraction")
    else:
        print(f"\n✓ All {total_expected} examples processed")

    # Show sample of directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), '').count(os.sep)
        if level <= 2:  # Only show first 2 levels
            indent = ' ' * 2 * level
            print(f"{indent}{Path(root).name}/")
            if level == 1:
                # Show count of files in each category
                img_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                if img_files:
                    print(f"{indent}  ({len(img_files)} images)")


if __name__ == "__main__":
    main()