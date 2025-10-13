#!/usr/bin/env python3
"""
Extract images and metadata from local HuggingFace datasets (Arrow format).
Preserves folder structure and creates a metadata manifest.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import re

try:
    from datasets import load_from_disk, Dataset, DatasetDict, Features
    from datasets import Image as HFImage
except ImportError:
    print("Error: datasets library not installed. Install with: pip install datasets")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow library not installed. Install with: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm isn't installed
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
    strategy: str
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
    filename = None

    if strategy == "preserve" and original_path:
        # Try to preserve original filename
        orig_path = Path(original_path)
        filename = sanitize_filename(orig_path.name)

        # If original path contains label name, try to preserve more structure
        if label_name and label_name in orig_path.parts:
            try:
                label_idx = orig_path.parts.index(label_name)
                # Keep everything after the label
                sub_parts = orig_path.parts[label_idx+1:]
                if sub_parts:
                    # Add subdirectories
                    for sub in sub_parts[:-1]:
                        parts.append(sanitize_filename(sub))
                    filename = sanitize_filename(sub_parts[-1])
            except Exception:
                pass

    elif strategy == "indexed":
        # Use index-based naming
        filename = f"image_{idx:06d}.jpg"

    # Fallback to index if no filename determined
    if not filename:
        # Try to extract original filename if possible
        if original_path:
            filename = sanitize_filename(Path(original_path).name)
        else:
            filename = f"image_{idx:06d}.jpg"

    # Ensure proper extension
    if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']):
        # Default to .jpg if no recognized extension
        name, _ = os.path.splitext(filename)
        filename = name + ".jpg"

    parts.append(filename)
    return Path(*parts)


def extract_dataset_split(
    dataset: Dataset,
    output_dir: Path,
    split_name: str,
    strategy: str,
    manifest_file
) -> Tuple[int, int, int]:
    """
    Extract all images from a dataset split.
    Returns (success_count, skip_count, error_count)
    """
    success_count = 0
    skip_count = 0
    error_count = 0

    print(f"\nProcessing split '{split_name}' with {len(dataset)} examples...")

    # Get features for label mapping
    features = dataset.features if hasattr(dataset, 'features') else None

    for idx in tqdm(range(len(dataset)), desc=f"Extracting {split_name}", unit="img"):
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
                original_path, strategy
            )

            output_path = output_dir / rel_path

            # Create directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            if hasattr(img, 'save'):
                # Determine format from extension
                ext = output_path.suffix.lower()

                # Handle RGBA images (with transparency)
                if img.mode == 'RGBA':
                    if ext in ['.jpg', '.jpeg']:
                        # Convert RGBA to RGB for JPEG
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                        rgb_img.save(output_path, 'JPEG', quality=95)
                    else:
                        # Save as PNG to preserve transparency
                        output_path = output_path.with_suffix('.png')
                        img.save(output_path, 'PNG')
                else:
                    # Handle RGB and other modes
                    if ext in ['.jpg', '.jpeg']:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(output_path, 'JPEG', quality=95)
                    elif ext == '.png':
                        img.save(output_path, 'PNG')
                    else:
                        # Default to JPEG for RGB, PNG for others
                        if img.mode in ['RGB', 'L']:
                            output_path = output_path.with_suffix('.jpg')
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(output_path, 'JPEG', quality=95)
                        else:
                            output_path = output_path.with_suffix('.png')
                            img.save(output_path, 'PNG')
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
                "original_path": original_path
            }

            # Add other metadata fields (excluding image data)
            for key, value in example.items():
                if key not in ["image", "img", "picture", "photo"] and key != "label":
                    try:
                        # Only include JSON-serializable data
                        json.dumps(value)
                        metadata[key] = value
                    except Exception:
                        # Skip non-serializable fields
                        pass

            manifest_file.write(json.dumps(metadata) + "\n")
            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError processing index {idx}: {e}", file=sys.stderr)

    return success_count, skip_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract images and metadata from local HuggingFace datasets"
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
        "--strategy",
        choices=["preserve", "indexed"],
        default="preserve",
        help="Naming strategy: 'preserve' keeps original names, 'indexed' uses sequential numbering"
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

    # Open manifest file
    manifest_path = output_dir / args.manifest

    total_success = 0
    total_skip = 0
    total_error = 0

    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        # Handle different dataset types
        if isinstance(dataset, DatasetDict):
            # Multiple splits
            print(f"Dataset contains {len(dataset)} splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                success, skip, error = extract_dataset_split(
                    split_data, output_dir, split_name,
                    args.strategy, manifest_file
                )
                total_success += success
                total_skip += skip
                total_error += error

        elif isinstance(dataset, Dataset):
            # Single split
            success, skip, error = extract_dataset_split(
                dataset, output_dir, "train",
                args.strategy, manifest_file
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

    # Show sample of directory structure
    print("\nSample directory structure:")
    sample_count = 0
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{Path(root).name}/")

        # Show first few files in each directory
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:
            if not file.endswith('.jsonl'):
                print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... ({len(files) - 3} more files)")

        sample_count += 1
        if sample_count >= 5:
            print("... (more directories)")
            break


if __name__ == "__main__":
    main()