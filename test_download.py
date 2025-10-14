#!/usr/bin/env python3
"""
Quick test script to verify HuggingFace dataset structure and download small sample.
"""

import json
from datasets import load_dataset

def test_dataset_structure():
    """Test loading a small sample and show the structure."""

    print("Testing ScreenSpot dataset structure...")
    print("="*60)

    try:
        # Try to load just one sample to check structure
        print("\n1. Testing ScreenSpot-Pro structure:")
        ds_pro = load_dataset("likaixin/ScreenSpot-Pro", split="train", streaming=True)
        sample_pro = next(iter(ds_pro))

        print("   Available fields:")
        for key in sample_pro.keys():
            if key != 'image':
                value = sample_pro[key]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"   - {key}: {value}")

        # Check bbox format
        if 'bbox' in sample_pro:
            print(f"\n   Bbox format: {sample_pro['bbox']} (length={len(sample_pro['bbox'])})")
            print(f"   Instruction: {sample_pro.get('instruction', 'N/A')}")
            print(f"   Application: {sample_pro.get('application', 'N/A')}")

    except Exception as e:
        print(f"   Error loading ScreenSpot-Pro: {e}")

    try:
        print("\n2. Testing ScreenSpot-v2 structure:")
        # Try alternative sources
        for ds_name in ["OS-Copilot/ScreenSpot-v2", "likaixin/ScreenSpot-v2-variants"]:
            try:
                print(f"   Trying: {ds_name}")
                ds_v2 = load_dataset(ds_name, split="train", streaming=True)
                sample_v2 = next(iter(ds_v2))

                print("   Available fields:")
                for key in sample_v2.keys():
                    if key != 'image':
                        value = sample_v2[key]
                        if isinstance(value, str) and len(value) > 50:
                            value = value[:50] + "..."
                        print(f"   - {key}: {value}")

                if 'bbox' in sample_v2:
                    print(f"\n   Bbox format: {sample_v2['bbox']} (length={len(sample_v2['bbox'])})")

                break

            except Exception as e:
                print(f"     Failed: {e}")
                continue

    except Exception as e:
        print(f"   Error with all v2 sources: {e}")

    print("\n" + "="*60)
    print("Test complete! Use download_full_datasets.py to get complete data.")


def download_tiny_sample():
    """Download just 5 samples to test the full pipeline."""

    print("\nDownloading tiny sample (5 annotations)...")

    try:
        # Get 5 samples
        ds = load_dataset("likaixin/ScreenSpot-Pro", split="train[:5]")

        annotations = []
        for idx, sample in enumerate(ds):
            ann = {
                "img_filename": f"test_image_{idx}.png",
                "bbox": sample.get('bbox', [0, 0, 100, 100]),
                "instruction": sample.get('instruction', 'test instruction'),
                "application": sample.get('application', 'unknown'),
                "platform": sample.get('platform', 'unknown'),
                "ui_type": sample.get('ui_type', 'unknown'),
                "id": f"test_{idx}"
            }
            annotations.append(ann)

        # Save test annotations
        with open("test_annotations.json", "w") as f:
            json.dump(annotations, f, indent=2)

        print(f"✅ Saved {len(annotations)} test annotations to test_annotations.json")
        print("\nSample annotation:")
        print(json.dumps(annotations[0], indent=2))

    except Exception as e:
        print(f"❌ Error downloading sample: {e}")


if __name__ == "__main__":
    test_dataset_structure()
    download_tiny_sample()