#!/usr/bin/env python3
"""Simplified dataset downloader for UI-Venus."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

@dataclass(frozen=True)
class DatasetConfig:
    key: str
    repo_id: str
    output_dirname: str


DATASETS: dict[str, DatasetConfig] = {
    "screenspot-pro": DatasetConfig(
        key="screenspot-pro",
        repo_id="likaixin/ScreenSpot-Pro",
        output_dirname="Screenspot-pro",
    ),
    "screenspot-v2": DatasetConfig(
        key="screenspot-v2",
        repo_id="likaixin/ScreenSpot-v2-variants",
        output_dirname="ScreenSpot-v2-variants",
    ),
}


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mirror_directory(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")
    for item in src.rglob("*"):
        relative = item.relative_to(src)
        target = dst / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _copy_files(files: Iterable[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        if file_path.exists() and file_path.is_file():
            shutil.copy2(file_path, destination / file_path.name)


def download_dataset(
    config: DatasetConfig,
    output_root: Path,
    cache_dir: Path | None,
    include_images: bool,
) -> bool:
    target_root = _ensure_directory(output_root / config.output_dirname)

    allow_patterns = ["annotations/**", "metadata.json", "dataset_infos.json", "README.md"]
    if include_images:
        allow_patterns.append("images/**")
    else:
        print(f"- {config.key}: skipping images (annotations only).")

    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=config.repo_id,
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=allow_patterns,
                local_dir=None,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        )
    except HfHubHTTPError as exc:
        print(f"✗ {config.key}: failed to download from Hugging Face. {exc}")
        print("  Make sure HF_TOKEN is set and you have accepted the dataset terms.")
        return False
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"✗ {config.key}: unexpected error during download. {exc}")
        return False

    copied_any = False
    for folder in ("annotations", "images"):
        if folder == "images" and not include_images:
            continue
        src_dir = snapshot_path / folder
        dst_dir = target_root / folder
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            _mirror_directory(src_dir, dst_dir)
            print(f"✓ {config.key}: copied {folder} → {dst_dir}")
            copied_any = True
        else:
            print(f"⚠ {config.key}: no '{folder}' directory found in snapshot.")

    extra_files = [
        snapshot_path / "metadata.json",
        snapshot_path / "dataset_infos.json",
        snapshot_path / "README.md",
    ]
    _copy_files(extra_files, target_root)

    if not copied_any:
        print(f"✗ {config.key}: nothing copied. Check dataset structure or access rights.")
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ScreenSpot datasets (annotations + optional images) from Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where datasets will be stored.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=["screenspot-pro", "screenspot-v2", "both"],
        default="both",
        help="Select which dataset to download.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Only download annotations (skip images).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_root = _ensure_directory(Path(args.output_dir).expanduser().resolve())
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_keys = (
        [args.dataset] if args.dataset != "both" else ["screenspot-pro", "screenspot-v2"]
    )

    include_images = not args.no_images

    results: dict[str, bool] = {}
    for key in dataset_keys:
        config = DATASETS[key]
        print(f"\nDownloading {config.repo_id} → {config.output_dirname}")
        ok = download_dataset(config, output_root, cache_dir, include_images)
        results[key] = ok

    failed = [key for key, ok in results.items() if not ok]
    if failed:
        print("\nDownload finished with errors:")
        for key in failed:
            print(f"- {key}")
        return 1

    print("\nAll requested datasets downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
