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
class FolderMapping:
    source: str
    target: str
    requires_images: bool = False


DEFAULT_FOLDER_MAPPINGS: tuple[FolderMapping, ...] = (
    FolderMapping("annotations", "annotations"),
    FolderMapping("images", "images", requires_images=True),
)

DEFAULT_EXTRA_FILES: tuple[str, ...] = (
    "metadata.json",
    "dataset_infos.json",
    "README.md",
)


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    repo_id: str
    output_dirname: str
    folder_mappings: tuple[FolderMapping, ...] = DEFAULT_FOLDER_MAPPINGS
    extra_files: tuple[str, ...] = DEFAULT_EXTRA_FILES
    allow_patterns: tuple[str, ...] | None = None


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
    "cagui-grounding": DatasetConfig(
        key="cagui-grounding",
        repo_id="openbmb/CAGUI",
        output_dirname="CAGUI_grounding",
        folder_mappings=(
            FolderMapping("CAGUI_grounding/code", "code"),
            FolderMapping("CAGUI_grounding/images", "images", requires_images=True),
        ),
        extra_files=("README.md",),
    ),
    "ui-vision": DatasetConfig(
        key="ui-vision",
        repo_id="ServiceNow/ui-vision",
        output_dirname="ui-vision",
    ),
}

DATASET_CHOICES: tuple[str, ...] = tuple(DATASETS.keys()) + ("both", "all")


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

    allow_patterns: set[str] = set(config.allow_patterns or ())
    if not include_images and any(mapping.requires_images for mapping in config.folder_mappings):
        print(f"- {config.key}: skipping image payloads (text-only download).")

    for mapping in config.folder_mappings:
        if mapping.requires_images and not include_images:
            continue
        allow_patterns.add(f"{mapping.source}/**")

    for extra in config.extra_files:
        allow_patterns.add(extra)

    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=config.repo_id,
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=sorted(allow_patterns),
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
    for mapping in config.folder_mappings:
        if mapping.requires_images and not include_images:
            continue
        src_dir = snapshot_path / mapping.source
        dst_dir = target_root / mapping.target
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            _mirror_directory(src_dir, dst_dir)
            print(f"✓ {config.key}: copied {mapping.source} → {dst_dir}")
            copied_any = True
        else:
            print(f"⚠ {config.key}: snapshot missing '{mapping.source}' directory.")

    extra_files = [
        snapshot_path / extra for extra in config.extra_files
    ]
    _copy_files(extra_files, target_root)

    if not copied_any:
        print(f"✗ {config.key}: nothing copied. Check dataset structure or access rights.")
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ScreenSpot, CAGUI, and UI-Vision datasets (annotations/code plus optional images) from Hugging Face."
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
        choices=DATASET_CHOICES,
        default="both",
        help="Select which dataset(s) to download.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Only download annotations/code (skip images).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_root = _ensure_directory(Path(args.output_dir).expanduser().resolve())
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_keys = (
        list(DATASETS.keys())
        if args.dataset == "all"
        else ["screenspot-pro", "screenspot-v2"]
        if args.dataset == "both"
        else [args.dataset]
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
