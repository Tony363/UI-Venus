#!/bin/bash
set -euo pipefail

export PYTHONPATH=.

model_path="${MODEL_PATH:-inclusionAI/UI-Venus-Navi-7B}"
image_dir="${IMAGE_DIR:-Screenspot-pro/images/common_mac}"
output_root="${OUTPUT_ROOT:-experiment1_outputs/mac_outputs}"
context_file="${CONTEXT_FILE:-}"
max_pixels="${MAX_PIXELS:-937664}"
min_pixels="${MIN_PIXELS:-830000}"

if [[ ! -d "${image_dir}" ]]; then
    echo "Image directory '${image_dir}' does not exist; aborting." >&2
    exit 1
fi

mkdir -p "${output_root}"

mapfile -t variant_ids < <(
    python - <<'PY'
from models.navigation.experiment1_variants import list_prompt_variants
for vid in list_prompt_variants():
    print(vid)
PY
)

if [[ ${#variant_ids[@]} -eq 0 ]]; then
    echo "No experiment 1 prompt variants discovered; aborting." >&2
    exit 1
fi

mapfile -t grouped_traces < <(
    python - "$image_dir" <<'PY'
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def parse_datetime(path: Path) -> datetime:
    stem = path.stem.replace("\u202f", " ").replace("\u00a0", " ").strip()
    lower_stem = stem.lower()

    prefix = "screenshot "
    if lower_stem.startswith(prefix):
        remainder = stem[len(prefix) :]
        if " at " in remainder:
            date_part, time_part = remainder.split(" at ", 1)
            normalized_time = time_part.replace(".", ":").strip()
            try:
                return datetime.strptime(
                    f"{date_part.strip()} {normalized_time}",
                    "%Y-%m-%d %I:%M:%S %p",
                )
            except ValueError as exc:
                print(
                    f"Warning: failed to parse datetime from '{path.name}' "
                    f"with Screenshot format ({exc}).",
                    file=sys.stderr,
                )
        else:
            print(
                f"Warning: expected ' at ' separator in '{path.name}'.",
                file=sys.stderr,
            )
    match = re.match(
        r"^screenshot_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$", lower_stem
    )
    if match:
        date_part, time_part = match.groups()
        try:
            return datetime.strptime(
                f"{date_part} {time_part.replace('-', ':')}",
                "%Y-%m-%d %H:%M:%S",
            )
        except ValueError as exc:
            print(
                f"Warning: failed to parse datetime from '{path.name}' "
                f"with snake case format ({exc}).",
                file=sys.stderr,
            )

    fallback = datetime.fromtimestamp(path.stat().st_mtime)
    print(
        f"Warning: using filesystem timestamp for '{path.name}' "
        f"({fallback.isoformat()}).",
        file=sys.stderr,
    )
    return fallback


def main() -> None:
    image_dir = Path(sys.argv[1])
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory '{image_dir}' is not a directory.")

    grouped: Dict[str, List[tuple[datetime, Path]]] = {}
    for path in image_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed_exts:
            continue
        dt = parse_datetime(path)
        grouped.setdefault(dt.date().isoformat(), []).append((dt, path.resolve()))

    if not grouped:
        return

    for date_key in sorted(grouped.keys()):
        ordered = sorted(
            grouped[date_key],
            key=lambda pair: (pair[0], pair[1].name.lower()),
        )
        images = [str(path) for _, path in ordered]
        print(json.dumps({"date": date_key, "images": images}, ensure_ascii=False))


if __name__ == "__main__":
    main()
PY
)

if [[ ${#grouped_traces[@]} -eq 0 ]]; then
    echo "No image files found in '${image_dir}'; aborting." >&2
    exit 1
fi

echo "Discovered ${#variant_ids[@]} autonomous prompt variants."
total_image_count=0
for group_json in "${grouped_traces[@]}"; do
    group_size="$(python - "$group_json" <<'PY'
import json
import sys

group = json.loads(sys.argv[1])
print(len(group["images"]))
PY
)"
    total_image_count=$((total_image_count + group_size))
done

echo "Discovered ${#grouped_traces[@]} date-based chat histories covering ${total_image_count} images under '${image_dir}'."

context_args=()
if [[ -n "${context_file}" ]]; then
    context_args+=(--context "${context_file}")
fi

for group_json in "${grouped_traces[@]}"; do
    readarray -t group_info < <(
        python - "$group_json" <<'PY'
import json
import sys
from pathlib import Path

group = json.loads(sys.argv[1])
date_key = group["date"]
images = group["images"]

print(date_key)
print(len(images))
print("\n".join(images))
PY
    )

    group_date="${group_info[0]}"
    group_size="${group_info[1]}"
    echo "Processing ${group_size} images for date '${group_date}'..."

    trace_file="$(mktemp -p "${TMPDIR:-/tmp}" "navi_common_mac_trace.XXXXXX.json")"
    python - "${group_json}" "$trace_file" <<'PY'
import json
import sys
from pathlib import Path

group = json.loads(sys.argv[1])
trace_path = Path(sys.argv[2])

traces = []
for index, image in enumerate(group["images"], start=1):
    image_path = Path(image)
    traces.append(
        [
            {
                "task": (
                    f"Autonomous evaluation for {group['date']} "
                    f"image {index}: '{image_path.name}'."
                ),
                "image_path": str(image_path),
            }
        ]
    )

trace_path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
PY

    image_output_dir="${output_root}/${group_date}"
    mkdir -p "${image_output_dir}"

    for variant_id in "${variant_ids[@]}"; do
        output_file="${image_output_dir}/${variant_id}.json"
        echo "  Running variant '${variant_id}' -> ${output_file}"

        args=(
            python models/navigation/runner.py
            --mode autonomous
            --variant_id "${variant_id}"
            --model_path "${model_path}"
            --input_file "${trace_file}"
            --output_file "${output_file}"
            --max_pixels="${max_pixels}"
            --min_pixels="${min_pixels}"
        )

        if [[ ${#context_args[@]} -gt 0 ]]; then
            args+=("${context_args[@]}")
        fi

        "${args[@]}"
    done

    rm -f "${trace_file}"
done

echo "Completed autonomous experiments for ${#grouped_traces[@]} grouped chat histories (${total_image_count} total images)."
