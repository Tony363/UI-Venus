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

mapfile -t image_paths < <(
    python - "$image_dir" <<'PY'
from pathlib import Path
import sys

image_dir = Path(sys.argv[1])
allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

if not image_dir.is_dir():
    raise SystemExit(f"Image directory '{image_dir}' is not a directory.")

paths = sorted(
    str(path.resolve())
    for path in image_dir.iterdir()
    if path.is_file() and path.suffix.lower() in allowed_exts
)

for image_path in paths:
    print(image_path)
PY
)

if [[ ${#image_paths[@]} -eq 0 ]]; then
    echo "No image files found in '${image_dir}'; aborting." >&2
    exit 1
fi

echo "Discovered ${#variant_ids[@]} autonomous prompt variants."
echo "Discovered ${#image_paths[@]} images under '${image_dir}'."

context_args=()
if [[ -n "${context_file}" ]]; then
    context_args+=(--context "${context_file}")
fi

for image_path in "${image_paths[@]}"; do
    image_basename="$(basename "${image_path}")"
    image_stem="${image_basename%.*}"
    echo "Processing image '${image_basename}'..."

    trace_file="$(mktemp -p "${TMPDIR:-/tmp}" "navi_common_mac_trace.XXXXXX.json")"
    python - "$image_path" "$trace_file" <<'PY'
import json
import sys
from pathlib import Path

image_path = Path(sys.argv[1])
trace_path = Path(sys.argv[2])

trace = [
    [
        {
            "task": f"Autonomous evaluation for screenshot '{image_path.name}'.",
            "image_path": str(image_path),
        }
    ]
]

trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
PY

    image_output_dir="${output_root}/${image_stem}"
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

echo "Completed autonomous experiments for ${#image_paths[@]} images."
