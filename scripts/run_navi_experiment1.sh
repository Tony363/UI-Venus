#!/bin/bash
set -euo pipefail

export PYTHONPATH=.

model_path="${MODEL_PATH:-inclusionAI/UI-Venus-Navi-7B}"
input_file="${INPUT_FILE:-examples/trace/trace.json}"
output_file="${OUTPUT_FILE:-./saved_trace.json}"
variant_id="${1:-D3_P2_V2_H0}"
context_file="${CONTEXT_FILE:-}"

args=(
    python models/navigation/runner.py
    --mode autonomous
    --variant_id "${variant_id}"
    --model_path "${model_path}"
    --input_file "${input_file}"
    --output_file "${output_file}"
    --max_pixels=937664
    --min_pixels=830000
)

if [[ -n "${context_file}" ]]; then
    args+=(--context "${context_file}")
fi

"${args[@]}"
