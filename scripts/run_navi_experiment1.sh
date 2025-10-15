#!/bin/bash
set -euo pipefail

export PYTHONPATH=.

model_path="${MODEL_PATH:-inclusionAI/UI-Venus-Navi-7B}"
input_file="${INPUT_FILE:-examples/trace/trace.json}"
context_file="${CONTEXT_FILE:-}"
max_pixels="${MAX_PIXELS:-937664}"
min_pixels="${MIN_PIXELS:-830000}"

timestamp="$(date +%Y%m%d-%H%M%S)"
output_parent="experiment1_outputs"
output_dir="${output_parent}/${timestamp}"

mkdir -p "${output_dir}"

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

echo "Discovered ${#variant_ids[@]} autonomous prompt variants."
echo "Saving outputs under ${output_dir}"

for variant_id in "${variant_ids[@]}"; do
	output_file="${output_dir}/${variant_id}.json"
	echo "Running variant ${variant_id} -> ${output_file}"

    variant_context_file="${context_file}"
    if [[ -z "${variant_context_file}" && "${variant_id}" == *_ENHANCED ]]; then
        default_context_file="experiment1_outputs/autonomous_context_enhanced.json"
        if [[ -f "${default_context_file}" ]]; then
            echo "Applying default enhanced context: ${default_context_file}"
            variant_context_file="${default_context_file}"
        else
            echo "Warning: expected enhanced context file ${default_context_file} not found."
        fi
    fi

    args=(
        python models/navigation/runner.py
        --mode autonomous
        --variant_id "${variant_id}"
        --model_path "${model_path}"
        --input_file "${input_file}"
        --output_file "${output_file}"
        --max_pixels="${max_pixels}"
        --min_pixels="${min_pixels}"
    )

    if [[ -n "${variant_context_file}" ]]; then
        args+=(--context "${variant_context_file}")
    fi

    "${args[@]}"
done
