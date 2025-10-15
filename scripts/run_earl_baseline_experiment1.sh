#!/bin/bash
set -euo pipefail

export PYTHONPATH=.

usage() {
    cat <<'USAGE'
Usage: scripts/run_earl_baseline_experiment1.sh [options] [-- extra pipeline args...]

Options:
  --dry-run             Render prompts without invoking the model.
  --input-dir DIR       Directory containing variant navigation traces (defaults to newest run under experiment1_outputs/).
  --model-path PATH     Override the vLLM checkpoint path (required unless --dry-run).
  --output-parent DIR   Parent directory for EARL outputs (default: experiment1_outputs/earl_baseline).
  -h, --help            Show this help message.

Environment variables:
  PYTHON_BIN     Python interpreter to use (default: python)
  INPUT_DIR      Fallback input directory (overridden by --input-dir)
  MODEL_PATH     Default vLLM checkpoint (overridden by --model-path)
  OUTPUT_PARENT  Output parent directory (overridden by --output-parent)

Arguments after -- are forwarded to models.navigation.earl_pipeline.
USAGE
}

DRY_RUN=0
CLI_INPUT_DIR=""
CLI_MODEL_PATH=""
CLI_OUTPUT_PARENT=""
declare -a EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --input-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --input-dir" >&2; exit 1; }
            CLI_INPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            [[ $# -ge 2 ]] || { echo "Missing value for --model-path" >&2; exit 1; }
            CLI_MODEL_PATH="$2"
            shift 2
            ;;
        --output-parent)
            [[ $# -ge 2 ]] || { echo "Missing value for --output-parent" >&2; exit 1; }
            CLI_OUTPUT_PARENT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
input_dir="${CLI_INPUT_DIR:-${INPUT_DIR:-}}"
output_parent="${CLI_OUTPUT_PARENT:-${OUTPUT_PARENT:-experiment1_outputs/earl_baseline}}"
output_parent="${output_parent%/}"
output_parent_name="$(basename "${output_parent}")"

if [[ -z "${input_dir}" ]]; then
    input_dir="$(OUTPUT_PARENT_BASENAME="${output_parent_name}" python - <<'PY'
import os
import pathlib

base = pathlib.Path("experiment1_outputs")
skip = {name for name in {os.environ.get("OUTPUT_PARENT_BASENAME", ""), "earl"} if name}
latest = ""
if base.exists():
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name not in skip),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        latest = candidates[0].as_posix()
print(latest, end="")
PY
)"
fi

if [[ -z "${input_dir}" ]]; then
    echo "ERROR: Unable to determine input directory. Set INPUT_DIR or provide --input-dir." >&2
    exit 1
fi

if [[ ! -d "${input_dir}" ]]; then
    echo "ERROR: Input directory '${input_dir}' does not exist." >&2
    exit 1
fi

mapfile -t variant_ids < <(
    python - <<'PY'
from models.navigation.experiment1_variants import list_prompt_variants
for vid in list_prompt_variants():
    print(vid)
PY
)

if [[ ${#variant_ids[@]} -eq 0 ]]; then
    echo "ERROR: No experiment 1 prompt variants discovered; aborting." >&2
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="${output_parent}/${timestamp}"
mkdir -p "${output_dir}"

model_path="${CLI_MODEL_PATH:-${MODEL_PATH:-}}"
if [[ ${DRY_RUN} -eq 0 && -z "${model_path}" ]]; then
    echo "ERROR: Set MODEL_PATH or pass --model-path when not running with --dry-run." >&2
    exit 1
fi

echo "Using navigation traces from: ${input_dir}"
echo "Writing EARL baseline outputs to: ${output_dir}"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "Running in dry-run mode; rendered prompts will be saved alongside outputs."
fi

processed=0
skipped=0

for variant_id in "${variant_ids[@]}"; do
    trace_file="${input_dir}/${variant_id}.json"
    if [[ ! -f "${trace_file}" ]]; then
        echo "Skipping ${variant_id}: missing trace file ${trace_file}" >&2
        ((skipped++))
        continue
    fi

    output_file="${output_dir}/${variant_id}.belief_trace.jsonl"
    prompts_file="${output_dir}/${variant_id}.prompts.txt"

    echo "Processing variant ${variant_id} -> ${output_file}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        prompts_args=()
        prompts_flag_set=0
        for arg in "${EXTRA_ARGS[@]}"; do
            if [[ "${arg}" == "--prompts-out" ]]; then
                prompts_flag_set=1
                break
            fi
        done
        if [[ ${prompts_flag_set} -eq 0 ]]; then
            prompts_args=(--prompts-out "${prompts_file}")
        fi
        "${PYTHON_BIN}" -m models.navigation.earl_pipeline \
            --input "${trace_file}" \
            --output "${output_file}" \
            --dry-run \
            "${prompts_args[@]}" \
            "${EXTRA_ARGS[@]}"
    else
        "${PYTHON_BIN}" -m models.navigation.earl_pipeline \
            --input "${trace_file}" \
            --output "${output_file}" \
            --model-path "${model_path}" \
            "${EXTRA_ARGS[@]}"
    fi

    ((processed++))
done

if [[ ${processed} -eq 0 ]]; then
    echo "ERROR: No trace files matching experiment 1 variants were found in ${input_dir}." >&2
    exit 1
fi

echo "EARL baseline pipeline complete. Processed ${processed} variant(s)."
if [[ ${skipped} -gt 0 ]]; then
    echo "Skipped ${skipped} variant(s) without trace outputs." >&2
fi

echo "Results available under ${output_dir}"
