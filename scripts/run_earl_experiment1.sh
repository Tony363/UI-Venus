#!/bin/bash
set -euo pipefail

export PYTHONPATH=.

usage() {
    cat <<'USAGE'
Usage: scripts/run_earl_experiment1.sh [--dry-run] [--input-dir DIR] [--model-path PATH] [extra pipeline args...]

Runs the EARL baseline pipeline over each navigation trace JSON in DIR (defaults to the most
recent directory under experiment1_outputs/ that is not the EARL results folder). Results are
written to experiment1_outputs/earl/<timestamp>/.

Environment variables:
  PYTHON_BIN   Python interpreter to use (default: python)
  MODEL_PATH   Default text-only model for vLLM (required unless --dry-run or --model-path provided)
  INPUT_DIR    Directory containing navigation traces (overridden by --input-dir)
USAGE
}

DRY_RUN=0
CLI_INPUT_DIR=""
CLI_MODEL_PATH=""
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

if [[ -z "${input_dir}" ]]; then
    input_dir="$(python - <<'PY'
import pathlib
base = pathlib.Path("experiment1_outputs")
latest = ""
if base.exists():
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name != "earl"),
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
    echo "ERROR: Unable to determine input directory. Set INPUT_DIR or pass --input-dir." >&2
    exit 1
fi

if [[ ! -d "${input_dir}" ]]; then
    echo "ERROR: Input directory '${input_dir}' does not exist." >&2
    exit 1
fi

mapfile -t trace_files < <(find "${input_dir}" -maxdepth 1 -type f -name '*.json' | sort)
if [[ ${#trace_files[@]} -eq 0 ]]; then
    echo "ERROR: No .json trace files found under ${input_dir}" >&2
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
output_root="experiment1_outputs/earl"
output_dir="${output_root}/${timestamp}"
mkdir -p "${output_dir}"

model_path="${CLI_MODEL_PATH:-${MODEL_PATH:-}}"
if [[ ${DRY_RUN} -eq 0 && -z "${model_path}" ]]; then
    echo "ERROR: Set MODEL_PATH or provide --model-path when not running in --dry-run mode." >&2
    exit 1
fi

echo "Using input directory: ${input_dir}"
echo "Writing EARL outputs to: ${output_dir}"

for trace_file in "${trace_files[@]}"; do
    variant_name="$(basename "${trace_file}" .json)"
    output_file="${output_dir}/${variant_name}.belief_trace.jsonl"
    echo "Processing ${variant_name}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        prompts_out="${output_dir}/${variant_name}.prompts.txt"
        "${PYTHON_BIN}" -m models.navigation.earl_pipeline \
            --input "${trace_file}" \
            --output "${output_file}" \
            --dry-run \
            --prompts-out "${prompts_out}" \
            "${EXTRA_ARGS[@]}"
    else
        "${PYTHON_BIN}" -m models.navigation.earl_pipeline \
            --input "${trace_file}" \
            --output "${output_file}" \
            --model-path "${model_path}" \
            "${EXTRA_ARGS[@]}"
    fi
done

echo "EARL baseline processing complete. Results stored in ${output_dir}."
