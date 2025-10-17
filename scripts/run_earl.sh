#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}"
cd "${PROJECT_ROOT}"

usage() {
    cat <<'USAGE'
Usage: scripts/run_earl.sh [options] [-- extra pipeline args...]

Modes:
  single       Process a single navigation trace file (default when --input-file is set or no other hints provided).
  batch        Process every *.json trace under a directory.
  experiment   Process the curated Experiment 1 variant set, enforcing expected filenames.

Options:
  --mode {single|batch|experiment|auto}  Execution mode (default: auto infer).
  --dry-run                              Render prompts without invoking the model.
  --input-file FILE                      Trace JSON input for single mode (default: PROJECT_ROOT/saved_trace.json).
  --output-file FILE                     Belief trace output for single mode (default: PROJECT_ROOT/belief_trace.jsonl).
  --input-dir DIR                        Directory containing trace JSONs for batch/experiment modes.
  --output-parent DIR                    Parent directory to store run outputs (default per mode).
  --model-path PATH                      Override the vLLM checkpoint path (required unless --dry-run).
  --help, -h                             Show this help message.

Environment variables:
  PYTHON_BIN     Python interpreter to use (default: python)
  INPUT_FILE     Fallback for --input-file.
  OUTPUT_FILE    Fallback for --output-file.
  INPUT_DIR      Fallback for --input-dir.
  OUTPUT_PARENT  Fallback for --output-parent.
  MODEL_PATH     Default vLLM checkpoint path.

Arguments after -- are forwarded to models.navigation.earl_pipeline.
USAGE
}

MODE="auto"
DRY_RUN=0
CLI_INPUT_FILE=""
CLI_OUTPUT_FILE=""
CLI_INPUT_DIR=""
CLI_OUTPUT_PARENT=""
CLI_MODEL_PATH=""
declare -a EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || { echo "Missing value for --mode" >&2; exit 1; }
            MODE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --input-file)
            [[ $# -ge 2 ]] || { echo "Missing value for --input-file" >&2; exit 1; }
            CLI_INPUT_FILE="$2"
            shift 2
            ;;
        --output-file)
            [[ $# -ge 2 ]] || { echo "Missing value for --output-file" >&2; exit 1; }
            CLI_OUTPUT_FILE="$2"
            shift 2
            ;;
        --input-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --input-dir" >&2; exit 1; }
            CLI_INPUT_DIR="$2"
            shift 2
            ;;
        --output-parent)
            [[ $# -ge 2 ]] || { echo "Missing value for --output-parent" >&2; exit 1; }
            CLI_OUTPUT_PARENT="$2"
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
MODEL_PATH="${CLI_MODEL_PATH:-${MODEL_PATH:-}}"

case "${MODE}" in
    auto)
        if [[ -n "${CLI_INPUT_FILE}" || -n "${INPUT_FILE:-}" || -n "${CLI_OUTPUT_FILE}" || -n "${OUTPUT_FILE:-}" ]]; then
            MODE="single"
        elif [[ -n "${CLI_INPUT_DIR}" || -n "${INPUT_DIR:-}" ]]; then
            MODE="batch"
        else
            MODE="single"
        fi
        ;;
    single|batch|experiment)
        ;;
    *)
        echo "ERROR: Unsupported mode '${MODE}'. Expected single, batch, experiment, or auto." >&2
        exit 1
        ;;
esac

prompts_flag_present=0
for arg in "${EXTRA_ARGS[@]}"; do
    case "${arg}" in
        --prompts-out|--prompts-out=*)
            prompts_flag_present=1
            break
            ;;
    esac
done

ensure_model_path() {
    if [[ ${DRY_RUN} -eq 0 && -z "${MODEL_PATH}" ]]; then
        echo "ERROR: Set MODEL_PATH or pass --model-path when not running with --dry-run." >&2
        exit 1
    fi
}

discover_input_dir() {
    local skip_names_csv="$1"
    python - <<PY
import os
import pathlib

skip_names = {name for name in "${skip_names_csv}".split(",") if name}
base = pathlib.Path("experiment1_outputs")
latest = ""
if base.exists():
    candidates = sorted(
        (
            p for p in base.iterdir()
            if p.is_dir() and p.name not in skip_names
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        latest = candidates[0].as_posix()
print(latest, end="")
PY
}

run_single() {
    local input_file="${CLI_INPUT_FILE:-${INPUT_FILE:-${PROJECT_ROOT}/saved_trace.json}}"
    local output_file="${CLI_OUTPUT_FILE:-${OUTPUT_FILE:-${PROJECT_ROOT}/belief_trace.jsonl}}"

    if [[ ! -f "${input_file}" ]]; then
        echo "ERROR: Input file '${input_file}' does not exist." >&2
        exit 1
    fi

    ensure_model_path

    local -a cmd=("${PYTHON_BIN}" -m models.navigation.earl_pipeline
        --input "${input_file}"
        --output "${output_file}")

    if [[ ${DRY_RUN} -eq 1 ]]; then
        cmd+=(--dry-run)
        if [[ ${prompts_flag_present} -eq 0 ]]; then
            local prompts_default="${output_file}.prompts.txt"
            cmd+=(--prompts-out "${prompts_default}")
        fi
    else
        cmd+=(--model-path "${MODEL_PATH}")
    fi

    cmd+=("${EXTRA_ARGS[@]}")
    exec "${cmd[@]}"
}

run_batch_common() {
    local mode_label="$1"
    local enforce_variants="$2"
    local default_parent="$3"

    ensure_model_path

    local input_dir="${CLI_INPUT_DIR:-${INPUT_DIR:-}}"
    local output_parent="${CLI_OUTPUT_PARENT:-${OUTPUT_PARENT:-${default_parent}}}"

    if [[ -z "${input_dir}" ]]; then
        local skip_names="${output_parent##*/}"
        if [[ "${mode_label}" == "experiment" ]]; then
            skip_names="${skip_names},earl"
        fi
        input_dir="$(discover_input_dir "${skip_names}")"
    fi

    if [[ -z "${input_dir}" ]]; then
        echo "ERROR: Unable to determine input directory. Set INPUT_DIR or pass --input-dir." >&2
        exit 1
    fi

    if [[ ! -d "${input_dir}" ]]; then
        echo "ERROR: Input directory '${input_dir}' does not exist." >&2
        exit 1
    fi

    local timestamp
    timestamp="$(date +%Y%m%d-%H%M%S)"
    local output_parent_trimmed="${output_parent%/}"
    local output_dir="${output_parent_trimmed}/${timestamp}"
    mkdir -p "${output_dir}"

    echo "Using navigation traces from: ${input_dir}"
    echo "Writing ${mode_label} outputs to: ${output_dir}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "Running in dry-run mode; rendered prompts will be saved alongside outputs."
    fi

    local processed=0
    local skipped=0

    if [[ "${enforce_variants}" == "1" ]]; then
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

        for variant_id in "${variant_ids[@]}"; do
            local trace_file="${input_dir}/${variant_id}.json"
            if [[ ! -f "${trace_file}" ]]; then
                echo "Skipping ${variant_id}: missing trace file ${trace_file}" >&2
                ((skipped++))
                continue
            fi
            process_trace "${trace_file}" "${output_dir}" "${variant_id}"
            ((processed++))
        done
    else
        mapfile -t trace_files < <(find "${input_dir}" -maxdepth 1 -type f -name '*.json' | sort)
        if [[ ${#trace_files[@]} -eq 0 ]]; then
            echo "ERROR: No .json trace files found under ${input_dir}" >&2
            exit 1
        fi
        for trace_file in "${trace_files[@]}"; do
            local variant_name
            variant_name="$(basename "${trace_file}" .json)"
            process_trace "${trace_file}" "${output_dir}" "${variant_name}"
            ((processed++))
        done
    fi

    if [[ ${processed} -eq 0 ]]; then
        echo "ERROR: No trace files were processed." >&2
        exit 1
    fi

    echo "EARL pipeline complete. Processed ${processed} file(s)."
    if [[ ${skipped} -gt 0 ]]; then
        echo "Skipped ${skipped} missing variant(s)." >&2
    fi
    echo "Results available under ${output_dir}"
}

process_trace() {
    local trace_file="$1"
    local output_dir="$2"
    local variant_name="$3"

    local output_file="${output_dir}/${variant_name}.belief_trace.jsonl"
    local prompts_file="${output_dir}/${variant_name}.prompts.txt"

    echo "Processing ${variant_name}"

    local -a cmd=("${PYTHON_BIN}" -m models.navigation.earl_pipeline
        --input "${trace_file}"
        --output "${output_file}")

    if [[ ${DRY_RUN} -eq 1 ]]; then
        cmd+=(--dry-run)
        if [[ ${prompts_flag_present} -eq 0 ]]; then
            cmd+=(--prompts-out "${prompts_file}")
        fi
    else
        cmd+=(--model-path "${MODEL_PATH}")
    fi

    cmd+=("${EXTRA_ARGS[@]}")
    "${cmd[@]}"
}

case "${MODE}" in
    single)
        run_single
        ;;
    batch)
        run_batch_common "batch" "0" "experiment1_outputs/earl"
        ;;
    experiment)
        run_batch_common "experiment" "1" "experiment1_outputs/earl_baseline"
        ;;
    *)
        echo "ERROR: Unexpected mode '${MODE}'." >&2
        exit 1
        ;;
esac
