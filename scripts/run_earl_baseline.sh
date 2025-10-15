#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_FILE="${1:-${PROJECT_ROOT}/saved_trace.json}"
OUTPUT_FILE="${2:-${PROJECT_ROOT}/belief_trace.jsonl}"

if [[ "${3:-}" == "--dry-run" ]]; then
    shift 3 || true
    exec "${PYTHON_BIN}" -m models.navigation.earl_pipeline \
        --input "${INPUT_FILE}" \
        --output "${OUTPUT_FILE}" \
        --dry-run "$@"
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "ERROR: Set MODEL_PATH to a local vLLM-compatible checkpoint or run with --dry-run." >&2
    exit 1
fi

exec "${PYTHON_BIN}" -m models.navigation.earl_pipeline \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_FILE}" \
    --model-path "${MODEL_PATH}" \
    "$@"
