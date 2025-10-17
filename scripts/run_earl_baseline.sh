#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_earl.sh"

input_file=""
output_file=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            break
            ;;
        *)
            if [[ -z "${input_file}" ]]; then
                input_file="$1"
                shift
                continue
            elif [[ -z "${output_file}" ]]; then
                output_file="$1"
                shift
                continue
            else
                break
            fi
            ;;
    esac
done

cmd=("${RUN_SCRIPT}" --mode single)
if [[ -n "${input_file}" ]]; then
    cmd+=(--input-file "${input_file}")
fi
if [[ -n "${output_file}" ]]; then
    cmd+=(--output-file "${output_file}")
fi
exec "${cmd[@]}" "$@"
