# Repository Guidelines

This contributor guide helps new agents align with UI-Venus workflows and avoid common pitfalls.

## Project Structure & Module Organization
The repository separates evaluation, runtime, and datasets so you can swap in checkpoints or logs without touching core code.
- `models/grounding/` holds evaluation drivers such as `eval_screenspot_pro.py` for 7B/72B checkpoints.
- `models/navigation/` provides the agent runtime, VLLM adapters, and helper utilities; study `ui_venus_navi_agent.py` before editing flows.
- `scripts/` exposes shell entry points like `bash scripts/run_gd_7b.sh` and `run_navi_72b.sh`, already wired with dataset/log defaults.
- `tests/` contains smoke checks (`quick_start.py`, `test_gpu_setup.py`) and should house any new regression coverage.
- Bulk assets live under `ScreenSpot-v2-variants/`, `Screenspot-pro/`, `data/`, and illustrative JSON schemas stay in `examples/`; sample outputs reside in `venus_7b/`.

## Build, Test, and Development Commands
Set up once with `pip install -r requirements.txt` (Python 3.10+, CUDA-enabled PyTorch recommended). Validate environments via `python validate_setup.py`.
- `bash scripts/run_gd_7b.sh` / `run_gd_72b.sh`: launch grounding evaluations after adjusting dataset paths.
- `bash scripts/run_navi_7b.sh` / `run_navi_72b.sh`: replay navigation traces; ensure `log_path` points to a writable directory.
- `python tests/quick_start.py`: fast inference smoke test; compare outputs in `venus_7b/`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, snake_case functions, and PascalCase classes. Preserve type hints and dataclasses to expose agent state. Format Python via `black`, keep shell scripts POSIX-compliant, and prefer structured logging helpers in `models/.../utils.py` over `print`.

## Testing Guidelines
Name new tests `test_<feature>.py` and keep fixtures lightweight under `examples/`. Gate any heavy GPU runs behind env flags. Use `python tests/test_gpu_setup.py` to confirm device visibility and mixed precision before long jobs. Document required artifacts in docstrings so downstream automation can reproduce results.

## Commit & Pull Request Guidelines
Write concise, lower-case commit subjects (e.g., `data download scripts`) under 60 characters. Reference evaluation logs, dataset variants, or reproduction commands in bodies. Pull requests should cover problem, solution, verification commands, and attach screenshots or sample traces when UX outputs change. Double-check that large dataset paths remain ignored and secrets stay out of diffs.

## Security & Configuration Tips
Export `HF_HOME` and `HF_TOKEN` instead of hard-coding credentials. Never commit checkpoints, raw screenshots, or downloaded corpora; rely on documented data paths. When editing orchestration scripts, keep placeholder paths and note required overrides inline.
