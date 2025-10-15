# Repository Guidelines

## Project Structure & Module Organization
- `models/grounding/` – evaluation drivers (e.g., `eval_screenspot_pro.py`) for grounding checkpoints.  
- `models/navigation/` – runtime agents, VLLM adapters, and helpers; start with `ui_venus_navi_agent.py` before editing flows.  
- `scripts/` – executable entry points such as `bash scripts/run_gd_7b.sh` and `bash scripts/run_navi_72b.sh`, pre-wired with dataset/log defaults.  
- `tests/` – smoke checks (`quick_start.py`, `test_gpu_setup.py`) and the home for new regression coverage.  
- Assets and examples: `ScreenSpot-v2-variants/`, `Screenspot-pro/`, `data/`, `venus_7b/`, and `examples/`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` – one-time environment setup (Python 3.10+, CUDA-enabled PyTorch recommended).  
- `python validate_setup.py` – sanity-check local dependencies and device visibility.  
- `bash scripts/run_gd_7b.sh` / `bash scripts/run_gd_72b.sh` – launch grounding evaluations after adjusting dataset paths.  
- `bash scripts/run_navi_7b.sh` / `bash scripts/run_navi_72b.sh` – replay navigation traces; ensure `log_path` is writable.  
- `python tests/quick_start.py` – fast inference smoke test; compare outputs in `venus_7b/`.

## Coding Style & Naming Conventions
- Python: PEP 8, four-space indentation, snake_case functions, PascalCase classes, keep type hints and dataclasses.  
- Format with `black`; prefer structured logging helpers in `models/.../utils.py` over `print`.  
- Shell scripts stay POSIX-compliant; document required environment variables inline.

## Testing Guidelines
- Name new tests `test_<feature>.py`; keep fixtures lightweight under `examples/`.  
- Gate GPU-heavy runs behind env flags; run `python tests/test_gpu_setup.py` before long jobs.  
- Document required artifacts and reproduction steps in docstrings.

## Commit & Pull Request Guidelines
- Commit subjects: concise, lower case, ≤60 characters (e.g., `data download scripts`).  
- Reference evaluation logs, dataset variants, or reproduction commands in commit bodies.  
- PRs should state problem, solution, verification commands, and attach screenshots or sample traces when UX outputs change.  
- Double-check that large dataset paths remain ignored and secrets do not leak into diffs.

## Security & Configuration Tips
- Export `HF_HOME` and `HF_TOKEN`; never hard-code credentials.  
- Do not commit checkpoints, raw screenshots, or downloaded corpora—rely on documented data paths.  
- When editing orchestration scripts, keep placeholder paths and note required overrides inline.
