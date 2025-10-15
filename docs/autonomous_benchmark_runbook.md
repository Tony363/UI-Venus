# UI-Venus Autonomous Benchmark Runbook

This runbook captures the steps required to reproduce the benchmarks from the UI-Venus technical report (arXiv:2508.10833) using refreshed autonomous system prompts. Follow the sections in order when bringing a new variant online.

## 1. Scope & Benchmarks

- Grounding: ScreenSpot-v2, ScreenSpot-Pro, UI-Vision, optional OSWorld-G and AgentCPM.
- Navigation: AndroidWorld, AndroidControl (low/high), GUI-Odyssey.
- Goal: execute the full benchmark suite with autonomous prompt variants and compare against instructed baselines.

## 2. Environment Preparation

1. Install dependencies and validate the GPU stack:

   ```bash
   pip install -r requirements.txt
   python validate_setup.py
   ```

2. Configure model access:
   - Export `MODEL_PATH` (e.g., `inclusionAI/UI-Venus-Navi-7B` or a local checkpoint).
   - Set `HF_HOME` and `HF_TOKEN` for Hugging Face downloads.

3. Verify dataset directories align with the repository layout:
   - `ScreenSpot-v2-variants/`, `Screenspot-pro/`, `ui-vision/`
   - `osworld_g/`, `CAGUI_grounding/`
   - `vis_androidworld/` (contains `UI-Venus-androidworld.zip` and viewers)
   - Any custom AndroidControl/GUI-Odyssey traces converted into JSON.

## 3. Integrate New Autonomous Prompts

1. Append each system prompt to `docs/experiment1_prompt_variations.md` as a `PROMPT_*` Python block. Maintain the naming scheme `D[1-4]_P[1-3]_V[1-2]_H[0-1]` (with optional suffix such as `_ENHANCED`).
2. Ensure required placeholders (for example `{previous_actions}` or `{ui_elements_summary}`) exist in the template. Enhanced prompts should declare `_ENHANCED` so `requires_ui_summary` is set automatically (`models/navigation/experiment1_variants.py`).
3. No code changes are needed; the loader executes the Markdown blocks at runtime and rebuilds the variant cache on demand.

## 4. Smoke-Test Prompts

Run a quick check on the sample trace to confirm formatting, parsing, and confidence gates:

```bash
python -m models.navigation.runner \
  --mode autonomous \
  --variant_id D3_P2_V2_H0 \
  --model_path "$MODEL_PATH" \
  --input_file examples/trace/trace.json \
  --output_file experiment1_outputs/smoke.json
```

The runner records prompt payloads and structured outputs in `experiment1_outputs/`. Inspect the log to verify `<think>`, `<action>`, and `<conclusion>` parsing plus belief-state diagnostics (`models/navigation/runner.py`).

## 5. Navigation Benchmarks

1. **Prepare trace files** per benchmark:
   - Convert AndroidWorld, AndroidControl, and GUI-Odyssey episodes into arrays of `{ "image_path": ..., "task": ... }`.
   - Optionally hide instructions in autonomous runs by keeping `"task"` only as ground truth metadata.
2. **Execute variant sweeps** with `scripts/run_navi_experiment1.sh`:

   ```bash
   MODEL_PATH=/path/to/uivenus-navi-7b \
   INPUT_FILE=/data/androidworld/autonomous_trace.json \
   OUTPUT_DIR=results/androidworld_autonomous \
   CONTEXT_FILE=metadata/context.json \
   bash scripts/run_navi_experiment1.sh
   ```

   - The script enumerates all variants and saves each run as `<variant_id>.json` under the chosen output directory.
   - Add `--context_kv key=value` pairs inside the script if certain prompts require additional context.
3. **Baseline comparison**: rerun `models/navigation/runner.py` with `--mode instructed` on the same traces to produce supervised logs for delta analysis.
4. **Visualization & QA**: launch the AndroidWorld viewer to inspect trajectories:

   ```bash
   python vis_androidworld/vis_androidworld_trace.py \
     --path results/androidworld_autonomous \
     --port 5050
   ```

   Open `http://localhost:5050` to review steps, reasoning, and actions.

## 6. Grounding Benchmarks

1. Update `scripts/run_gd_7b.sh` (and `run_gd_72b.sh`) with correct image/annotation paths and log locations.
2. Launch the full suite to mirror the paper metrics:

   ```bash
   bash scripts/run_gd_7b.sh
   ```

   - Uncomment the OSWorld-G or AgentCPM sections as needed.
   - Results accumulate in `venus_7b/*.json`; repeat with 72B checkpoints for high-capacity baselines.

## 7. Metric Aggregation

1. Parse navigation outputs using the testing protocol described in `docs/experiment1_prompt_testing_protocol.md`. Derive:
   - Task Success Rate, Goal Inference Accuracy, Action Efficiency, Recovery Rate.
   - Confidence calibration metrics when variants emit confidence hints.
2. Summarize variant performance by scenario and persona. Consider storing computed scores in `experiment1_outputs/metrics/*.json` for reproducibility.
3. For grounding, reuse evaluation scripts or notebooks that consume the generated logs and compute mean accuracy per UI type.

## 8. Reporting & Next Steps

1. Document configuration details (model commit, dataset hash, context fields) in the experiment folder.
2. Highlight top-performing variants and any threshold adjustments in `docs/experiment1_prompt_variations.md` or a dedicated changelog.
3. When filing a PR, include:
   - Commands executed (cover both navigation and grounding suites).
   - Pointers to output directories.
   - Observed metric deltas vs. instructed baselines.
4. Optional: automate sweeps and aggregation via CI or a notebook once the process is stable.

Following this playbook ensures your autonomous prompt updates are vetted end-to-end and benchmark numbers remain aligned with the UI-Venus technical report.
