# Experiment 1 Pilot Runbook

This document captures the practical steps for bringing Experiment 1 online and validating that the autonomous prompt variants run end-to-end inside the `UI-Venus` repository.

## 1. Prerequisites
- Python 3.10+
- Dependencies from `requirements.txt`
- Access to a local model checkpoint (e.g. `Qwen/Qwen2.5-VL-72B-Instruct` or UI-Venus distilled checkpoints)
- AndroidWorld trace JSON (use `examples/trace/trace.json` for quick smoke testing)

Validate the environment after installation:
```bash
python validate_setup.py
```

## 2. Updated Runner Overview
`models/navigation/runner.py` now understands two execution modes:

| Mode | Description | Key Flags |
|------|-------------|-----------|
| `instructed` (default) | Original behavior that passes each task description into the prompt | `--mode instructed` |
| `autonomous` | Removes task descriptions and injects an Experiment 1 system prompt variant instead | `--mode autonomous --variant_id <id>` |

### Prompt Variant Loader
- Variants are parsed directly from `docs/experiment1_prompt_variations.md`.
- Use `--variant_id` to choose a prompt. Pass `default` (or omit `--variant_id`) to load `D3_P2_V2_H0`.
- Optional context values (e.g. app metadata) can be supplied via:
  - `--context /path/to/context.json`
  - `--context_kv key=value` (repeatable)

### Output Schema
Runner output is now a structured JSON object:
```json
{
  "run_summary": {
    "mode": "autonomous",
    "variant_id": "D3_P2_V2_H0",
    "input_file": "examples/trace/trace.json",
    "history_length": 0,
    "autonomous_context": {
      "...": "..."
    }
  },
  "traces": [
    {
      "trace_index": 0,
      "mode": "autonomous",
      "variant_id": "D3_P2_V2_H0",
      "inputs": [
        {"item_index": 0, "task": "launch calculator", "image_path": "..."},
        {"item_index": 1, "task": "launch calculator", "image_path": "..."}
      ],
      "ground_truth_task": "launch calculator",
      "steps": [
        {
          "think": "...",
          "action": "...",
          "belief_state": {
            "goal": "...",
            "confidence": 0.56,
            "evidence": "...",
            "plan": "...",
            "raw_text": "..."
          },
          "diagnostics": {
            "autonomous_mode": true,
            "variant_id": "D3_P2_V2_H0",
            "confidence_threshold": 0.5,
            "probe_budget": 2,
            "confidence_gate_passed": true,
            "confidence_margin": 0.06,
            "goal": "...",
            "evidence": "...",
            "plan": "...",
            "probe_count": 0
          }
        }
      ]
    }
  ]
}
```

## 3. Quickstart Commands

### 3.1 Instructed Baseline
```bash
python -m models.navigation.runner \
  --mode instructed \
  --model_path /root/models/uivenus-7B \
  --input_file examples/trace/trace.json \
  --output_file outputs/instructed_trace.json \
  --history_length 3
```

### 3.2 Autonomous Variant Sweep
Example: pilot run for three prompt variants.
```bash
for variant in D1_P1_V1_H0 D3_P2_V2_H0 D4_P2_V2_H1; do
  python -m models.navigation.runner \
    --mode autonomous \
    --variant_id "$variant" \
    --model_path /root/models/uivenus-7B \
    --input_file examples/trace/trace.json \
    --output_file "outputs/autonomous_${variant}.json" \
    --history_length 3
done
```

### 3.3 Providing Supplemental Context
```bash
python -m models.navigation.runner \
  --mode autonomous \
  --variant_id D3_P2_V2_H1_ENHANCED \
  --context_kv app_category=productivity \
  --context_kv user_idle_time="5 minutes" \
  --context ui_metadata.json \
  --model_path /root/models/uivenus-7B \
  --input_file examples/trace/trace.json \
  --output_file outputs/autonomous_enhanced.json
```

## 4. Pilot Evaluation Checklist
1. **Sanity-check belief extraction** – confirm `belief_state.goal` mirrors the agent's narrated intent.
2. **Probe budget tracking** – verify `diagnostics.probe_count` stays within each variant's limit.
3. **Confidence gating** – inspect `confidence_gate_passed` vs. the variant's threshold.
4. **Compare against baseline** – run the same trace with `--mode instructed` and measure delta in success or action counts.

## 5. Next Steps
- Integrate AndroidWorld scenario batches and log outputs under `experiments/experiment1/runs/`.
- Feed the generated JSON into the testing protocol scripts (to be authored) for metric aggregation.
- Identify top-performing variants for ACE initialization (Experiment 2).
