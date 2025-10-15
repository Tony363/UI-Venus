# Experiment 1 Priority Replay (2025-10-15)

## Context
- Instrumented `models/navigation/runner.py` to stop trace playback once an autonomous variant issues `Finished(...)`, capturing the stop point, and recording `finished_event`, `stopped_on_finished`, and `skipped_items_after_finished` in the trace payload.
- Added fallback metadata injection in `VenusNaviAgent` so empty `Finished(content='')` calls are populated from autonomous context (e.g., filename/size/ID).
- Archived the original experiment outputs under `experiment1_outputs/20251015-014654/` before replaying variants.
- Introduced `experiment1_outputs/autonomous_context_enhanced.json`; `run_navi_experiment1.sh` now auto-loads it for `_ENHANCED` variants when no explicit `CONTEXT_FILE` is provided.

## Replay Summary
Only the variants that previously triggered `Finished` mid-trace were replayed by trimming their histories to the finish step and re-exporting logs with the new instrumentation.

| Variant | Directness | Persona | Verbosity | History | Steps | Finished? | Finished Step | Skipped Frames | Last Action |
|---|---|---|---|---|---|---|---|---|---|
| D2_P2_V2_H1 | light | balanced | detailed | with_history | 6 | Y | 5 | 1 | Finished(content='壁纸_3.jpg UNKNOWN_SIZE UNKNOWN_ID') |
| D2_P3_V1_H0 | light | proactive | minimal | no_history | 6 | Y | 5 | 1 | Finished(content='壁纸_3.jpg UNKNOWN_SIZE UNKNOWN_ID') |
| D4_P3_V1_H0 | explicit | proactive | minimal | no_history | 6 | Y | 5 | 1 | Finished(content='壁纸_3.jpg UNKNOWN_SIZE UNKNOWN_ID') |

## Notes & Follow-ups
- `skipped_items_after_finished = 1` indicates one frame per trace is now omitted after the `Finished` signal, matching the new short-circuit logic.
- The fallback metadata currently uses placeholders (`UNKNOWN_SIZE`, `UNKNOWN_ID`). Replace these with real values by updating `autonomous_context_enhanced.json` once the UI details are confirmed or OCR is integrated.
- Re-run remaining variants once model weights are available to regenerate complete logs under the new instrumentation.
