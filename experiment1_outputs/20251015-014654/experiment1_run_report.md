# Autonomous Navigation Experiment 1 Report

## Run Context
- Command: `bash scripts/run_navi_experiment1.sh` with default environment variables (`MODEL_PATH=inclusionAI/UI-Venus-Navi-7B`, `INPUT_FILE=examples/trace/trace.json`, `OUTPUT_DIR=experiment1_outputs`, no `CONTEXT_FILE`).
- Scenario: single autonomous trace labelled “在夸克浏览器中，查看云图片中的"壁纸_3.jpg"的详细信息，并记住文件名，文件大小和文件ID，并以空格分隔符分隔，输出” (Quark browser: open wallpaper_3.jpg details and report name, size, and ID separated by spaces).
- Prompt coverage: 14 autonomous prompt variants discovered via `models.navigation.experiment1_variants.list_prompt_variants()`, one rollout per variant.
- All runs completed seven UI steps (matching the seven screenshots in the trace) and produced JSON logs under `experiment1_outputs/`.

## Variant Outcomes
| Variant | Directness | Persona | Verbosity | History | UI Summary | Conf. | Probes | Steps | Finished | Last Action | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CONFIDENCE_ADAPTIVE | custom | custom | custom | with_history | N | 0.5 | 2 | 7 | N | Wait() | Implements dynamic confidence guidance inside the prompt. |
| CONTEXT_HEAVY | custom | custom | custom | with_history | N | 0.5 | 2 | 7 | N | Wait() | Supports optional environmental context placeholders. |
| D1_P1_V1_H0 | implicit | cautious | minimal | no_history | N | 0.7 | 1 | 7 | N | Click(box=(352, 1019)) |  |
| D1_P2_V1_H1 | implicit | balanced | minimal | with_history | N | 0.5 | 2 | 7 | N | Click(box=(352, 1019)) |  |
| D2_P1_V2_H0 | light | cautious | detailed | no_history | N | 0.7 | 1 | 7 | N | Click(box=(352, 1019)) |  |
| D2_P2_V2_H1 | light | balanced | detailed | with_history | N | 0.5 | 2 | 7 | Y | Click(box=(352, 1019)) |  |
| D2_P3_V1_H0 | light | proactive | minimal | no_history | N | 0.35 | 3 | 7 | Y | Click(box=(352, 1019)) |  |
| D3_P1_V2_H1 | moderate | cautious | detailed | with_history | N | 0.7 | 1 | 7 | N | Click(box=(352, 1019)) |  |
| D3_P2_V2_H0 | moderate | balanced | detailed | no_history | N | 0.5 | 2 | 7 | N | Click(box=(352, 1019)) |  |
| D3_P2_V2_H1_ENHANCED | moderate | balanced | detailed | with_history | Y | 0.5 | 2 | 7 | N | Click(box=(352, 1019)) | Enhanced template expects a UI summary input. |
| D3_P3_V1_H1 | moderate | proactive | minimal | with_history | N | 0.35 | 3 | 7 | N | Click(box=(359, 1019)) |  |
| D4_P1_V2_H0 | explicit | cautious | detailed | no_history | N | 0.7 | 1 | 7 | N | Click(box=(359, 1019)) | Final belief mis-identified the goal as “Download the selected image.” |
| D4_P2_V2_H1 | explicit | balanced | detailed | with_history | N | 0.5 | 2 | 7 | N | Click(box=(352, 1019)) |  |
| D4_P3_V1_H0 | explicit | proactive | minimal | no_history | N | 0.35 | 3 | 7 | Y | Finished(content='') | Declared completion without emitting the requested file metadata. |

## Observations
- None of the variants produced the requested filename, size, and file ID; most concluded by clicking the on-screen “取消” control instead of extracting data.
- CONFIDENCE_ADAPTIVE and CONTEXT_HEAVY advocated canceling in their reasoning but executed `Wait()` on the final frame, indicating action selection drift.
- D2_P2_V2_H1 and D2_P3_V1_H0 invoked `Finished(content='')` mid-run yet continued with a follow-up click, suggesting the loop processes remaining frames even after a finish signal.
- D4_P1_V2_H0’s higher confidence gate (0.7) led it to assert a download-oriented goal that conflicts with the ground-truth instruction, highlighting potential overconfidence at high directness levels.
- The ENHANCED template ran without a UI summary context file, so its behavior mirrors the non-enhanced baseline, implying the extra placeholder remains unused.

## Next Steps
1. Replay priority variants with instrumentation to capture whether `Finished` calls short-circuit the runner or require explicit break conditions.
2. Provide the expected UI summary input when testing `D3_P2_V2_H1_ENHANCED` to validate the additional conditioning.
3. Adjust prompts or post-processing so successful runs emit the requested metadata payload instead of neutral conclusions before integrating into evaluation pipelines.
