# autonomous_api.py Module Guide

This document explains how `autonomous_api.py` wires together the UI-Venus autonomous navigation pipeline, the modules it depends on, and how to run the FastAPI server that hosts it.

## Module Responsibilities
- Exposes a FastAPI application that brokers autonomous navigation jobs for the Venus agent.
- Validates incoming job requests, resolves the correct Experiment 1 prompt variant, and kicks off background execution.
- Tracks job lifecycle state in memory so clients can poll for completion and inspect results.
- Normalises runtime configuration via environment variables and the shared `ModelConfig` structure.

## Imports and Dependencies

### Standard Library
- `asyncio` orchestrates background execution and provides the shared `Lock` guarding the job store.
- `logging`, `os`, `uuid`, `datetime`, `timezone`, and `Path` handle instrumentation, environment introspection, identifiers, timestamps, and filesystem access.
- `dataclasses` (`dataclass`, `asdict`, `replace`) shapes internal job records and prepares belief state snapshots for response payloads.
- `Enum` and core typing helpers (`Dict`, `Optional`, `Any`) describe job status semantics and type annotations.

### Third-Party Packages
- `fastapi.FastAPI` and `fastapi.HTTPException` expose the HTTP interface and standard error responses.
- `pydantic.BaseModel`, `Field`, and validators define request payloads and enforce prompt/image constraints.

### Internal Modules
- `models.navigation.experiment1_variants` supplies the prompt variant registry (`get_prompt_variant`) and its error types.
- `models.navigation.runner` provides `ModelConfig` default values plus the shared `setup_logger`.
- `models.navigation.ui_venus_navi_agent.VenusNaviAgent` executes the actual multimodal navigation inference loop.

## Prompt Asset Resolution
- Prompt files live under the module’s sibling `system_prompts/` directory; `PROMPT_ROOT` resolves that folder relative to `autonomous_api.py`.
- Incoming prompt names must start with the `PROMPT_` prefix. `_resolve_prompt` appends `.txt` automatically, verifies the file exists, and exposes the variant identifier (the portion after the prefix) for downstream lookups.
- Sanitisation is applied twice: the request validator rejects names containing path separators or `..`, and `_resolve_prompt` checks the resolved path stays inside `system_prompts/`. Any violation returns HTTP 400 with a curated list of available prompt files.
- When a prompt file is missing, the API responds with HTTP 400 and includes the list of available prompt filenames.

## Request and Job Models
- `AutonomousStartRequest` (Pydantic) captures five fields: `prompt_name`, `image_path`, optional `context` overrides, `history_length`, and `include_screenshot`. Custom validators ensure prompt names retain the `PROMPT_` prefix, remain filename-only values, and that `image_path` is non-empty.
- `JobStatus` enumerates lifecycle phases (`queued`, `running`, `succeeded`, `failed`) for client-friendly status checks.
- `AutonomousJob` is a dataclass storing immutable request data plus mutable runtime fields (timestamps, result payload, error string).
- `AutonomousJobStore` keeps jobs in memory using an `asyncio.Lock` to guarantee safe concurrent access. Jobs are addressed by UUIDs generated on creation.

## Background Execution Model
- `start_autonomous_run` creates a job and immediately schedules `_execute_job` with `asyncio.create_task`, decoupling the HTTP response from the long-running inference call.
- `_execute_job` updates the job status to `running`, executes `_run_autonomous_inference` in a background thread (`asyncio.to_thread`), and persists success or failure details on completion.
- `_serialize_job` composes the response payload for `GET` requests, optionally including the inference result and any raised error.
- Because storage is in-memory only, restarting the server clears active and historical jobs; use an external queue or persistent cache if long-term tracking is required.

## Model Configuration and Environment Variables
- `_load_model_config` mirrors CLI flags from `models/navigation/runner.py`. It reads environment variables, falling back to `ModelConfig` defaults when unset or invalid.
- Supported overrides (all optional) include:
  - `UI_VENUS_MODEL_PATH`
  - `UI_VENUS_TP`
  - `UI_VENUS_GPU_MEM_UTIL`
  - `UI_VENUS_MAX_TOKENS`
  - `UI_VENUS_MAX_PIXELS`
  - `UI_VENUS_MIN_PIXELS`
  - `UI_VENUS_MAX_MODEL_LEN`
  - `UI_VENUS_MAX_NUM_SEQS`
  - `UI_VENUS_TEMPERATURE`
  - `UI_VENUS_TOP_P`
  - `UI_VENUS_TOP_K`
  - `UI_VENUS_SAMPLES`
- Invalid numeric values trigger a warning and revert to the corresponding `ModelConfig` defaults (for example, the default `model_path` is `Qwen/Qwen2.5-VL-72B-Instruct`, tensor parallel size `4`, temperature `0.0`).

## Autonomous Inference Pipeline
`_run_autonomous_inference` performs the heavy lifting:
- Validates that the referenced screenshot exists.
- Builds the `ModelConfig` and a scoped logger dedicated to the job.
- Loads the Experiment 1 prompt variant, substituting the prompt template contents read from disk.
- Instantiates `VenusNaviAgent` with the resolved configuration, history length, optional autonomous context, and prompt variant parameters (confidence threshold, probe budget).
- Executes a single `agent.step` call, dumps the agent history (optionally with base64-encoded screenshot), and exports the belief trace (converted via `asdict` for JSON).
- Returns a result object containing agent status, action JSON, history, belief trace, and prompt metadata. Any exception bubbles up to `_execute_job`, marking the job as `failed`.

## API Endpoints
- `POST /autonomous/runs`
  - Validates the request, normalises the prompt context dictionary to strings, ensures the image file exists, enqueues the job, and returns metadata (`job_id`, `queued_at`, `variant_id`, `prompt_name`).
  - Typical client flow: submit the prompt name (for example `PROMPT_AUTONOMOUS_DEFAULT`), an absolute or repo-relative screenshot path, and optional context overrides.

- `GET /autonomous/runs/{job_id}`
  - Fetches job state and returns the serialized payload. Missing IDs produce HTTP 404.
  - When a job succeeds, the payload includes the agent history and the final action JSON; failures expose an error message.

## Logging and Error Handling
- The module-level logger (`ui_venus.autonomous_api`) is seeded via `setup_logger` from `models.navigation.runner`, mirrors output to `logs/autonomous_api.log`, and retains console streaming.
- Each job emits through child loggers (`ui_venus.autonomous_api.job.<uuid>`), ensuring both stdout and the persistent log file capture step-level traces.
- `HTTPException` conveys user-facing errors (missing prompts, missing images, unknown jobs), while unexpected exceptions are logged with stack traces and captured in the job record.

## Running the Server
- Outbound clients should target the hosted endpoint at https://8000-01k7khhhscx852hdr4mpgt26jr.cloudspaces.litng.ai.
- Install the FastAPI stack (for example `pip install fastapi uvicorn`) alongside project dependencies defined in `requirements.txt`.
- Launch the service with uvicorn from the repository root:

```bash
uvicorn autonomous_api:app --host 0.0.0.0 --port 8000
```

- Ensure `system_prompts/` contains the Experiment 1 prompt files and that any required GPU resources are available for `VenusNaviAgent`.
- Export the relevant `UI_VENUS_*` environment variables before starting the server to tune model inference without modifying code.

## Testing with curl

Set the base URL once so each request targets the deployed endpoint:

```bash
BASE_URL="https://8000-01k7khhhscx852hdr4mpgt26jr.cloudspaces.litng.ai"
```

Submit a new autonomous run (update `prompt_name` and `image_path` to match assets available to the server):

```bash
curl -sS -X POST "$BASE_URL/autonomous/runs" \
  -H "Content-Type: application/json" \
  --data '{
    "prompt_name": "PROMPT_D3_P1_V2_H1",
    "image_path": "/teamspace/studios/this_studio/UI-Venus/examples/screenshots/sample.png",
    "context": {
      "goal": "Open the settings page"
    },
    "history_length": 0,
    "include_screenshot": false
  }'
```

The response includes a `job_id` you can poll to monitor execution. Replace `YOUR_JOB_ID` below with that value:

```bash
curl -sS "$BASE_URL/autonomous/runs/YOUR_JOB_ID"
```

Append `| jq` to either command if you want pretty-printed JSON. The run status transitions from `queued` to `running` and eventually to `succeeded` or `failed`, with detailed results attached once complete.

## Request Lifecycle
- `POST /autonomous/runs`
  - Pydantic validates the JSON payload (`prompt_name`, `image_path`, optional `context`, `history_length`, `include_screenshot`).
  - `_resolve_prompt` appends `.txt` if needed, confines lookups to `system_prompts/`, and surfaces the variant id (for example `D3_P1_V2_H1`).
  - The server checks that `image_path` exists on disk. Missing files return HTTP 400.
  - Any context dictionary is normalised to strings. Keys and values are later injected into the agent, so simple scalars work best.
  - An `AutonomousJob` record enters the in-memory store (`JobStatus.QUEUED`), and `asyncio.create_task` triggers `_execute_job` without blocking the HTTP response.
  - The response body contains `job_id`, `status`, `queued_at`, `variant_id`, and the resolved `prompt_name`.

- `_execute_job` (background task)
  - Moves the job to `running`, then invokes `_run_autonomous_inference` inside `asyncio.to_thread` so the model executes off the event loop.
  - Catches exceptions, marking the job `failed` and attaching the error message.

- `_run_autonomous_inference`
  - Loads runtime configuration from `UI_VENUS_*` environment variables via `_load_model_config`.
  - Reads the prompt template from disk, wraps it inside the chosen `PromptVariantConfig`, and initialises `VenusNaviAgent`.
  - Applies the context override twice: first when the agent is created, then via `set_autonomous_context(**context)` so values such as `"goal"` or `"ui_elements_summary"` populate prompt placeholders like `{previous_actions}`.
  - Executes a single autonomous `agent.step(goal=None, image_path=...)`, exports the agent history (optionally embedding a base64 screenshot), and captures the belief-state diagnostics.
  - Returns a dictionary with fields (`variant_id`, `prompt_source`, `agent_status`, `agent_action`, `history`, `belief_trace`, `confidence_threshold`, `max_probes`) that becomes the job result.

## Polling and Payloads
- `GET /autonomous/runs/{job_id}` always echoes the original request (including the context object) plus lifecycle timestamps (`created_at`, `started_at`, `finished_at`).
- When `status` is `succeeded`, the `result` object mirrors the structure produced by `_run_autonomous_inference`. For `failed` jobs, `error` contains the stringified exception.
- `include_screenshot=true` instructs the agent to attach a base64 PNG under `history[*].raw_screenshot_base64`; otherwise the value is omitted to keep payloads smaller.
- History truncation obeys `history_length`: `0` keeps the entire trace, any positive integer keeps only the most recent N steps when formatting `previous_actions`.
- Because the job store lives in memory, restarting the FastAPI process clears queued and completed jobs. Consider persisting job state externally if you need durability across restarts.
