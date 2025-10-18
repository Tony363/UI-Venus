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
- `json` parses the optional context string supplied with multipart form submissions.
- `logging`, `os`, `uuid`, `datetime`, `timezone`, and `Path` handle instrumentation, environment introspection, identifiers, timestamps, and filesystem access.
- `dataclasses` (`dataclass`, `asdict`, `replace`) shapes internal job records and prepares belief state snapshots for response payloads.
- `Enum` and core typing helpers (`Dict`, `Optional`, `Any`) describe job status semantics and type annotations.

### Third-Party Packages
- `fastapi.FastAPI`, `fastapi.UploadFile`, `fastapi.File`, and `fastapi.Form` expose the HTTP interface, handle multipart form uploads, and surface validation errors via `HTTPException`.
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

## Screenshot Upload Handling
- Uploaded screenshots arrive as `UploadFile` instances via `multipart/form-data`. `_persist_upload_file` streams the request body into a scratch `uploads/` directory next to `autonomous_api.py`.
- Only a small set of image MIME types are accepted (`image/png`, `image/jpeg`, `image/jpg`, `image/webp`). Unsupported content types trigger HTTP 400 with the allowlist attached for debugging.
- Filenames keep their original extension when possible; otherwise a suffix is derived from the MIME type. Uploaded data is stored under a UUID-derived filename to avoid collisions.
- The saved path is injected into the `AutonomousStartRequest.image_path` field so downstream code continues to operate on filesystem paths.
- When a job was created from an upload, `_run_autonomous_inference` cleans up the temporary file once processing completes (even on failure) to prevent storage leaks.

## Request and Job Models
- `AutonomousStartRequest` (Pydantic) captures five fields: `prompt_name`, `image_path`, optional `context` overrides, `history_length`, and `include_screenshot`. The API now builds this model internally after persisting the uploaded file, so `image_path` always points to the on-disk copy inside `uploads/`. Custom validators ensure prompt names retain the `PROMPT_` prefix, remain filename-only values, and that `image_path` is non-empty.
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
- The repository ships with a `.env` file that sets `UI_VENUS_MODEL_PATH=inclusionAI/UI-Venus-Navi-7B` and `UI_VENUS_TP=1`, so out-of-the-box runs target the 7B checkpoint on a single GPU. Remove or adjust those entries if you prefer a different model.

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

Submit a new autonomous run (update `prompt_name` and the screenshot path as needed):

```bash
curl -sS -X POST "$BASE_URL/autonomous/runs" \
  -F prompt_name=PROMPT_D3_P1_V2_H1 \
  -F image=@examples/screenshots/sample.png \
  -F context='{"goal": "Open the settings page"}' \
  -F history_length=0 \
  -F include_screenshot=false
```

The response includes a `job_id` you can poll to monitor execution. Replace `YOUR_JOB_ID` below with that value:

```bash
curl -sS "$BASE_URL/autonomous/runs/YOUR_JOB_ID"
```

Append `| jq` to either command if you want pretty-printed JSON. The run status transitions from `queued` to `running` and eventually to `succeeded` or `failed`, with detailed results attached once complete.

## Request Lifecycle
- `POST /autonomous/runs`
  - Clients submit a `multipart/form-data` payload. Required fields: `prompt_name` (text) and `image` (file). Optional fields: `history_length` (integer), `include_screenshot` (boolean-ish text such as `true`/`false`), and `context` (JSON string containing a flat object).
  - The endpoint writes the uploaded screenshot to `uploads/<uuid>.<ext>` and parses the optional `context` string into a dictionary. Invalid JSON produces HTTP 400 with a descriptive message.
  - `_resolve_prompt` appends `.txt` if needed, confines lookups to `system_prompts/`, and surfaces the variant id (for example `D3_P1_V2_H1`).
  - After model validation, an `AutonomousJob` record enters the in-memory store (`JobStatus.QUEUED`) with a flag telling the background worker to delete the temporary screenshot once finished. The file path is included in the stored request so the agent can load it from disk.
  - The HTTP response contains `job_id`, `status`, `queued_at`, `variant_id`, and the resolved `prompt_name`.

  Example `curl` invocation:

  ```bash
  curl -X POST http://localhost:8000/autonomous/runs \
    -F prompt_name=PROMPT_EXPERIMENT1_BASE \
    -F image=@/path/to/screenshot.png \
    -F history_length=2 \
    -F include_screenshot=true \
    -F context='{"goal": "Open the inbox", "user_id": "qa-bot"}'
  ```

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
