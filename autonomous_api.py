from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from models.navigation.experiment1_variants import (
    PromptVariantError,
    PromptVariantConfig,
    get_prompt_variant,
)
from models.navigation.runner import ModelConfig, setup_logger
from models.navigation.ui_venus_navi_agent import VenusNaviAgent


PROMPT_PREFIX = "PROMPT_"
PROMPT_EXTENSION = ".txt"
PROMPT_ROOT = Path(__file__).resolve().parents[2] / "system_prompts"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AutonomousStartRequest(BaseModel):
    """
    Request model for starting an autonomous navigation run.

    The request mirrors the workflow described in docs/autonomous_benchmark_runbook.md,
    where a caller provides a target screenshot and selects one of the Experiment 1
    autonomous system prompts.
    """

    prompt_name: str = Field(
        ...,
        description="Filename inside system_prompts containing the autonomous system prompt.",
    )
    image_path: str = Field(
        ...,
        description="Absolute or repository-relative path to the screenshot to analyse.",
    )
    context: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional key-value context overrides passed to the autonomous prompt.",
    )
    history_length: int = Field(
        default=0,
        ge=0,
        description="Number of previous agent steps to retain inside the prompt template.",
    )
    include_screenshot: bool = Field(
        default=False,
        description="Return the base64-encoded screenshot alongside the agent history.",
    )

    @validator("prompt_name")
    def _validate_prompt_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("prompt_name must not be empty.")
        if not cleaned.upper().startswith(PROMPT_PREFIX):
            raise ValueError(
                f"prompt_name must start with '{PROMPT_PREFIX}' and match a file in system_prompts."
            )
        return cleaned

    @validator("image_path")
    def _validate_image_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("image_path must not be empty.")
        return cleaned


@dataclass
class AutonomousJob:
    job_id: str
    request: AutonomousStartRequest
    variant_id: str
    prompt_path: Path
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AutonomousJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, AutonomousJob] = {}
        self._lock = asyncio.Lock()

    async def create(
        self, request: AutonomousStartRequest, variant_id: str, prompt_path: Path
    ) -> AutonomousJob:
        job = AutonomousJob(
            job_id=str(uuid.uuid4()),
            request=request,
            variant_id=variant_id,
            prompt_path=prompt_path,
            status=JobStatus.QUEUED,
            created_at=_now(),
        )
        async with self._lock:
            self._jobs[job.job_id] = job
        return job

    async def get(self, job_id: str) -> AutonomousJob:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    async def update(self, job_id: str, **changes: Any) -> AutonomousJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            for key, value in changes.items():
                setattr(job, key, value)
            self._jobs[job_id] = job
            return job


job_store = AutonomousJobStore()
logger = setup_logger("ui_venus.autonomous_api")
app = FastAPI(title="UI-Venus Autonomous API", version="0.1.0")


def _resolve_prompt(prompt_name: str) -> tuple[str, Path]:
    normalized = prompt_name.strip()
    if not normalized.endswith(PROMPT_EXTENSION):
        normalized = f"{normalized}{PROMPT_EXTENSION}"
    prompt_path = PROMPT_ROOT / normalized
    if not prompt_path.is_file():
        available = [p.name for p in PROMPT_ROOT.glob(f"{PROMPT_PREFIX}*{PROMPT_EXTENSION}")]
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Prompt file '{normalized}' not found in system_prompts.",
                "available_prompts": available,
            },
        )

    variant_id = normalized[: -len(PROMPT_EXTENSION)]
    if variant_id.upper().startswith(PROMPT_PREFIX):
        variant_id = variant_id[len(PROMPT_PREFIX) :]

    return variant_id, prompt_path


def _load_model_config() -> ModelConfig:
    """
    Derive ModelConfig from environment variables, falling back to defaults.

    Environment variables mirror the CLI flags in models/navigation/runner.py.
    """

    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid integer for %s=%s; falling back to %d", name, raw, default)
            return default

    def _float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid float for %s=%s; falling back to %f", name, raw, default)
            return default

    config = ModelConfig(
        model_path=os.getenv("UI_VENUS_MODEL_PATH", ModelConfig.model_path),
        tensor_parallel_size=_int_env("UI_VENUS_TP", ModelConfig.tensor_parallel_size),
        gpu_memory_utilization=_float_env(
            "UI_VENUS_GPU_MEM_UTIL", ModelConfig.gpu_memory_utilization
        ),
        max_tokens=_int_env("UI_VENUS_MAX_TOKENS", ModelConfig.max_tokens),
        max_pixels=_int_env("UI_VENUS_MAX_PIXELS", ModelConfig.max_pixels),
        min_pixels=_int_env("UI_VENUS_MIN_PIXELS", ModelConfig.min_pixels),
        max_model_len=_int_env("UI_VENUS_MAX_MODEL_LEN", ModelConfig.max_model_len),
        max_num_seqs=_int_env("UI_VENUS_MAX_NUM_SEQS", ModelConfig.max_num_seqs),
        temperature=_float_env("UI_VENUS_TEMPERATURE", ModelConfig.temperature),
        top_p=_float_env("UI_VENUS_TOP_P", ModelConfig.top_p),
        top_k=_int_env("UI_VENUS_TOP_K", ModelConfig.top_k),
        n=_int_env("UI_VENUS_SAMPLES", ModelConfig.n),
    )
    logger.info("Model configuration resolved: %s", config)
    return config


def _serialize_job(job: AutonomousJob, include_result: bool = True) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status.value,
        "prompt_name": job.prompt_path.name,
        "variant_id": job.variant_id,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "request": job.request.dict(),
    }
    if include_result:
        payload["result"] = job.result
        payload["error"] = job.error
    return payload


async def _execute_job(job_id: str) -> None:
    try:
        job = await job_store.get(job_id)
    except KeyError:
        logger.error("Job %s disappeared before execution.", job_id)
        return

    await job_store.update(job_id, status=JobStatus.RUNNING, started_at=_now())
    logger.info("Job %s started with prompt %s", job.job_id, job.prompt_path.name)

    try:
        result = await asyncio.to_thread(_run_autonomous_inference, job)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Job %s failed: %s", job.job_id, exc)
        await job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            finished_at=_now(),
        )
        return

    await job_store.update(
        job_id,
        status=JobStatus.SUCCEEDED,
        result=result,
        finished_at=_now(),
        error=None,
    )
    logger.info("Job %s completed successfully", job.job_id)


def _load_variant_config(variant_id: str, prompt_path: Path) -> PromptVariantConfig:
    try:
        variant = get_prompt_variant(variant_id)
    except PromptVariantError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Variant '{variant_id}' is not defined in docs/experiment1_prompt_variations.md.",
                "details": str(exc),
            },
        ) from exc

    prompt_template = prompt_path.read_text(encoding="utf-8")
    updated_variant = replace(variant, prompt_template=prompt_template)
    return updated_variant


def _run_autonomous_inference(job: AutonomousJob) -> Dict[str, Any]:
    image_path = Path(job.request.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path '{job.request.image_path}' not found.")

    config = _load_model_config()
    local_logger = logging.getLogger(f"ui_venus.autonomous_job.{job.job_id}")
    if not local_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        local_logger.addHandler(handler)
    local_logger.setLevel(logging.INFO)

    variant_config = _load_variant_config(job.variant_id, job.prompt_path)

    agent = VenusNaviAgent(
        config,
        local_logger,
        history_length=job.request.history_length,
        autonomous_variant_id=None,
        autonomous_context=job.request.context,
    )
    agent.set_autonomous_variant(job.variant_id)
    agent.autonomous_variant = variant_config
    agent.autonomous_confidence_threshold = variant_config.confidence_threshold
    agent.autonomous_probe_budget = variant_config.max_probes

    if job.request.context:
        agent.set_autonomous_context(**job.request.context)

    action_json = agent.step(goal=None, image_path=str(image_path))
    history = agent.export_history(include_screenshot=job.request.include_screenshot)
    belief_trace = [asdict(snapshot) for snapshot in agent.belief_state_history]

    last_step = agent.history[-1] if agent.history else None
    agent_status = last_step.status if last_step else "unknown"

    return {
        "variant_id": job.variant_id,
        "prompt_source": str(job.prompt_path),
        "agent_status": agent_status,
        "agent_action": action_json,
        "history": history,
        "belief_trace": belief_trace,
        "confidence_threshold": variant_config.confidence_threshold,
        "max_probes": variant_config.max_probes,
    }


@app.post("/autonomous/runs")
async def start_autonomous_run(request: AutonomousStartRequest) -> Dict[str, Any]:
    variant_id, prompt_path = _resolve_prompt(request.prompt_name)

    image_path = Path(request.image_path)
    if not image_path.is_file():
        raise HTTPException(
            status_code=400,
            detail={"error": f"Image path '{request.image_path}' not found."},
        )

    if request.context:
        request.context = {str(k): str(v) for k, v in request.context.items()}

    job = await job_store.create(request, variant_id, prompt_path)
    asyncio.create_task(_execute_job(job.job_id))

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "queued_at": job.created_at.isoformat(),
        "variant_id": job.variant_id,
        "prompt_name": job.prompt_path.name,
    }


@app.get("/autonomous/runs/{job_id}")
async def get_autonomous_run(job_id: str) -> Dict[str, Any]:
    try:
        job = await job_store.get(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail={"error": f"Job '{job_id}' not found."}
        ) from exc
    return _serialize_job(job)
