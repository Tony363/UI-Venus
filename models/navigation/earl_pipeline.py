from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:  # pragma: no cover - vllm optional for dry runs
    LLM = None
    SamplingParams = None


# ---------------------------------------------------------------------------
# Prompt Loading
# ---------------------------------------------------------------------------

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_EXPERIMENT_DOC = "experiment1_earl_implementation.md"
_PROMPT_NAME = "EARL_INFERENCE_PROMPT"


def load_earl_prompt(prompt_name: str = _PROMPT_NAME) -> str:
    doc_path = _DOCS_DIR / _EXPERIMENT_DOC
    if not doc_path.exists():
        raise FileNotFoundError(
            f"Expected experiment markdown at {doc_path}, but it is missing."
        )

    source = doc_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{prompt_name}\s*=\s*\"\"\"(.*?)\"\"\"",
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        raise KeyError(
            f"Prompt '{prompt_name}' is not defined in {_EXPERIMENT_DOC}."
        )

    prompt = match.group(1).strip()
    return _normalize_prompt_template(prompt)


def _normalize_prompt_template(prompt: str) -> str:
    """
    Escape literal braces so that str.format only processes the placeholders we
    provide. The markdown template contains instructional braces such as {...}
    that would otherwise trigger formatting errors.
    """
    placeholders = {"checkpoint_pct", "observed_prefix", "ui_affordances"}
    escaped = prompt.replace("{", "{{").replace("}", "}}")
    for placeholder in placeholders:
        escaped = escaped.replace(f"{{{{{placeholder}}}}}", f"{{{placeholder}}}")
    return escaped


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GoalHypothesis:
    goal: str
    weight: float
    qualitative_tier: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class EARLInferenceResult:
    checkpoint: float
    particles: List[GoalHypothesis]
    resampled: bool
    raw_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryStep:
    index: int
    action: Optional[str]
    think: Optional[str]
    query: Optional[str]
    image_path: Optional[str]
    diagnostics: Dict[str, Any]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    trajectory_id: str
    steps: List[TrajectoryStep]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Particle Filter State (tracking only; inference is handled by the LLM output)
# ---------------------------------------------------------------------------


class EARLParticleFilter:
    """
    Lightweight state tracker that mirrors the structure described in the EARL
    experiment doc. The LLM response determines the updated belief state; this
    class records those snapshots so subsequent checkpoints can access them.
    """

    def __init__(self, particle_cap: int = 4) -> None:
        self.particle_cap = particle_cap
        self.particles: List[GoalHypothesis] = []
        self.checkpoint_history: Dict[float, List[GoalHypothesis]] = {}

    def update(self, checkpoint: float, hypotheses: List[GoalHypothesis]) -> None:
        trimmed = hypotheses[: self.particle_cap]
        self.particles = trimmed
        self.checkpoint_history[checkpoint] = trimmed

    def latest_particles(self) -> List[GoalHypothesis]:
        return list(self.particles)


# ---------------------------------------------------------------------------
# LLM Interface
# ---------------------------------------------------------------------------


class LanguageModel:
    """Minimal interface required by the EARL pipeline."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class EchoModel(LanguageModel):
    """
    Utility model for dry runs; it simply echoes the prompt so downstream
    parsing can be exercised without invoking an actual model.
    """

    def generate(self, prompt: str) -> str:
        return prompt


@dataclass
class TextVLLMConfig:
    model_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.6
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    max_model_len: int = 32768


class TextVLLM(LanguageModel):
    """
    Thin wrapper around vLLM for text-only EARL prompting. This keeps the
    navigation-specific multi-modal client untouched while enabling a separate
    pipeline for belief inference.
    """

    def __init__(self, config: TextVLLMConfig, logger: Optional[logging.Logger] = None) -> None:
        if LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vLLM is unavailable. Install vLLM or run the pipeline with --dry-run."
            )

        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        self.model = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            n=config.n,
        )
        self.logger.info(
            "Initialized TextVLLM(model=%s, tp=%d, max_tokens=%d, temperature=%.2f)",
            config.model_path,
            config.tensor_parallel_size,
            config.max_tokens,
            config.temperature,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.model.generate([prompt], sampling_params=self.sampling_params)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no outputs for the EARL prompt.")
        return outputs[0].outputs[0].text


# ---------------------------------------------------------------------------
# Prompt Rendering Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int = 240) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_observed_prefix(steps: Sequence[TrajectoryStep]) -> str:
    if not steps:
        return "No actions observed."

    lines = []
    for step in steps:
        action = step.action or "unknown action"
        thought = _truncate(step.think or "", 160)
        if thought:
            lines.append(f"Step {step.index}: {action} | rationale: {thought}")
        else:
            lines.append(f"Step {step.index}: {action}")
    return "\n".join(lines)


def _summarize_ui_affordances(step: Optional[TrajectoryStep]) -> str:
    if step is None:
        return "UI state description unavailable."

    diagnostics_text = step.diagnostics.get("raw_belief_text")
    if diagnostics_text:
        return _truncate(diagnostics_text, 320)

    if step.think:
        return _truncate(step.think, 320)

    if step.query:
        return _truncate(step.query, 320)

    return "UI state description unavailable."


# ---------------------------------------------------------------------------
# EARL Inference Core
# ---------------------------------------------------------------------------


class EARLInferenceCore:
    def __init__(
        self,
        llm: LanguageModel,
        particle_filter: Optional[EARLParticleFilter] = None,
        prompt_template: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template or load_earl_prompt()
        self.filter = particle_filter or EARLParticleFilter()
        self.logger = logger or logging.getLogger(__name__)

    def run_checkpoint(
        self,
        trajectory_prefix: Sequence[TrajectoryStep],
        progress_pct: float,
    ) -> EARLInferenceResult:
        """
        Render the EARL prompt for the provided prefix and parse the model output.
        """
        checkpoint_pct = int(round(progress_pct * 100))
        prompt = self.prompt_template.format(
            checkpoint_pct=checkpoint_pct,
            observed_prefix=_format_observed_prefix(trajectory_prefix),
            ui_affordances=_summarize_ui_affordances(
                trajectory_prefix[-1] if trajectory_prefix else None
            ),
        )

        self.logger.debug("Rendered EARL prompt for %.2f%%:\n%s", progress_pct, prompt)

        raw_response = self.llm.generate(prompt)
        result = parse_earl_response(raw_response, progress_pct)

        self.filter.update(progress_pct, result.particles)
        return result


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------


_PREDICTION_BLOCK_RE = re.compile(
    r"<predictions>(?P<body>.*?)</predictions>", flags=re.DOTALL | re.IGNORECASE
)
_RESAMPLE_RE = re.compile(r"resample:\s*\[(?P<flag>yes|no)", flags=re.IGNORECASE)
_WEIGHT_RE = re.compile(
    r"-\s*(?P<goal>.+?)\s*::\s*(?P<weight>[0-9]*\.?[0-9]+)"
    r"(?:\s*\((?P<tier>[^)]+)\))?",
    flags=re.IGNORECASE,
)


def parse_earl_response(text: str, checkpoint: float) -> EARLInferenceResult:
    """
    Extract the weighted goals from the model response. The parser is lenient to
    accommodate minor deviations from the expected format.
    """
    resampled = False
    metadata: Dict[str, Any] = {}

    think_section_match = _RESAMPLE_RE.search(text)
    if think_section_match:
        resampled = think_section_match.group("flag").lower() == "yes"
        metadata["resample_line"] = think_section_match.group(0)

    prediction_match = _PREDICTION_BLOCK_RE.search(text)
    if not prediction_match:
        raise ValueError(
            "EARL response did not contain a <predictions> block; received:\n"
            f"{text}"
        )

    body = prediction_match.group("body")
    hypotheses: List[GoalHypothesis] = []
    for line in body.splitlines():
        match = _WEIGHT_RE.search(line)
        if not match:
            continue
        goal = match.group("goal").strip()
        weight_str = match.group("weight")
        try:
            weight = float(weight_str)
        except ValueError:
            weight = float("nan")
        tier = match.group("tier")
        hypotheses.append(
            GoalHypothesis(
                goal=goal,
                weight=weight,
                qualitative_tier=tier.strip() if tier else None,
                notes=line.strip(),
            )
        )

    if not hypotheses:
        raise ValueError(
            "Unable to parse weighted goals from EARL response lines:\n"
            f"{body}"
        )

    return EARLInferenceResult(
        checkpoint=checkpoint,
        particles=hypotheses,
        resampled=resampled,
        raw_response=text,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Trajectory Loading
# ---------------------------------------------------------------------------


def _load_runner_output(path: Path) -> Iterable[Trajectory]:
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    for trace in traces:
        steps_raw = trace.get("steps", [])
        steps = [
            TrajectoryStep(
                index=index,
                action=step.get("action"),
                think=step.get("think"),
                query=step.get("query"),
                image_path=step.get("image_path"),
                diagnostics=step.get("diagnostics", {}),
                raw=step,
            )
            for index, step in enumerate(steps_raw, start=1)
        ]
        trajectory_id = str(trace.get("ground_truth_task") or trace.get("trace_index"))
        metadata = {
            "mode": trace.get("mode"),
            "variant_id": trace.get("variant_id"),
        }
        yield Trajectory(trajectory_id=trajectory_id, steps=steps, metadata=metadata)


def _load_jsonl_dataset(path: Path) -> Iterable[Trajectory]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            steps_raw = payload.get("steps", [])
            steps = [
                TrajectoryStep(
                    index=idx,
                    action=step.get("action"),
                    think=step.get("think"),
                    query=step.get("ui_state"),
                    image_path=step.get("image_path"),
                    diagnostics=step.get("diagnostics", {}),
                    raw=step,
                )
                for idx, step in enumerate(steps_raw, start=1)
            ]
            yield Trajectory(
                trajectory_id=str(payload.get("trajectory_id")),
                steps=steps,
                metadata={k: v for k, v in payload.items() if k not in {"trajectory_id", "steps"}},
            )


def load_trajectories(path: Path) -> List[Trajectory]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file {path} does not exist.")

    if path.suffix == ".jsonl":
        trajectories = list(_load_jsonl_dataset(path))
    else:
        trajectories = list(_load_runner_output(path))

    if not trajectories:
        raise ValueError(f"No trajectories parsed from {path}.")
    return trajectories


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class EARLPipelineConfig:
    checkpoints: Tuple[float, ...] = (0.25, 0.50, 0.75)
    output_path: Path = Path("belief_trace.jsonl")


class EARLPipeline:
    def __init__(
        self,
        llm: LanguageModel,
        config: Optional[EARLPipelineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm = llm
        self.config = config or EARLPipelineConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.inference = EARLInferenceCore(llm=self.llm, logger=self.logger)

    def run(self, trajectories: Sequence[Trajectory]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []

        for trajectory in trajectories:
            self.logger.info("Running EARL baseline on trajectory %s", trajectory.trajectory_id)
            num_steps = len(trajectory.steps)
            if num_steps == 0:
                self.logger.warning("Trajectory %s has no steps; skipping.", trajectory.trajectory_id)
                continue

            for checkpoint in self.config.checkpoints:
                prefix_length = max(1, int(math.ceil(num_steps * checkpoint)))
                prefix = trajectory.steps[:prefix_length]
                try:
                    result = self.inference.run_checkpoint(prefix, checkpoint)
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.error(
                        "EARL inference failed for trajectory %s at %.2f%%: %s",
                        trajectory.trajectory_id,
                        checkpoint,
                        exc,
                    )
                    outputs.append(
                        {
                            "trajectory_id": trajectory.trajectory_id,
                            "checkpoint": checkpoint,
                            "error": str(exc),
                            "metadata": trajectory.metadata,
                        }
                    )
                    continue

                outputs.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "checkpoint": checkpoint,
                        "ranked_goals": [
                            {
                                "goal": hyp.goal,
                                "weight": hyp.weight,
                                "qualitative_tier": hyp.qualitative_tier,
                                "notes": hyp.notes,
                            }
                            for hyp in result.particles
                        ],
                        "resampled": result.resampled,
                        "raw_response": result.raw_response,
                        "metadata": {
                            **trajectory.metadata,
                            **result.metadata,
                        },
                    }
                )

        return outputs

    def write_outputs(self, records: Sequence[Dict[str, Any]], output_path: Optional[Path] = None) -> None:
        output_path = output_path or self.config.output_path
        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False))
                file.write("\n")
        self.logger.info("Wrote EARL baseline results to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EARL baseline pipeline as defined in docs/experiment1_earl_implementation.md",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to trajectory data (saved_trace.json from runner or JSONL dataset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("belief_trace.jsonl"),
        help="Destination JSONL file for EARL belief traces.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model invocation and only emit rendered prompts for inspection.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Text-only model to use with vLLM. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism for vLLM TextVLLM backend.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization fraction for vLLM backend.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens generated per EARL prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for vLLM backend.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter for vLLM backend.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter for vLLM backend.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions per prompt. Defaults to 1.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum sequence length for vLLM backend.",
    )
    parser.add_argument(
        "--prompts-out",
        type=Path,
        default=None,
        help="Optional path to save rendered prompts when using --dry-run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for pipeline execution.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger("EARLPipeline")

    trajectories = load_trajectories(args.input)

    if args.dry_run:
        prompts_path = args.prompts_out or args.output.with_suffix(".prompts.txt")
        logger.info("Running in dry-run mode; prompts will be written to %s", prompts_path)

        echo_model = EchoModel()
        inference = EARLInferenceCore(llm=echo_model, logger=logger)
        checkpoints = (0.25, 0.50, 0.75)

        with prompts_path.open("w", encoding="utf-8") as file:
            for trajectory in trajectories:
                for checkpoint in checkpoints:
                    length = max(1, int(math.ceil(len(trajectory.steps) * checkpoint)))
                    prefix = trajectory.steps[:length]
                    prompt = inference.prompt_template.format(
                        checkpoint_pct=int(round(checkpoint * 100)),
                        observed_prefix=_format_observed_prefix(prefix),
                        ui_affordances=_summarize_ui_affordances(prefix[-1] if prefix else None),
                    )
                    header = f"# trajectory={trajectory.trajectory_id} checkpoint={checkpoint}"
                    file.write(header + "\n")
                    file.write(prompt + "\n\n")
        logger.info("Dry run completed; no inference records generated.")
        return

    if not args.model_path:
        parser.error("--model-path is required unless --dry-run is specified.")

    vllm_config = TextVLLMConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.n,
        max_model_len=args.max_model_len,
    )
    model = TextVLLM(config=vllm_config, logger=logger)
    pipeline = EARLPipeline(llm=model, logger=logger, config=EARLPipelineConfig(output_path=args.output))
    results = pipeline.run(trajectories)
    pipeline.write_outputs(results, args.output)


if __name__ == "__main__":
    main()
