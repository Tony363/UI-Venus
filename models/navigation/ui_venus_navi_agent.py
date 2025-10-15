import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from models.navigation.experiment1_variants import (
    DEFAULT_VARIANT_ID,
    PromptVariantConfig,
    PromptVariantError,
    get_prompt_variant,
)
from models.navigation.ui_venus_navi_vllm import NaviVLLM
from qwen_vl_utils import smart_resize
from .utils import USER_PROMPT, parse_answer


ACTION_MAPPING = {
    "click",
    "drag",
    "scroll",
    "type",
    "launch",
    "wait",
    "finished",
    "calluser",
    "longpress",
    "pressback",
    "presshome",
    "pressenter",
    "pressrecent",
    "answer",
}

_PROBE_ACTIONS = {"wait", "scroll", "pressback", "pressrecent"}


class _SafeDict(dict):
    """dict subclass that returns 'unknown' for missing keys when formatting."""

    def __missing__(self, key):
        return "unknown"


@dataclass
class BeliefStateSnapshot:
    goal: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    plan: Optional[str] = None
    raw_text: Optional[str] = None

    def to_prompt_snippet(self) -> str:
        """
        Create a compact string suitable for prompt injection when needed.
        """
        parts: List[str] = []
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        if self.confidence is not None:
            parts.append(f"Confidence: {self.confidence:.2f}")
        if self.evidence:
            parts.append(f"Evidence: {self.evidence}")
        if self.plan:
            parts.append(f"Plan: {self.plan}")
        if not parts:
            return "Goal: unknown | Confidence: 0.00"
        return " | ".join(parts)


@dataclass
class StepData:
    image_path: str
    raw_screenshot: Image.Image
    query: str
    generated_text: str
    think: str
    action: str
    _conclusion: str
    action_output_json: Optional[Dict[str, Any]] = None
    status: str = "success"
    belief_state: Optional[BeliefStateSnapshot] = None
    diagnostics: Optional[Dict[str, Any]] = None

    def to_dict(self, include_screenshot: bool = False) -> dict:
        """
        Convert this step to a JSON-serializable dict.

        Args:
            include_screenshot (bool): Whether to include base64-encoded image.

        Returns:
            dict: Serializable step data.
        """
        data = asdict(self)
        data["raw_screenshot"] = None

        if include_screenshot and self.raw_screenshot is not None:
            import base64
            from io import BytesIO

            buffer = BytesIO()
            self.raw_screenshot.save(buffer, format="PNG")
            data["raw_screenshot_base64"] = base64.b64encode(buffer.getvalue()).decode(
                "utf-8"
            )

        return data


class VenusNaviAgent:
    def __init__(
        self,
        model_config,
        logger: logging.Logger,
        history_length: int = 0,
        autonomous_variant_id: Optional[str] = None,
        autonomous_context: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = NaviVLLM(model_config=model_config, logger=logger)
        self.max_pixels = model_config.max_pixels
        self.min_pixels = model_config.min_pixels
        self.logger = logger
        self.history: List[StepData] = []
        self.history_length = max(0, history_length)
        self.resize_factor = self._infer_resize_factor()

        self.autonomous_variant: Optional[PromptVariantConfig] = None
        self.autonomous_context: Dict[str, str] = dict(autonomous_context or {})
        self.autonomous_confidence_threshold: Optional[float] = None
        self.autonomous_probe_budget: Optional[int] = None
        self.probe_count = 0
        self.last_action: Optional[Dict[str, Any]] = None
        self.belief_state_history: List[BeliefStateSnapshot] = []

        if autonomous_variant_id:
            self.set_autonomous_variant(autonomous_variant_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_autonomous_variant(self, variant_id: Optional[str]) -> None:
        """
        Configure the agent to use an autonomous prompt variant.

        Args:
            variant_id: Identifier from experiment 1 prompt catalog. When None,
                autonomous mode is disabled. The special values "default" or
                "auto" resolve to DEFAULT_VARIANT_ID.
        """
        if variant_id is None:
            self.autonomous_variant = None
            self.autonomous_confidence_threshold = None
            self.autonomous_probe_budget = None
            self.logger.info("Autonomous mode disabled; reverting to instructed mode.")
            return

        resolved_id = variant_id
        if isinstance(variant_id, str) and variant_id.lower() in {"default", "auto"}:
            resolved_id = DEFAULT_VARIANT_ID

        try:
            config = get_prompt_variant(resolved_id)
        except PromptVariantError as exc:
            self.logger.error("Failed to load autonomous prompt variant '%s': %s", variant_id, exc)
            raise

        self.autonomous_variant = config
        self.autonomous_confidence_threshold = config.confidence_threshold
        self.autonomous_probe_budget = config.max_probes
        self.probe_count = 0

        self.logger.info(
            "Loaded autonomous prompt variant %s (threshold=%.2f, max_probes=%d)",
            config.variant_id,
            config.confidence_threshold,
            config.max_probes,
        )

    def set_autonomous_context(self, **context: Optional[str]) -> None:
        """
        Update supplemental context used when rendering autonomous prompts.
        """
        for key, value in context.items():
            if value is None:
                continue
            self.autonomous_context[key] = value

    def reset(self) -> None:
        self.logger.info("Agent Reset")
        self.history = []
        self.belief_state_history = []
        self.probe_count = 0
        self.last_action = None

    def step(self, goal: Optional[str], image_path: str):
        self.logger.info("----------step %d", len(self.history) + 1)
        try:
            raw_screenshot = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self.logger.error("Can't load %s: %s", image_path, exc)
            return None

        original_width, original_height = raw_screenshot.size
        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=self.resize_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        size_params = {
            "original_width": original_width,
            "original_height": original_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
        }

        user_query = self._build_query(goal)
        generated_text = self.model([(image_path, user_query)])[0][0]

        if goal is None:
            log_goal = f"[autonomous:{self.autonomous_variant.variant_id if self.autonomous_variant else 'unset'}]"
        else:
            log_goal = goal
        mode_label = "autonomous" if goal is None else "instructed"
        variant_label = (
            self.autonomous_variant.variant_id if goal is None and self.autonomous_variant else "N/A"
        )
        self.logger.info("Goal: %s", log_goal)
        self.logger.info(
            "Prompt payload (mode=%s, variant=%s): %s",
            mode_label,
            variant_label,
            user_query,
        )
        self.logger.info("ACTION text: %r", str(generated_text))

        try:
            think_text = generated_text.split("<think>")[1].split("</think>")[0].strip("\n")
            answer_text = generated_text.split("<action>")[1].split("</action>")[0].strip("\n")
            conclusion_text = generated_text.split("<conclusion>")[1].split("</conclusion>")[0].strip("\n")
        except (IndexError, AttributeError) as exc:
            self.logger.warning("Failed to parse structured response: %s", exc)
            think_text = generated_text
            answer_text = "Wait()"
            conclusion_text = ""

        self.logger.info("Think: %s", think_text)
        self.logger.info("Answer: %s", answer_text)

        belief_snapshot = self._parse_belief_state(think_text)
        self.belief_state_history.append(belief_snapshot)
        self._maybe_log_confidence_gap(belief_snapshot)
        diagnostics = self._build_step_diagnostics(belief_snapshot)

        self.last_action = None

        try:
            action_name, action_params = parse_answer(answer_text)
            action_json = {"action": action_name, "params": action_params}
            action_json = self._convert_coordinate(action_json, size_params)
            answer_text, conclusion_text = self._maybe_inject_finished_payload(
                action_json, answer_text, conclusion_text
            )
            self.last_action = action_json
        except Exception as exc:
            self.logger.warning("Failed to parse_answer: %s", exc)
            step_data = StepData(
                image_path=image_path,
                raw_screenshot=raw_screenshot,
                query=user_query,
                generated_text=generated_text,
                think=think_text,
                action=answer_text,
                _conclusion=conclusion_text,
                status="failed",
                belief_state=belief_snapshot,
                diagnostics=diagnostics,
            )
            self.history.append(step_data)
            return None

        self._record_probe_usage(action_json["action"])
        if diagnostics is not None:
            diagnostics["probe_count"] = self.probe_count

        step_data = StepData(
            image_path=image_path,
            raw_screenshot=raw_screenshot,
            query=user_query,
            generated_text=generated_text,
            think=think_text,
            action=answer_text,
            _conclusion=conclusion_text,
            action_output_json=action_json,
            status="success",
            belief_state=belief_snapshot,
            diagnostics=diagnostics,
        )
        self.history.append(step_data)

        self.logger.info("Action: %r", str(action_json))
        return action_json

    def export_history(self, include_screenshot: bool = False):
        serialized_history = [
            step.to_dict(include_screenshot=include_screenshot) for step in self.history
        ]
        return serialized_history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_history(self) -> str:
        if not self.history:
            return ""

        recent_history = (
            self.history if self.history_length == 0 else self.history[-self.history_length :]
        )
        history_entries = [
            f"Step {i}: <think>{step.think}</think><action>{step.action}</action>"
            for i, step in enumerate(recent_history)
        ]
        return "\n".join(history_entries)

    def _build_autonomous_context(self, history_str: str) -> Dict[str, str]:
        context = dict(self.autonomous_context)
        context.setdefault("ui_elements_summary", "None")
        context.setdefault("app_category", "unknown")
        context.setdefault("user_idle_time", "unknown")
        context.setdefault("time_context", "unknown")
        context.setdefault("app_history", "unknown")
        context["previous_actions"] = history_str or "None"
        return context

    def _build_query(self, goal: Optional[str]) -> str:
        history_str = self._format_history()

        if goal is not None or self.autonomous_variant is None:
            if goal is None:
                raise ValueError(
                    "Goal must be provided when autonomous mode is disabled."
                )
            return USER_PROMPT.format(user_task=goal, previous_actions=history_str)
        if self.autonomous_variant is None:
            raise ValueError("Autonomous variant not configured for goal-less step.")

        context = self._build_autonomous_context(history_str)
        safe_context = _SafeDict(
            {key: "None" if value is None else str(value) for key, value in context.items()}
        )
        template = self.autonomous_variant.prompt_template
        try:
            return template.format_map(safe_context)
        except KeyError as exc:
            missing_key = exc.args[0]
            self.logger.warning(
                "Missing placeholder '%s' when rendering autonomous prompt for variant %s; "
                "defaulting to 'unknown'.",
                missing_key,
                self.autonomous_variant.variant_id if self.autonomous_variant else "N/A",
            )
            safe_context[missing_key] = "unknown"
            return template.format_map(safe_context)

    def _infer_resize_factor(self) -> int:
        """
        Derive the resize factor expected by qwen_vl_utils.smart_resize.

        Falls back to 28 (14 patch size * 2 merge size) when processor metadata
        is unavailable, matching Qwen2 VL defaults.
        """
        processor_wrapper = getattr(self.model, "processor", None)
        image_processor = getattr(processor_wrapper, "image_processor", None) if processor_wrapper else None

        patch_size = getattr(image_processor, "patch_size", None) if image_processor else None
        merge_size = getattr(image_processor, "merge_size", None) if image_processor else None
        size_divisor = getattr(image_processor, "size_divisor", None) if image_processor else None

        if patch_size and merge_size:
            return int(patch_size * merge_size)
        if size_divisor:
            return int(size_divisor)
        if patch_size:
            return int(patch_size)

        self.logger.warning(
            "Falling back to default resize factor=28; unable to read patch/merge size from processor."
        )
        return 28

    def _rescale_coordinate(
        self, x: float, y: float, orig_size: Tuple[int, int], resized_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        o_w, o_h = orig_size
        r_w, r_h = resized_size
        x_scaled = int(x * o_w / r_w)
        y_scaled = int(y * o_h / r_h)
        return (max(0, min(x_scaled, o_w)), max(0, min(y_scaled, o_h)))

    def _convert_coordinate(self, action_json: dict, size_params: dict):
        orig_size = (size_params["original_width"], size_params["original_height"])
        resized_size = (size_params["resized_width"], size_params["resized_height"])
        action_type = action_json["action"].lower()
        try:
            if action_type in {"click", "longpress"}:
                x, y = action_json["params"]["box"]
                action_json["params"]["box"] = self._rescale_coordinate(x, y, orig_size, resized_size)
            elif action_type == "drag":
                x1, y1 = action_json["params"]["start"]
                x2, y2 = action_json["params"]["end"]
                action_json["params"]["start"] = self._rescale_coordinate(x1, y1, orig_size, resized_size)
                action_json["params"]["end"] = self._rescale_coordinate(x2, y2, orig_size, resized_size)
            elif action_type == "scroll":
                if "start" in action_json["params"] and len(action_json["params"]["start"]) > 0:
                    x, y = action_json["params"]["start"]
                    action_json["params"]["start"] = self._rescale_coordinate(x, y, orig_size, resized_size)
                if "end" in action_json["params"] and len(action_json["params"]["end"]) > 0:
                    x, y = action_json["params"]["end"]
                    action_json["params"]["end"] = self._rescale_coordinate(x, y, orig_size, resized_size)
        except (KeyError, ValueError, TypeError) as exc:
            self.logger.warning("Coordinate conversion failed: %s, action_json=%s", exc, action_json)

        return action_json

    def _parse_belief_state(self, think_text: str) -> BeliefStateSnapshot:
        if not think_text:
            return BeliefStateSnapshot(raw_text="")

        goal: Optional[str] = None
        evidence: Optional[str] = None
        plan: Optional[str] = None
        confidence: Optional[float] = None

        goal_patterns = [
            r"(?:inferred goal|selected goal|primary goal|top goal|goal)\s*[:=]\s*(.+)",
        ]
        evidence_patterns = [
            r"(?:evidence|key evidence|reasoning)\s*[:=]\s*(.+)",
        ]
        plan_patterns = [
            r"(?:plan|next steps|execution plan|action sequence)\s*[:=]\s*(.+)",
        ]

        lines = [line.strip() for line in think_text.splitlines() if line.strip()]
        for line in lines:
            line_clean = line.lstrip("-• ").strip()

            if goal is None:
                for pattern in goal_patterns:
                    match = re.search(pattern, line_clean, re.IGNORECASE)
                    if match:
                        goal = match.group(1).strip().rstrip(".")
                        break

            if confidence is None:
                match = re.search(
                    r"confidence\s*(?:[:=]\s*|\s+)([0-9]*\.?[0-9]+)",
                    line_clean,
                    re.IGNORECASE,
                )
                if match:
                    try:
                        confidence = float(match.group(1))
                    except ValueError:
                        confidence = None

            if evidence is None:
                for pattern in evidence_patterns:
                    match = re.search(pattern, line_clean, re.IGNORECASE)
                    if match:
                        evidence = match.group(1).strip()
                        break

            if plan is None:
                for pattern in plan_patterns:
                    match = re.search(pattern, line_clean, re.IGNORECASE)
                    if match:
                        plan = match.group(1).strip()
                        break

        return BeliefStateSnapshot(
            goal=goal,
            confidence=confidence,
            evidence=evidence,
            plan=plan,
            raw_text=think_text.strip(),
        )

    def _record_probe_usage(self, action_name: str) -> None:
        if not self.autonomous_variant:
            return

        if action_name.lower() in _PROBE_ACTIONS:
            self.probe_count += 1
            if (
                self.autonomous_probe_budget is not None
                and self.probe_count > self.autonomous_probe_budget
            ):
                self.logger.warning(
                    "Probe budget exceeded for variant %s (%d/%d)",
                    self.autonomous_variant.variant_id,
                    self.probe_count,
                    self.autonomous_probe_budget,
                )

    def _maybe_log_confidence_gap(self, belief_snapshot: BeliefStateSnapshot) -> None:
        if (
            not self.autonomous_variant
            or belief_snapshot.confidence is None
            or self.autonomous_confidence_threshold is None
        ):
            return

        if belief_snapshot.confidence < self.autonomous_confidence_threshold:
            self.logger.info(
                "Belief confidence %.2f below threshold %.2f for variant %s",
                belief_snapshot.confidence,
                self.autonomous_confidence_threshold,
                self.autonomous_variant.variant_id,
            )

    def _build_step_diagnostics(
        self, belief_snapshot: BeliefStateSnapshot
    ) -> Optional[Dict[str, Any]]:
        if not self.autonomous_variant:
            return None

        diagnostics: Dict[str, Any] = {
            "autonomous_mode": True,
            "variant_id": self.autonomous_variant.variant_id,
            "confidence_threshold": self.autonomous_confidence_threshold,
            "probe_budget": self.autonomous_probe_budget,
            "raw_belief_text": belief_snapshot.raw_text,
        }

        if (
            belief_snapshot.confidence is not None
            and self.autonomous_confidence_threshold is not None
        ):
            gate_passed = belief_snapshot.confidence >= self.autonomous_confidence_threshold
            diagnostics["confidence_gate_passed"] = gate_passed
            diagnostics["confidence_margin"] = (
                belief_snapshot.confidence - self.autonomous_confidence_threshold
            )
        else:
            diagnostics["confidence_gate_passed"] = None
            diagnostics["confidence_margin"] = None

        diagnostics["goal"] = belief_snapshot.goal
        diagnostics["evidence"] = belief_snapshot.evidence
        diagnostics["plan"] = belief_snapshot.plan

        return diagnostics

    def _maybe_inject_finished_payload(
        self, action_json: Dict[str, Any], answer_text: str, conclusion_text: str
    ) -> Tuple[str, str]:
        action_name = action_json.get("action", "")
        if not isinstance(action_name, str) or action_name.lower() != "finished":
            return answer_text, conclusion_text

        params = action_json.setdefault("params", {})
        if not isinstance(params, dict):
            params = {}
            action_json["params"] = params

        existing_content = params.get("content")
        fallback_content = self._build_finished_payload(existing_content)
        if not fallback_content or (
            isinstance(existing_content, str) and existing_content.strip()
        ):
            return answer_text, conclusion_text

        params["content"] = fallback_content
        self.logger.info("Injecting fallback Finished payload: %s", fallback_content)
        updated_answer = f"Finished(content='{fallback_content}')"
        if not conclusion_text.strip():
            conclusion_text = fallback_content
        return updated_answer, conclusion_text

    def _build_finished_payload(self, existing_content: Optional[str]) -> Optional[str]:
        if isinstance(existing_content, str) and existing_content.strip():
            return existing_content.strip()

        filename = self.autonomous_context.get("target_filename")
        if not filename:
            summary = self.autonomous_context.get("ui_elements_summary", "") or ""
            match = re.search(r"([\w\-.]+\.(?:png|jpg|jpeg|gif))", summary)
            if match:
                filename = match.group(1)

        size = self.autonomous_context.get("target_file_size")
        if not size:
            summary = self.autonomous_context.get("ui_elements_summary", "") or ""
            match = re.search(r"(\d+(?:\.\d+)?\s*(?:KB|MB|GB))", summary, re.IGNORECASE)
            if match:
                size = match.group(1)

        file_id = self.autonomous_context.get("target_file_id")
        if not file_id:
            summary = self.autonomous_context.get("ui_elements_summary", "") or ""
            match = re.search(r"(?:ID|Id|id)[:：]?\s*([A-Za-z0-9_-]+)", summary)
            if match:
                file_id = match.group(1)

        parts: List[str] = []
        if filename:
            parts.append(str(filename).strip())
        if size:
            parts.append(str(size).strip())
        if file_id:
            parts.append(str(file_id).strip())

        if parts:
            while len(parts) < 3:
                parts.append("UNKNOWN")
            return " ".join(parts[:3])

        return None
