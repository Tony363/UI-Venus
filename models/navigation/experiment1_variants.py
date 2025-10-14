from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


class PromptVariantError(RuntimeError):
    """Raised when experiment 1 prompt variants cannot be loaded."""


@dataclass(frozen=True)
class PromptVariantConfig:
    """
    Configuration container for an autonomous prompt variant.

    Attributes:
        variant_id: Public identifier (e.g. "D3_P2_V2_H1").
        prompt_template: Full system prompt template string.
        confidence_threshold: Minimum confidence required before acting.
        max_probes: Maximum number of exploratory probe actions.
        directness: Human-readable directness label.
        persona: Persona descriptor ("cautious", "balanced", "proactive", ...).
        verbosity: Planning verbosity descriptor.
        history_mode: Whether history is emphasised ("with_history"/"no_history").
        requires_ui_summary: True when the template expects a UI summary input.
        raw_name: Original constant name parsed from the markdown source.
        notes: Free-form notes about additional requirements.
    """

    variant_id: str
    prompt_template: str
    confidence_threshold: float
    max_probes: int
    directness: str
    persona: str
    verbosity: str
    history_mode: str
    requires_ui_summary: bool = False
    raw_name: Optional[str] = None
    notes: Optional[str] = None


_PERSONA_CONFIDENCE = {
    "1": 0.7,  # cautious
    "2": 0.5,  # balanced
    "3": 0.35,  # proactive
}

_PERSONA_PROBES = {
    "1": 1,
    "2": 2,
    "3": 3,
}

_DIRECTNESS_LABELS = {
    "1": "implicit",
    "2": "light",
    "3": "moderate",
    "4": "explicit",
}

_PERSONA_LABELS = {
    "1": "cautious",
    "2": "balanced",
    "3": "proactive",
}

_VERBOSITY_LABELS = {
    "1": "minimal",
    "2": "detailed",
}

_HISTORY_LABELS = {
    "0": "no_history",
    "1": "with_history",
}

_DEFAULT_CONFIDENCE = 0.5
_DEFAULT_MAX_PROBES = 2

_DOC_FILENAME = "experiment1_prompt_variations.md"
_PROMPT_PREFIX = "PROMPT_"


_PROMPT_CACHE: Dict[str, str] = {}
_VARIANT_CACHE: Dict[str, PromptVariantConfig] = {}


def _docs_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / _DOC_FILENAME


def _read_docs() -> str:
    doc_path = _docs_path()
    try:
        return doc_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptVariantError(
            f"Unable to load experiment 1 prompt variants; missing {doc_path}."
        ) from exc


def _extract_prompt_definitions() -> Dict[str, str]:
    if _PROMPT_CACHE:
        return _PROMPT_CACHE

    markdown = _read_docs()
    code_blocks = re.findall(r"```python\n(.*?)```", markdown, re.DOTALL)

    if not code_blocks:
        raise PromptVariantError(
            "No python code blocks found in experiment1 prompt variation docs."
        )

    extracted: Dict[str, str] = {}
    for block in code_blocks:
        namespace: Dict[str, str] = {}
        exec(block, {}, namespace)  # nosec: trusted repository-controlled source
        for name, value in namespace.items():
            if name.startswith(_PROMPT_PREFIX) and isinstance(value, str):
                extracted[name] = value

    if not extracted:
        raise PromptVariantError(
            "Failed to evaluate any PROMPT_* definitions from markdown source."
        )

    _PROMPT_CACHE.update(extracted)
    return _PROMPT_CACHE
'''
  1. Smoke-test both modes with python -m models.navigation.runner --mode instructed and
     --mode autonomous --variant_id D3_P2_V2_H0 on examples/trace/trace.json once the model
     weights are accessible.
'''

def _split_standard_variant(raw_variant_name: str) -> Optional[Dict[str, str]]:
    parts = raw_variant_name.split("_")
    if len(parts) < 4:
        return None

    directness_code, persona_code, verbosity_code, history_code = parts[:4]
    if not (
        directness_code.startswith("D")
        and persona_code.startswith("P")
        and verbosity_code.startswith("V")
        and history_code.startswith("H")
    ):
        return None

    suffix = "_".join(parts[4:]) if len(parts) > 4 else ""

    directness_level = directness_code[1:]
    persona_level = persona_code[1:]
    verbosity_level = verbosity_code[1:]
    history_level = history_code[1:]

    directness_label = _DIRECTNESS_LABELS.get(directness_level, "unknown")
    persona_label = _PERSONA_LABELS.get(persona_level, "unknown")
    verbosity_label = _VERBOSITY_LABELS.get(verbosity_level, "unknown")
    history_label = _HISTORY_LABELS.get(history_level, "unknown")

    confidence = _PERSONA_CONFIDENCE.get(persona_level, _DEFAULT_CONFIDENCE)
    max_probes = _PERSONA_PROBES.get(persona_level, _DEFAULT_MAX_PROBES)

    requires_ui_summary = suffix.upper() == "ENHANCED"

    variant_id = "_".join(
        filter(None, [directness_code, persona_code, verbosity_code, history_code, suffix])
    )

    return {
        "variant_id": variant_id,
        "directness": directness_label,
        "persona": persona_label,
        "verbosity": verbosity_label,
        "history_mode": history_label,
        "confidence_threshold": confidence,
        "max_probes": max_probes,
        "requires_ui_summary": requires_ui_summary,
        "notes": "Enhanced template expects a UI summary input."
        if requires_ui_summary
        else None,
    }


def _parse_special_variant(raw_variant_name: str) -> Dict[str, str]:
    raw_upper = raw_variant_name.upper()
    notes = None
    requires_ui_summary = False

    if raw_upper == "CONTEXT_HEAVY":
        notes = "Supports optional environmental context placeholders."
        requires_ui_summary = False
    elif raw_upper == "CONFIDENCE_ADAPTIVE":
        notes = "Implements dynamic confidence guidance inside the prompt."

    return {
        "variant_id": raw_variant_name,
        "directness": "custom",
        "persona": "custom",
        "verbosity": "custom",
        "history_mode": "with_history",
        "confidence_threshold": _DEFAULT_CONFIDENCE,
        "max_probes": _DEFAULT_MAX_PROBES,
        "requires_ui_summary": requires_ui_summary,
        "notes": notes,
    }


def _build_variant_config(raw_name: str, prompt_template: str) -> PromptVariantConfig:
    metadata = _split_standard_variant(raw_name)
    if metadata is None:
        metadata = _parse_special_variant(raw_name)

    return PromptVariantConfig(
        variant_id=metadata["variant_id"],
        prompt_template=prompt_template,
        confidence_threshold=metadata["confidence_threshold"],
        max_probes=metadata["max_probes"],
        directness=metadata["directness"],
        persona=metadata["persona"],
        verbosity=metadata["verbosity"],
        history_mode=metadata["history_mode"],
        requires_ui_summary=metadata["requires_ui_summary"],
        raw_name=raw_name,
        notes=metadata.get("notes"),
    )


def load_all_prompt_variants() -> Dict[str, PromptVariantConfig]:
    """
    Load and cache all experiment 1 prompt variants defined in the documentation.
    """
    if _VARIANT_CACHE:
        return dict(_VARIANT_CACHE)

    prompt_definitions = _extract_prompt_definitions()
    for constant_name, prompt_template in prompt_definitions.items():
        raw_variant_name = constant_name[len(_PROMPT_PREFIX) :]
        config = _build_variant_config(raw_variant_name, prompt_template)
        _VARIANT_CACHE[config.variant_id] = config

    return dict(_VARIANT_CACHE)


def get_prompt_variant(variant_id: str) -> PromptVariantConfig:
    """
    Retrieve a specific prompt variant configuration.

    Raises:
        PromptVariantError: If the variant is unknown.
    """
    variants = load_all_prompt_variants()
    try:
        return variants[variant_id]
    except KeyError as exc:
        available = ", ".join(sorted(variants.keys()))
        raise PromptVariantError(
            f"Unknown experiment 1 prompt variant '{variant_id}'. "
            f"Available variants: {available}"
        ) from exc


def list_prompt_variants() -> Iterable[str]:
    """
    Return an iterable of available variant identifiers.
    """
    variants = load_all_prompt_variants()
    return sorted(variants.keys())


DEFAULT_VARIANT_ID = "D3_P2_V2_H0"
