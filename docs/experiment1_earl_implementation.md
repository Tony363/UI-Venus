# Experiment 1: EARL Baseline Comparison for Autonomous UI-Venus

## Executive Summary

This document scopes an EARL-inspired baseline that we will evaluate alongside existing autonomous UI-Venus prompt stacks. Rather than refactoring the agent around EARL, the goal is to reproduce the paper’s inference procedure so we can compare its intent-recognition quality against our current autonomous prompting strategies. EARL (Early Intent Recognition in GUI Tasks Using Theory of Mind) provides a theoretical framework for inferring user intent from partial action sequences, which we treat here as an external yardstick.

## 1. EARL Theory of Mind Framework

### 1.1 Core Concepts

**Theory of Mind (ToM)**: The ability to model another agent's mental states—their beliefs, desires, and intentions—to predict and explain their behavior.

**Inverse Planning**: Given observed actions, infer the most likely goal that would have produced those actions under rational planning assumptions.

**Early Recognition**: Inferring intent from minimal action sequences (1-3 actions) rather than waiting for complete demonstrations.

### 1.2 EARL's Chain of Thought Algorithm

```
INITIALIZE:
    Particles = {g₁ … g₄} drawn from goal prior
    Weights = {¼, ¼, ¼, ¼}

for each observed action prefix a₁:t and UI state s_t:
    PROPAGATE:
        Sample successor mental states for each particle (belief, desire, intent)
        Carry forward linguistic goal description
    UPDATEWEIGHTS:
        For each particle gᵢ compute qualitative likelihood tier:
            match → weight *= HIGH
            partial → weight *= MED
            mismatch → weight *= LOW
    NORMALIZE weights so Σwᵢ = 1
    RESAMPLE if effective particle count < threshold to preserve diversity
    RECORD predictions at checkpoints (25%, 50%, 75% progress):
        SUMMARIZEBELIEFTRACE(particles) → top-k goal strings with weights

Return the goal distribution; no environment actions are executed during inference.
```

### 1.3 Mental State Components

```python
@dataclass
class MentalState:
    beliefs: Dict[str, float]  # What the user believes about UI state
    desires: List[str]          # What the user likely wants
    intentions: List[str]       # What the user plans to do
    confidence: float           # Overall confidence in assessment
```

### 1.4 Inference Scope vs. Control Extensions

EARL, as presented in the paper, is exclusively an **inference-time** algorithm: it consumes action prefixes and UI states and returns a ranked goal distribution without issuing GUI actions. For UI-Venus we therefore isolate EARL as a comparable module rather than a replacement for our autonomous controller:

- **EARL Inference Core (Baseline)**: Implements the particle-filter loop above, capped at four concurrent hypotheses, and surfaces predictions at the mandated checkpoints (25 %, 50 %, 75 % of a trajectory). We log its belief traces for side-by-side evaluation with our existing prompt stacks.
- **Autonomous Controller (Optional Ablation)**: Downstream components may react to EARL’s belief trace (e.g., to trigger probes or actions), but these behaviors are treated as separate ablations layered on top of the inference core. The default comparison run keeps controller logic disabled to match the paper.

All prompt templates and agent flows in this experiment must clearly indicate when we are executing the canonical EARL loop versus optional UI-Venus extensions so that comparison against our autonomous prompts stays fair.

## 2. Prompt Design Layers

EARL prompts must first elicit the inference-time particle filter decisions (Algorithm 1) and only then feed optional controller logic. To keep comparisons clean, the baseline experiment uses only the inference template; the controller template is supplied for optional ablations.

### 2.1 Inference Trace Template

```python
EARL_INFERENCE_PROMPT = """
**You are the EARL inference module.**

Goal: given partial trajectory prefixes, maintain a belief distribution over latent user goals. Do NOT propose UI actions.

### Inputs
- Checkpoint fraction: {checkpoint_pct}  # 25, 50, or 75
- Observed trajectory:
{observed_prefix}
- Current UI affordances: {ui_affordances}

### Particle Filter State
Particles (max 4):
1. [goal₁] | weight: [0-1] | notes: [mental state trace]
2. [goal₂] | weight: [0-1] | notes: [mental state trace]
3. [goal₃] | weight: [0-1] | notes: [mental state trace]
4. [goal₄] | weight: [0-1] | notes: [mental state trace]

### Required Reasoning
1. PROPAGATE: carry forward beliefs and note mental state transitions.
2. UPDATEWEIGHTS: assign qualitative likelihood tiers {{"match", "partial", "mismatch"}} with brief justification.
3. NORMALIZE: present final weights so they sum to 1.0.
4. RESAMPLE (if needed): state whether resampling occurred and why.
5. SUMMARIZEBELIEFTRACE: emit ranked goal list for the current checkpoint.

### Output Format
<think>
- propagate: [...]
- updateweights: [...]
- normalize: weights -> {...}
- resample: [yes/no, rationale]
- summarize:
  * top_goal: [goal string] | weight=[0-1]
  * alt_goal_2: [...]
  * alt_goal_3: [...]
</think>
<predictions>
checkpoint: {checkpoint_pct}
top_goal: [goal string]
weighted_goals:
- [goal₁] :: [weight]
- [goal₂] :: [weight]
- [goal₃] :: [weight]
- [goal₄] :: [weight]
</predictions>
"""
```

### 2.2 Controller Extension (Optional)

Controller prompts may consume the `<predictions>` block to decide on probes or actions. They must clearly label themselves as extensions so evaluators can isolate pure EARL inference runs.

```python
EARL_CONTROLLER_PROMPT = """
**You are the UI-Venus controller consuming EARL predictions.**

Inputs:
- EARL belief trace (25/50/75% checkpoints)
- Probe budget: {max_probes}
- Safety threshold: {threshold}

Decision Policy:
1. Inspect top_goal weight. If < {threshold}, select probe from PROBE_ACTIONS.
2. If ≥ {threshold} and action is reversible, suggest execution; otherwise, request confirmation.
3. Log all probes in belief_state.probe_history for auditability.

Output Format
<think>controller reasoning...</think>
<action>[Probe() or Execute()]</action>
<summary>[one-line justification]</summary>
"""
```

Document any experiment that leverages the controller layer separately from those evaluating the inference core.

> **Usage note**: The primary EARL comparison keeps the controller disabled so that reported results reflect inference-only behavior. Enable the controller prompt only for explicit ablations against our autonomous action-selection prompts.

## 3. EARL Prompt Variants

### 3.1 Variant Matrix

All variants respect the four-particle cap and checkpoint reporting. We sample these variants to benchmark how different EARL prompt styles perform relative to our autonomous system prompts. `Threshold` and `Probes` apply exclusively to the controller extension; inference-only runs set them to `—`.

| ID | ToM Level | Strategy | Threshold | Probes | Risk | Tokens |
|----|-----------|----------|-----------|--------|------|--------|
| E1 | Heavy | Decisive | 0.6 | 1 | Med | High |
| E2 | Heavy | Cautious | 0.8 | 2 | Low | High |
| E3 | Heavy | Proactive | 0.4 | 3 | High | High |
| E4 | Light | Decisive | 0.6 | 1 | Med | Med |
| E5 | Light | Cautious | 0.8 | 2 | Low | Med |
| E6 | Light | Proactive | 0.4 | 3 | High | Med |
| E7 | Minimal | Balanced | 0.5 | 2 | Med | Low |
| E8 | Heavy | Probe-First | 0.7 | 3 | Low | High |
| E9 | Light | Commit-First | 0.5 | 1 | High | Med |
| E10 | Heavy | Risk-Averse | 0.85 | 2 | Min | High |
| E11 | Light | Opportunistic | 0.35 | 1 | High | Med |
| E12 | Minimal | Efficient | 0.6 | 1 | Med | Min |
| E13 | Adaptive | Dynamic | Variable | 2 | Med | Med |

### 3.2 Detailed Variant Specifications

#### Variant E1: ToM-Heavy Decisive
```python
PROMPT_E1_TOM_HEAVY_DECISIVE = """
**You are an autonomous GUI Agent with advanced Theory of Mind.**

### Deep Mental State Modeling
Perform detailed reasoning about the user's cognitive state:
- Perception: What UI elements has the user likely noticed?
- Attention: What is the user likely focused on?
- Memory: What recent actions indicate retained intent?
- Goals: What objective explains this behavior pattern?

### Belief Tracking (Detailed)
For each hypothesis, provide:
- Goal statement
- Mental model: How user views the task
- Action plan: Expected next steps
- Weight: Posterior probability given evidence

### Controller Threshold
Act decisively when top_weight > 0.6 (only if controller layer active)
Maximum 1 probe action if uncertain

[Rest follows EARL_INFERENCE_PROMPT structure]
"""
```

#### Variant E2: ToM-Heavy Cautious
```python
PROMPT_E2_TOM_HEAVY_CAUTIOUS = """
**You are a cautious GUI Agent with Theory of Mind.**

### Conservative Mental State Assessment
Carefully model user intent with high evidence standards:
- Require multiple corroborating UI cues
- Consider alternative explanations
- Prefer exploration over assumption

### Belief Validation
Before committing to a goal:
- Check for contradictory evidence
- Verify UI state supports the hypothesis
- Ensure action reversibility

### Controller Threshold
Only act when top_weight > 0.8
Allow up to 2 probe actions for disambiguation

[Rest follows EARL_INFERENCE_PROMPT structure]
"""
```

#### Variant E7: Minimal Balanced
```python
PROMPT_E7_MINIMAL_BALANCED = """
**You are an efficient autonomous GUI Agent.**

### Quick Intent Assessment
Infer user goal from:
- Key UI elements
- Recent actions
- Common patterns

### Compact Belief State
Top goal: [goal] (weight: [0-1])
Alt goal: [goal] (weight: [0-1])
Evidence: [brief list]

### Decision Rule
Act if top_weight > 0.5, else probe (max 2)

[Compressed action list and output format]
"""
```

#### Variant E13: Adaptive Dynamic
```python
PROMPT_E13_ADAPTIVE = """
**You are an adaptive GUI Agent with situational awareness.**

### Context-Sensitive Thresholds
Adjust controller thresholds based on:
- Clear modal/dialog → Act at 0.3
- Standard screen → Act at 0.5
- Ambiguous state → Act at 0.7
- Risky action → Act at 0.85

### Dynamic Strategy
- High urgency (errors, timeouts) → Immediate action
- Exploration needed → Probe systematically
- Clear affordances → Direct execution

[Rest adapts based on UI context]
"""
```

## 4. Affordance Priors and Pattern Library

### 4.1 UI Pattern → Intent Mapping

These priors map interface patterns to qualitative likelihood tiers so the particle filter can implement `QUALWEIGHTS` using the same HIGH/MED/LOW buckets as the paper.

```python
AFFORDANCE_PRIORS = {
    # Authentication patterns
    "login_button + username_field": {
        "intent": "authenticate",
        "likelihood_tier": "match",
        "weight_multiplier": 1.3
    },

    # Search patterns
    "search_bar + magnifier_icon": {
        "intent": "find_information",
        "likelihood_tier": "match",
        "weight_multiplier": 1.25
    },

    # Commerce patterns
    "cart_icon + product_grid": {
        "intent": "purchase",
        "likelihood_tier": "match",
        "weight_multiplier": 1.28
    },

    # Creation patterns
    "fab + plus_icon": {
        "intent": "create_new",
        "likelihood_tier": "match",
        "weight_multiplier": 1.3
    },

    # Navigation patterns
    "hamburger_menu": {
        "intent": "explore_options",
        "likelihood_tier": "partial",
        "weight_multiplier": 1.1
    },

    # Settings patterns
    "gear_icon + toggle_switches": {
        "intent": "configure",
        "likelihood_tier": "match",
        "weight_multiplier": 1.28
    }
}
```

### 4.2 Probe Action Library

```python
PROBE_ACTIONS = {
    "reveal_menu": "Click(hamburger_menu)",
    "check_search": "Click(search_field)",
    "explore_tabs": "Click(next_tab)",
    "scroll_peek": "Scroll(direction='down', distance='small')",
    "back_peek": "PressBack()",
    "long_press_explore": "LongPress(main_element)"
}
```

## 5. Implementation Architecture

We package EARL as a standalone evaluation module so it can be invoked beside the existing autonomous agent without altering core navigation flows. Integration points below highlight how to plug the baseline into our harness while keeping our production prompts intact.

### 5.1 Agent Modifications

```python
class EARLParticleFilter:
    """Implements Algorithm 1 from EARL with particle cap = 4."""

    def __init__(self, particle_count: int = 4):
        self.max_particles = particle_count
        self.particles = self._initialize_particles()
        self.checkpoint_history: Dict[float, List[GoalHypothesis]] = {}

    def observe_prefix(
        self,
        trajectory_prefix: Sequence[Action],
        ui_state: UIState,
        progress_pct: float,
    ) -> List[GoalHypothesis]:
        """Run one EARL loop for the current prefix."""
        self._propagate(trajectory_prefix, ui_state)
        self._update_weights(trajectory_prefix, ui_state)  # qualitative tiers
        self._normalize()
        if self._needs_resample():
            self._resample()
        summary = self._summarize(progress_pct)
        self.checkpoint_history[progress_pct] = summary
        return summary

    # ... helper methods mirror Algorithm 1 primitives ...


class EARLInferenceCore:
    """Glue code that wraps the particle filter with prompting."""

    def __init__(self, filter_impl: EARLParticleFilter, prompt_template: str):
        self.filter = filter_impl
        self.prompt_template = prompt_template

    def run_checkpoint(
        self,
        trajectory_prefix: Sequence[Action],
        ui_state: UIState,
        progress_pct: float,
    ) -> EARLInferenceResult:
        summary = self.filter.observe_prefix(trajectory_prefix, ui_state, progress_pct)
        prompt = self.prompt_template.format(
            checkpoint_pct=f"{progress_pct:.2f}",
            observed_prefix=self._format_prefix(trajectory_prefix),
            ui_affordances=self._format_affordances(ui_state),
        )
        return call_model(prompt, summary)


class VenusEARLController(VenusNaviAgent):
    """Optional extension that consumes EARL predictions to choose probes/actions."""

    def __init__(self, inference_core: EARLInferenceCore, variant_config):
        super().__init__(model_config)
        self.inference_core = inference_core
        self.variant_config = variant_config
        self.probe_budget = variant_config.max_probes
        self.safety_threshold = variant_config.threshold

    def _plan(self, observation: UIState) -> ControllerDecision:
        prefixes = self._collect_prefixes(observation)
        earls = {
            pct: self.inference_core.run_checkpoint(prefix, observation, pct)
            for pct, prefix in prefixes.items()
        }
        belief_state = BeliefState.from_earl_outputs(earls)
        if belief_state.top_weight >= self.safety_threshold:
            return self._recommend_execution(belief_state)
        if belief_state.can_probe(self.probe_budget):
            return self._recommend_probe(belief_state)
        return ControllerDecision.defer(reason="insufficient belief weight")
```

### 5.2 Belief State Management

```python
@dataclass
class GoalHypothesis:
    goal: str
    weight: float
    qualitative_tier: str
    notes: str


@dataclass
class BeliefState:
    particles: List[GoalHypothesis]  # capped at 4
    checkpoint_history: Dict[float, List[GoalHypothesis]]
    probe_history: List[str] = field(default_factory=list)

    @property
    def top_goal(self) -> str:
        return max(self.particles, key=lambda h: h.weight).goal

    @property
    def top_weight(self) -> float:
        return max(self.particles, key=lambda h: h.weight).weight

    @classmethod
    def from_earl_outputs(
        cls, checkpoint_outputs: Dict[float, EARLInferenceResult]
    ) -> "BeliefState":
        latest_pct = max(checkpoint_outputs)
        particles = checkpoint_outputs[latest_pct].particles[:4]
        history = {
            pct: result.particles[:4] for pct, result in checkpoint_outputs.items()
        }
        return cls(particles=particles, checkpoint_history=history)

    def can_probe(self, budget: int) -> bool:
        return len(self.probe_history) < budget

    def get_probe_action(self) -> str:
        """Select informative probe based on entropy reduction."""
        return self._max_info_gain_probe()
```

### 5.3 Baseline Pipeline Entry Point

We ship a standalone runner at `models/navigation/earl_pipeline.py` that implements the inference-only EARL baseline. It loads the canonical prompt template from this document, renders checkpoints at 25 %, 50 %, and 75 %, and expects a text-only LLM backend.

- Dry-run prompt inspection:
  ```
  bash scripts/run_earl_baseline.sh saved_trace.json belief_trace.prompts --dry-run
  ```
- Full inference (set `MODEL_PATH` to a vLLM-compatible checkpoint):
  ```
  MODEL_PATH=/path/to/model \
  bash scripts/run_earl_baseline.sh saved_trace.json belief_trace.jsonl \
    --tensor-parallel-size 2 --max-tokens 1536
  ```

Outputs are written as JSONL records (`belief_trace.jsonl`) ready for the evaluation protocol below. The pipeline accepts runner outputs (`saved_trace.json`) or dataset JSONL files describing trajectories.

## 6. Evaluation Protocol

We execute EARL on the same replay trajectories and dataset splits used for our autonomous prompt experiments so that metrics are directly comparable. Unless an ablation is explicitly requested, controller logic remains disabled and we report inference-only scores.

### 6.1 Core EARL Metrics

**Perfect Match Rate (PMR)**  
Fraction of instances where the top EARL goal exactly matches the ground-truth intent string at each checkpoint.

**Weighted Mean Score (WMS)**  
For each prediction, run an NLI model comparing the predicted goal with the gold goal. Assign weights {entailment: 1.0, neutral: 0.5, contradiction: 0.0} and average across checkpoints.

**Checkpoint Coverage**  
Verify that predictions are emitted at the 25 %, 50 %, and 75 % prefixes via `SUMMARIZEBELIEFTRACE`; missing checkpoints invalidate the run.

### 6.2 Reporting Format

Produce a `belief_trace.jsonl` file per evaluation containing, for each prefix:
```json
{
  "trajectory_id": "...",
  "checkpoint": 0.25,
  "ranked_goals": [
    {"goal": "...", "weight": 0.46},
    {"goal": "...", "weight": 0.31},
    {"goal": "...", "weight": 0.18},
    {"goal": "...", "weight": 0.05}
  ],
  "resampled": true
}
```
Aggregate PMR and WMS over the dataset and report 95 % confidence intervals as in the paper.

### 6.3 Supplemental Controller Metrics (Optional)

If the UI-Venus controller extension is evaluated, add:
- **Time-to-Intent (TTI)**: steps between the first checkpoint and the point when `top_weight` crosses the controller threshold.
- **Probe Utility (PU)**: mean entropy reduction attributable to each probe action.
- **Action Safety Rate (ASR)**: percentage of executed actions that align with the gold goal when judged after completion.

Clearly separate controller results from the inference-only EARL metrics in experiment write-ups.

## 7. Expected Advantages of EARL Approach

### 7.1 Theoretical Benefits

1. **Principled Reasoning**: Grounded in cognitive science theory
2. **Early Recognition**: Faster intent inference from partial sequences
3. **Uncertainty Handling**: Explicit belief tracking and weight updates
4. **Adaptability**: Mental model updates based on evidence

### 7.2 Practical Benefits

1. **Reduced Ambiguity**: ToM reasoning disambiguates multiple valid goals
2. **Efficient Exploration**: Information-theoretic probe selection
3. **Stable Behavior**: Belief persistence reduces goal oscillation
4. **Interpretability**: Explicit mental state reasoning in outputs

## 8. Risk Mitigation

### 8.1 Computational Overhead
- Token-heavy ToM reasoning → Use E7-E12 minimal variants
- Complex belief updates → Cache and reuse computations

### 8.2 Goal Misidentification
- Wrong initial hypothesis → Allow belief revision
- Overconfident inference → Require multiple evidence sources

### 8.3 Action Safety
- Risky actions from wrong goals → High controller thresholds
- Irreversible actions → Require explicit confirmation

## 9. Integration with Experiment 2-4

### 9.1 Experiment 2: ACE Evolution
Use the best-performing EARL variant as an additional seed when running automated prompt evolution, keeping autonomous prompts in the candidate pool for comparison.

### 9.2 Experiment 3: Multi-modal Context
Enhance mental state with app metadata and user state

### 9.3 Experiment 4: DPO Fine-tuning
Generate preference pairs from belief state trajectories

## 10. Conclusion

The EARL-inspired implementation serves as a comparative baseline alongside our autonomous UI-Venus prompt stacks. By reproducing the paper’s inference-only particle filter, we can measure how Theory-of-Mind-style intent inference fares relative to the goal-directed behaviors already present in our system.

Key observations we expect to surface from the comparison:
- **Theory-driven**: EARL offers cognitively grounded reasoning over latent goals.
- **Belief-based**: Explicit hypothesis tracking exposes uncertainty patterns that we can contrast with autonomous prompts.
- **Lightweight integration**: Because the baseline stays inference-only, we can swap it in and out of evaluation harnesses without altering the core controller.

Rather than supplanting our existing flows, EARL adds another experimental datapoint that helps determine when ToM-style inverse planning yields measurable gains over our current autonomous prompting strategies.
