# Experiment 1: EARL-Inspired Implementation for Autonomous UI-Venus

## Executive Summary

This document presents an EARL-inspired approach to transform UI-Venus from instruction-following to autonomous goal-inferring behavior using Theory of Mind (ToM) principles and inverse planning. EARL (Early Intent Recognition in GUI Tasks Using Theory of Mind) provides a theoretical framework for inferring user intent from partial action sequences.

## 1. EARL Theory of Mind Framework

### 1.1 Core Concepts

**Theory of Mind (ToM)**: The ability to model another agent's mental states—their beliefs, desires, and intentions—to predict and explain their behavior.

**Inverse Planning**: Given observed actions, infer the most likely goal that would have produced those actions under rational planning assumptions.

**Early Recognition**: Inferring intent from minimal action sequences (1-3 actions) rather than waiting for complete demonstrations.

### 1.2 EARL's Chain of Thought Algorithm

```
For each UI state:
1. OBSERVE: Current screen, UI affordances, recent actions
2. HYPOTHESIZE: Generate candidate goals based on:
   - What would a rational user want given this UI?
   - What goals explain the observed action sequence?
3. SCORE: Assign probabilities using:
   - P(goal | actions, UI) ∝ P(actions | goal, UI) × P(goal | UI)
4. UPDATE: Refine beliefs based on new evidence
5. ACT: Choose action that:
   - Disambiguates between competing hypotheses (probe)
   - Progresses toward most likely goal (execute)
6. PERSIST: Maintain belief state across turns
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

## 2. EARL-Inspired System Prompt Core

### 2.1 Base Template

```python
EARL_PROMPT_CORE = """
**You are an autonomous GUI Agent with Theory of Mind capabilities.**

You must infer the user's goal by reasoning about their mental state and the observed UI.

### Theory of Mind Assessment
Before acting, model the user's mental state:
1. What does the user BELIEVE about the current app state?
2. What does the user likely WANT to accomplish?
3. What EVIDENCE supports these inferences?

### Belief State Tracking
Maintain up to 3 goal hypotheses:
- Hypothesis 1: [goal] | Evidence: [UI cues] | Confidence: [0-1]
- Hypothesis 2: [goal] | Evidence: [UI cues] | Confidence: [0-1]
- Hypothesis 3: [goal] | Evidence: [UI cues] | Confidence: [0-1]

### Previous Actions as Intent Signals
{previous_actions}

### Inverse Planning
Ask yourself: "What goal would make these observed actions rational?"
- If actions form a coherent sequence → high confidence in inferred goal
- If actions seem exploratory → user may be uncertain, probe for clarity
- If actions contradict → revise goal hypotheses

### UI Affordance Analysis
Identify intent cues from the interface:
- Modal dialogs/permissions → immediate resolution needed
- Empty required fields → likely completion intent
- Prominent CTAs → primary user flow
- Error messages → correction intent
- Search bars → information seeking

### Available Actions
[Standard action list preserved]

### Decision Process
1. Generate/update goal hypotheses using ToM reasoning
2. Calculate confidence based on evidence strength
3. IF confidence > {threshold}:
   - Execute goal-directed action
   ELSE:
   - Perform information-gathering probe
4. Limit probes to {max_probes} per episode

### Output Format
<think>
Mental State:
- User believes: [current understanding]
- User wants: [inferred goal]
- Evidence: [supporting observations]
- Confidence: [0-1]

Goal hypotheses:
1. [goal]: [confidence] because [reasoning]
2. [goal]: [confidence] because [reasoning]

Next action reasoning:
[Why this action advances the inferred goal or gathers information]
</think>

<action>[Action specification]</action>

<conclusion>[Brief summary]</conclusion>
"""
```

## 3. EARL Prompt Variants

### 3.1 Variant Matrix

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
- Confidence: Posterior probability given evidence

### Decision Threshold
Act decisively when confidence > 0.6
Maximum 1 probe action if uncertain

[Rest follows EARL_PROMPT_CORE structure]
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

### Decision Threshold
Only act when confidence > 0.8
Allow up to 2 probe actions for disambiguation

[Rest follows EARL_PROMPT_CORE structure]
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
Top goal: [goal] (confidence: [0-1])
Alt goal: [goal] (confidence: [0-1])
Evidence: [brief list]

### Decision Rule
Act if confidence > 0.5, else probe (max 2)

[Compressed action list and output format]
"""
```

#### Variant E13: Adaptive Dynamic
```python
PROMPT_E13_ADAPTIVE = """
**You are an adaptive GUI Agent with situational awareness.**

### Context-Sensitive Thresholds
Adjust confidence requirements based on:
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

```python
AFFORDANCE_PRIORS = {
    # Authentication patterns
    "login_button + username_field": {
        "intent": "authenticate",
        "confidence_boost": 0.4
    },

    # Search patterns
    "search_bar + magnifier_icon": {
        "intent": "find_information",
        "confidence_boost": 0.3
    },

    # Commerce patterns
    "cart_icon + product_grid": {
        "intent": "purchase",
        "confidence_boost": 0.35
    },

    # Creation patterns
    "fab + plus_icon": {
        "intent": "create_new",
        "confidence_boost": 0.4
    },

    # Navigation patterns
    "hamburger_menu": {
        "intent": "explore_options",
        "confidence_boost": 0.2
    },

    # Settings patterns
    "gear_icon + toggle_switches": {
        "intent": "configure",
        "confidence_boost": 0.35
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

### 5.1 Agent Modifications

```python
class EARLVenusAgent(VenusNaviAgent):
    def __init__(self, variant_config):
        super().__init__(model_config)
        self.prompt_template = load_earl_variant(variant_config)
        self.belief_state = BeliefState()
        self.mental_model = MentalStateTracker()
        self.probe_count = 0
        self.max_probes = variant_config.max_probes

    def _build_query(self, goal=None):
        """Build EARL-style autonomous prompt"""
        if goal is None:  # Autonomous mode
            return self.prompt_template.format(
                previous_actions=self._format_history(),
                threshold=self.confidence_threshold,
                max_probes=self.max_probes,
                belief_state=self.belief_state.to_string()
            )
        return super()._build_query(goal)

    def _update_mental_model(self, observation, action_result):
        """Update Theory of Mind model"""
        self.mental_model.update(
            ui_state=observation,
            action_taken=self.last_action,
            result=action_result
        )

        # Update belief state using inverse planning
        self.belief_state = self._inverse_plan(
            self.mental_model.trajectory,
            self.current_ui_state
        )

    def _inverse_plan(self, trajectory, ui_state):
        """Infer most likely goal from observed actions"""
        hypotheses = self._generate_hypotheses(ui_state)

        for hypothesis in hypotheses:
            # P(goal | actions) ∝ P(actions | goal) × P(goal)
            likelihood = self._action_likelihood(trajectory, hypothesis)
            prior = self._goal_prior(hypothesis, ui_state)
            hypothesis.confidence = likelihood * prior

        # Normalize confidences
        total = sum(h.confidence for h in hypotheses)
        for h in hypotheses:
            h.confidence /= total

        return BeliefState(hypotheses)
```

### 5.2 Belief State Management

```python
@dataclass
class BeliefState:
    hypotheses: List[GoalHypothesis]
    top_goal: str
    confidence: float
    evidence: List[str]
    probe_history: List[str]

    def to_string(self) -> str:
        """Compact string representation for prompt"""
        lines = []
        for i, h in enumerate(self.hypotheses[:3]):
            lines.append(f"{i+1}. {h.goal}: {h.confidence:.2f}")
        return "\n".join(lines)

    def should_probe(self, threshold: float) -> bool:
        """Determine if probing is needed"""
        return self.confidence < threshold

    def get_probe_action(self) -> str:
        """Select informative probe"""
        # Choose probe that maximizes information gain
        return self._max_info_gain_probe()
```

## 6. Evaluation Protocol

### 6.1 EARL-Specific Metrics

**Time-to-Intent (TTI)**
```
TTI = Steps until correct goal reaches confidence threshold
```

**Goal Stability (GS)**
```
GS = 1 - (goal_flips / total_steps)
```

**Probe Utility (PU)**
```
PU = Σ(confidence_delta_after_probe) / num_probes
```

**Mental Model Accuracy (MMA)**
```
MMA = Similarity(inferred_mental_state, ground_truth_state)
```

### 6.2 Comparative Analysis

Compare EARL variants against:
1. Direct prompt engineering variants (D1-D4, P1-P3)
2. Original instructed UI-Venus (upper bound)
3. Random action baseline (lower bound)

### 6.3 Ablation Studies

Test impact of:
- ToM reasoning (with/without mental state modeling)
- Affordance priors (with/without pattern library)
- Probe budget (0, 1, 2, 3 probes)
- Confidence thresholds (0.3 to 0.9)

## 7. Expected Advantages of EARL Approach

### 7.1 Theoretical Benefits

1. **Principled Reasoning**: Grounded in cognitive science theory
2. **Early Recognition**: Faster intent inference from partial sequences
3. **Uncertainty Handling**: Explicit belief tracking and confidence
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
- Risky actions from wrong goals → High confidence thresholds
- Irreversible actions → Require explicit confirmation

## 9. Integration with Experiment 2-4

### 9.1 Experiment 2: ACE Evolution
Use best EARL variant as seed for automated prompt evolution

### 9.2 Experiment 3: Multi-modal Context
Enhance mental state with app metadata and user state

### 9.3 Experiment 4: DPO Fine-tuning
Generate preference pairs from belief state trajectories

## 10. Conclusion

The EARL-inspired approach provides a theoretically grounded alternative to direct prompt engineering for autonomous UI navigation. By incorporating Theory of Mind reasoning and inverse planning, the system can infer user intent from minimal action sequences while maintaining interpretable belief states.

Key differentiators:
- **Theory-driven**: Based on cognitive science principles
- **Belief-based**: Explicit hypothesis tracking
- **Adaptive**: Dynamic confidence thresholds
- **Interpretable**: Clear mental state reasoning

This approach bridges the gap between instruction-following and full autonomy by modeling what users want based on psychological principles of goal-directed behavior.