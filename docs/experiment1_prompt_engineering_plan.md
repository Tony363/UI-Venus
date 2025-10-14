# Experiment 1: Prompt Engineering Plan for Autonomous UI-Venus

## Executive Summary

This document outlines the systematic approach to transform UI-Venus-Navi from an instruction-following agent to an autonomous goal-inferring agent through prompt engineering alone, without architectural changes or fine-tuning.

## 1. Current State Analysis

### Original USER_PROMPT Structure (utils.py)
```python
USER_PROMPT = """
**You are a GUI Agent.**
Your task is to analyze a given user task, review current screenshot and previous actions, and determine the next action to complete the task.

### User Task
{user_task}  # <-- This explicit instruction will be removed

### Previous Actions
{previous_actions}

### Available Actions
[14 action types listed]

### Instruction
[Detailed guidelines for task execution]
"""
```

### Key Components to Preserve
- Action schema (Click, Type, Scroll, etc.)
- Think/Action/Conclusion output structure
- Screenshot analysis capability
- Action history tracking
- Safety guidelines (modal handling, copy/paste, etc.)

## 2. Transformation Strategy

### Core Challenge
Replace explicit `{user_task}` with static instructions that enable the model to:
1. Infer user intent from visual context
2. Generate appropriate goals autonomously
3. Execute actions toward inferred goals
4. Maintain accuracy without explicit guidance

### Design Principles

#### 2.1 Goal Inference Mechanism
- **Visual Cue Recognition**: Prioritize UI affordances (buttons, dialogs, errors)
- **Context Awareness**: Use action history to understand task progression
- **Confidence Gating**: Act only when goal inference confidence exceeds threshold
- **Safe Exploration**: When uncertain, perform information-gathering actions

#### 2.2 Belief State Integration
Add internal tracking (not exposed to environment):
```
goal_hypothesis: "user wants to [inferred goal]"
evidence: ["login button visible", "username field empty"]
confidence: 0.75
success_criteria: "login form submitted"
```

#### 2.3 Prompt Architecture
```
[Role Definition] → [Goal Inference Instructions] → [Context Input] →
[Action Guidelines] → [Output Format]
```

## 3. Variation Dimensions

### 3.1 Directness Axis (D1-D4)
How explicitly we instruct goal inference:

**D1 - Implicit**: No mention of goals
- "Help the user with their current task"
- Relies on implicit understanding

**D2 - Light Hint**: Gentle guidance
- "Understand what the user needs and assist them"
- Soft goal awareness

**D3 - Moderate**: Clear but not prescriptive
- "Infer the user's likely goal from the UI and act accordingly"
- Balanced approach

**D4 - Explicit**: Direct instructions
- "Analyze the UI, identify the most probable user goal, then execute it"
- Maximum clarity

### 3.2 Persona Axis (P1-P3)
Agent's behavioral characteristics:

**P1 - Cautious**: Conservative approach
- Confidence threshold: 0.7
- Max exploration: 1 probe
- Prefers certainty over speed

**P2 - Balanced**: Moderate risk-taking
- Confidence threshold: 0.5
- Max exploration: 2 probes
- Balances accuracy and efficiency

**P3 - Proactive**: Exploratory behavior
- Confidence threshold: 0.35
- Max exploration: 3 probes
- Emphasizes task completion

### 3.3 Additional Dimensions

**Planning Verbosity (V1-V2)**
- V1: Single-step reactive
- V2: Multi-step planning

**History Emphasis (H0-H1)**
- H0: Current screenshot only
- H1: Incorporate previous actions

## 4. Prompt Template Structure

### Shared Core Components
```python
AUTONOMOUS_CORE = """
**You are an autonomous GUI Assistant.**

### Available Actions
[Maintain exact action list from original]

### Output Format
[Preserve think/action/conclusion structure]

### Safety Guidelines
[Keep all safety rules from original]
"""
```

### Variable Components
```python
# Inserted based on variant configuration
GOAL_INFERENCE_SECTION = """
### Your Approach
{directness_instruction}
{persona_instruction}
{planning_instruction}
{history_instruction}
"""
```

## 5. Implementation Approach

### 5.1 Minimal Code Changes
```python
# In ui_venus_navi_agent.py
def _build_query(self, goal: str = None) -> str:
    if goal is None:  # Autonomous mode
        return AUTONOMOUS_PROMPT.format(
            previous_actions=history_str,
            # No user_task parameter
        )
    else:  # Standard mode (for comparison)
        return USER_PROMPT.format(
            user_task=goal,
            previous_actions=history_str
        )
```

### 5.2 Belief State Tracking
```python
@dataclass
class BeliefState:
    goal_hypothesis: str
    evidence: List[str]
    confidence: float
    success_criteria: str

# Add to StepData for logging (not sent to environment)
```

### 5.3 Action Validation Layer
```python
def validate_action(action_json: dict, belief_state: BeliefState) -> bool:
    # Check action aligns with inferred goal
    # Prevent destructive actions without confirmation
    # Ensure modal priority handling
    return is_valid
```

## 6. Evaluation Strategy

### 6.1 Success Metrics
- **Task Success Rate**: Completion without explicit instructions
- **Goal Inference Accuracy**: Match between inferred and ground truth
- **Action Efficiency**: Steps to completion
- **Error Recovery**: Ability to correct wrong inferences

### 6.2 AndroidWorld Adaptation
- Withhold task descriptions during evaluation
- Provide only initial screenshot and let agent infer
- Track both task completion and goal inference quality

### 6.3 Variant Comparison
- 12 base variants (4 directness × 3 personas)
- 30+ episodes per variant
- Statistical significance testing
- Effect size analysis

## 7. Expected Outcomes

### Hypotheses
1. **Primary**: D3-P2 (Moderate-Balanced) will achieve best overall performance
2. **Secondary**: Higher directness improves goal accuracy but may reduce flexibility
3. **Tertiary**: Proactive personas excel in unambiguous scenarios but struggle with ambiguity

### Risk Factors
- Ambiguous UI states with multiple valid goals
- Cascading errors from incorrect initial inference
- Over-exploration in uncertain scenarios
- Difficulty generalizing across app domains

## 8. Next Steps

1. **Phase 1**: Create all 12 prompt variants (see experiment1_prompt_variations.md)
2. **Phase 2**: Implement minimal agent modifications
3. **Phase 3**: Set up AndroidWorld evaluation harness
4. **Phase 4**: Run pilot study (3 scenarios × 12 variants)
5. **Phase 5**: Analyze results and iterate
6. **Phase 6**: Full-scale evaluation

## 9. Success Criteria

The experiment succeeds if:
- At least one variant achieves >40% success rate without instructions
- Goal inference accuracy exceeds 60%
- Performance gap vs instructed baseline is <30%
- Clear patterns emerge across variation dimensions

## 10. Theoretical Contribution

This experiment investigates:
- Limits of in-context learning for complex reasoning
- Trade-offs between explicit instruction and emergent behavior
- Minimal intervention approaches to agent capability enhancement
- Prompt engineering as alternative to architectural changes