# Experiment 1: Testing Protocol for Autonomous System Prompts

## Executive Summary

This document defines the comprehensive testing methodology for evaluating 12+ system prompt variations that transform UI-Venus from instruction-following to autonomous goal-inferring behavior.

## 1. Experimental Design

### 1.1 Study Structure
- **Design**: 12×K factorial (12 prompt variants × K scenarios)
- **Replications**: 30+ episodes per variant per scenario
- **Total Episodes**: 12 × K × 30 = 360K episodes minimum
- **Control Condition**: Original UI-Venus with explicit instructions

### 1.2 Independent Variables
1. **Directness** (D1-D4): Implicit → Explicit goal inference
2. **Persona** (P1-P3): Cautious → Balanced → Proactive
3. **Verbosity** (V1-V2): Minimal → Detailed planning
4. **History** (H0-H1): Current only → Context-aware

### 1.3 Dependent Variables
- Task Success Rate (binary)
- Goal Inference Accuracy (semantic similarity)
- Action Efficiency (steps to completion)
- Error Recovery Rate
- Confidence Calibration (predicted vs actual)

## 2. AndroidWorld Benchmark Adaptation

### 2.1 Task Selection Criteria
Select scenarios with clear dominant intents:

**Tier 1: Unambiguous (High Agreement)**
- Permission dialogs (Allow/Deny)
- Error messages requiring dismissal
- Login screens with visible fields
- Single-CTA screens ("Get Started")

**Tier 2: Moderate Ambiguity**
- Settings screens with multiple toggles
- Search interfaces with suggestions
- Multi-step forms
- Navigation menus

**Tier 3: High Ambiguity (Multiple Valid Goals)**
- Home screens with multiple apps
- Content browsing interfaces
- Multi-tab applications
- Dashboard views

### 2.2 Instruction Withholding Protocol
```python
def prepare_autonomous_episode(original_task):
    """Convert instructed task to autonomous evaluation"""
    return {
        'screenshot': original_task.initial_screenshot,
        'ground_truth_goal': original_task.instruction,  # Hidden
        'allowed_goals': original_task.valid_alternatives,  # For ambiguous cases
        'success_criteria': original_task.completion_check,
        'max_steps': original_task.step_limit,
        'variant_id': selected_variant
    }
```

### 2.3 Success Evaluation
```python
def evaluate_success(episode_result):
    """Multi-criteria success evaluation"""
    return {
        'task_completed': check_completion(episode_result),
        'goal_match': compare_goals(
            episode_result.inferred_goal,
            episode_result.ground_truth_goal
        ),
        'alternative_success': check_allowed_alternatives(episode_result),
        'efficiency_score': calculate_efficiency(episode_result),
        'recovery_occurred': detect_recovery_pattern(episode_result)
    }
```

## 3. Metrics and Scoring

### 3.1 Primary Metrics

**Task Success Rate (TSR)**
```
TSR = (Completed Episodes) / (Total Episodes)
```

**Goal Inference Accuracy (GIA)**
```
GIA = SemanticSimilarity(Inferred Goal, Ground Truth Goal)
```
Using embedding similarity or keyword overlap.

**Action Efficiency (AE)**
```
AE = (Optimal Steps) / (Actual Steps)
```

### 3.2 Secondary Metrics

**Wrong Goal Rate (WGR)**
```
WGR = (Incorrect Goal Episodes) / (Total Episodes)
```

**Recovery Rate (RR)**
```
RR = (Corrected Wrong Goals) / (Wrong Goal Episodes)
```

**Exploration Efficiency (EE)**
```
EE = (Information Gain) / (Exploration Actions)
```

**Confidence Calibration Error (ECE)**
```
ECE = Σ|confidence - accuracy| × bin_weight
```

### 3.3 Composite Score
```python
def calculate_composite_score(metrics):
    """Weighted composite for variant ranking"""
    weights = {
        'task_success': 0.4,
        'goal_accuracy': 0.3,
        'efficiency': 0.2,
        'recovery': 0.1
    }
    return sum(metrics[k] * weights[k] for k in weights)
```

## 4. Testing Phases

### 4.1 Phase 1: Pilot Study (Week 1)
- **Scope**: 3 scenarios × 12 variants × 5 episodes
- **Goals**: Validate prompts, tune thresholds, identify failures
- **Output**: Refined prompts, calibrated thresholds

### 4.2 Phase 2: Main Experiment (Weeks 2-3)
- **Scope**: Full K scenarios × 12 variants × 30 episodes
- **Goals**: Complete data collection
- **Output**: Raw performance data

### 4.3 Phase 3: Analysis (Week 4)
- **Statistical Testing**: ANOVA, post-hoc comparisons
- **Effect Sizes**: Cohen's d for pairwise comparisons
- **Ablation**: Isolate impact of each dimension

## 5. Implementation Details

### 5.1 Agent Modifications
```python
class AutonomousVenusAgent(VenusNaviAgent):
    def __init__(self, variant_config):
        super().__init__(model_config)
        self.prompt_template = load_variant(variant_config.variant_id)
        self.confidence_threshold = variant_config.threshold
        self.max_probes = variant_config.probe_limit
        self.belief_state = BeliefState()

    def _build_query(self, goal=None):
        """Override to use autonomous prompt"""
        if goal is None:  # Autonomous mode
            return self.prompt_template.format(
                previous_actions=self._format_history(),
                ui_elements_summary=self._get_ui_summary()  # If enhanced
            )
        return super()._build_query(goal)

    def _update_belief(self, observation):
        """Track goal hypothesis (not sent to environment)"""
        self.belief_state.update(observation)
        self.logger.info(f"Belief: {self.belief_state}")
```

### 5.2 Logging Schema
```json
{
    "episode_id": "uuid",
    "variant_id": "D3_P2_V2_H1",
    "scenario": "login_screen",
    "timestamp": "2024-01-15T10:00:00Z",
    "steps": [
        {
            "step_num": 1,
            "screenshot": "base64...",
            "belief_state": {
                "goal_hypothesis": "user wants to login",
                "evidence": ["login button visible", "username field empty"],
                "confidence": 0.75,
                "success_criteria": "reach home screen"
            },
            "action": {
                "type": "Click",
                "params": {"box": [100, 200]}
            },
            "result": "username_field_focused"
        }
    ],
    "outcome": {
        "task_completed": true,
        "goal_matched": true,
        "steps_taken": 5,
        "time_elapsed": 15.2
    }
}
```

### 5.3 Confidence Thresholds by Persona
```python
CONFIDENCE_THRESHOLDS = {
    'P1': 0.7,  # Cautious
    'P2': 0.5,  # Balanced
    'P3': 0.35  # Proactive
}

PROBE_LIMITS = {
    'P1': 1,
    'P2': 2,
    'P3': 3
}
```

## 6. Statistical Analysis Plan

### 6.1 Hypothesis Testing
**H1**: D3 (Moderate) > D1 (Implicit) on task success
**H2**: P2 (Balanced) achieves best efficiency-accuracy trade-off
**H3**: H1 (History) > H0 (No History) on multi-step tasks
**H4**: V2 (Detailed) > V1 (Minimal) on complex scenarios

### 6.2 Analysis Methods
1. **Two-way ANOVA**: Main effects and interactions
2. **Tukey HSD**: Post-hoc pairwise comparisons
3. **Bootstrap CIs**: Non-parametric confidence intervals
4. **Regression**: Predict success from variant features

### 6.3 Sample Size Calculation
```python
from statsmodels.stats.power import TTestPower

power_analysis = TTestPower()
n = power_analysis.solve_power(
    effect_size=0.5,  # Medium effect
    power=0.8,
    alpha=0.05
)
# Result: ~30 episodes per condition
```

## 7. Quality Control

### 7.1 Data Validation
- Verify action schema compliance
- Check screenshot capture quality
- Validate JSON logging format
- Ensure random seed reproducibility

### 7.2 Failure Analysis
Track and categorize failures:
- Goal inference failures
- Action execution errors
- Timeout/infinite loops
- Invalid action attempts

### 7.3 Edge Case Handling
- Modal dialog interruptions
- Network errors
- App crashes
- Permission denials

## 8. Evaluation Schedule

### Week 1: Setup and Pilot
- Day 1-2: Environment setup, variant implementation
- Day 3-4: Pilot execution (180 episodes)
- Day 5: Analysis and refinement

### Week 2: Main Experiment Part 1
- Day 1-5: Execute 50% of main episodes
- Continuous monitoring and logging

### Week 3: Main Experiment Part 2
- Day 1-3: Complete remaining episodes
- Day 4-5: Data cleaning and validation

### Week 4: Analysis and Reporting
- Day 1-2: Statistical analysis
- Day 3-4: Visualization and interpretation
- Day 5: Report generation

## 9. Expected Outcomes and Decision Criteria

### 9.1 Success Thresholds
- **Minimum Viable**: One variant achieves >40% autonomous success
- **Target**: Best variant achieves >50% success
- **Stretch**: Best variant within 20% of instructed baseline

### 9.2 Go/No-Go Decisions
- **Proceed to Experiment 2**: If success threshold met
- **Iterate Prompts**: If close but not meeting threshold
- **Pivot Approach**: If fundamental limitations identified

### 9.3 Key Insights Expected
1. Optimal directness level for goal inference
2. Best persona for accuracy-efficiency balance
3. Value of history context
4. Impact of planning verbosity

## 10. Resource Requirements

### 10.1 Computational
- GPU: V100 or better for model inference
- Storage: ~500GB for screenshots and logs
- Parallel Execution: 4-8 concurrent episodes

### 10.2 Time Estimates
```
Per Episode: ~2-3 minutes
Total Episodes: 360K minimum
Sequential Time: 900 hours
With 8x Parallel: ~112 hours (~5 days)
```

### 10.3 Human Resources
- Initial setup: 2 person-days
- Monitoring: 1 hour/day during execution
- Analysis: 3 person-days

## 11. Risk Mitigation

### 11.1 Technical Risks
- **Model API limits**: Implement backoff and queuing
- **Storage overflow**: Incremental backup, compression
- **Crash recovery**: Checkpoint every 100 episodes

### 11.2 Experimental Risks
- **Variant bugs**: Pilot testing, syntax validation
- **Ambiguity handling**: Pre-define allowed alternatives
- **Data quality**: Automated validation checks

## 12. Deliverables

### 12.1 Outputs
1. Performance matrix (12 variants × K scenarios)
2. Statistical analysis report
3. Best variant recommendation
4. Failure pattern analysis
5. Confidence calibration curves

### 12.2 Next Steps
Based on results:
- Select top 3 variants for Experiment 2 (ACE)
- Identify prompt features for further optimization
- Design specialized prompts for failure categories
- Plan architectural enhancements if needed