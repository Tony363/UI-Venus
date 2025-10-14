# Experiment 1: Prompt Variation Strategy

## Overview

This document details the systematic prompt engineering strategy for Experiment 1, defining the specific prompt templates, variation axes, and design rationale for each of the 18 core variants to be tested.

## Design Philosophy

### Core Principles
1. **Minimalism First**: Start with the least information necessary and add complexity only where needed
2. **Behavioral Diversity**: Test fundamentally different approaches to goal inference
3. **Systematic Coverage**: Ensure variations span the theoretical space of prompt design
4. **Practical Grounding**: Base designs on established HCI and autonomous agent principles

## Prompt Architecture

### Base Template Structure
```
[ROLE_FRAMING]
[OBJECTIVE_DESCRIPTION]
[TOOL_DOCUMENTATION]
[OUTPUT_FORMAT_CONTRACT]
[TERMINATION_POLICY]
[OPTIONAL_K_SHOT_EXAMPLES]
[ERROR_RECOVERY_GUIDANCE]
```

## Variation Axes Detailed

### 1. Role Framing Variants

#### Navigator (Technical Expert)
```
You are a mobile UI navigator with expertise in understanding application interfaces and user interaction patterns.
```
- **Rationale**: Positions agent as technical expert focused on UI mechanics
- **Expected Behavior**: Systematic, methodical navigation
- **Hypothesis**: Will excel at complex multi-step tasks

#### Executor (Task-Oriented)
```
You are a task execution agent designed to complete user goals efficiently based on visual context.
```
- **Rationale**: Emphasizes goal completion over exploration
- **Expected Behavior**: Direct, efficient action sequences
- **Hypothesis**: Better for simple, clear tasks

#### Assistant (User-Centric)
```
You are a helpful assistant that understands what users typically want to accomplish in mobile applications.
```
- **Rationale**: Humanizes the agent, emphasizes user intent understanding
- **Expected Behavior**: More cautious, user-considerate actions
- **Hypothesis**: Better at ambiguous intent scenarios

### 2. Objective Style Variants

#### Concise
```
Analyze the screen and take the next logical action based on apparent user intent.
```
- **Rationale**: Minimal guidance, maximum inference requirement
- **Expected Behavior**: Relies heavily on implicit knowledge

#### Constraints-Based
```
Given the current screen state, identify the most likely user goal considering:
- Common app usage patterns
- Visual affordances and UI hints
- Incomplete or in-progress workflows
Then execute appropriate actions to achieve that goal.
```
- **Rationale**: Provides structured thinking framework
- **Expected Behavior**: More systematic goal inference

#### Verbose
```
Your task is to act as an autonomous agent that can understand user intentions from visual context alone.
Examine the current application screen carefully, considering all visible elements, their states, and relationships.
Infer what a typical user would want to accomplish given this context, taking into account common interaction
patterns, UI conventions, and logical next steps in standard workflows. Once you've identified the most probable
user intent, execute the appropriate sequence of actions to fulfill that goal.
```
- **Rationale**: Maximum context and guidance
- **Expected Behavior**: Thorough analysis before action

### 3. Tool Documentation Verbosity

#### Terse
```
Actions: Click(box), Drag(start,end), Scroll(start,end,dir), Type(text), Launch(app),
Wait(), Finished(text), CallUser(text), LongPress(box), PressBack/Home/Enter/Recent()
```

#### Moderate
```
Available Actions:
- Click(box=(x,y)): Tap at coordinates
- Drag(start=(x1,y1), end=(x2,y2)): Swipe between points
- Scroll(start=(x,y), end=(x,y), direction='up/down/left/right'): Scroll content
- Type(content='text'): Enter text in focused field
- Launch(app='name'): Open application
- Wait(): Pause for loading
- Finished(content='result'): Complete task with output
- CallUser(content='message'): Communicate with user
- Navigation: PressBack(), PressHome(), PressEnter(), PressRecent()
```

#### Verbose
```
Available Actions (use exactly as specified):

Click(box=(x, y))
  Purpose: Tap on UI elements like buttons, links, or fields
  Parameters: box - tuple of (x, y) coordinates
  Example: Click(box=(150, 300)) to tap a button at that location

Drag(start=(x1, y1), end=(x2, y2))
  Purpose: Swipe gestures for scrolling, sliding, or dragging elements
  Parameters: start - beginning coordinates, end - ending coordinates
  Example: Drag(start=(100, 500), end=(100, 200)) to scroll up

[... detailed documentation for all 14 actions ...]
```

### 4. K-Shot Examples

#### 0-Shot (No Examples)
No examples provided, pure zero-shot inference.

#### 1-Shot (Single Example)
```
Example:
Screen: Messaging app showing conversation list with unread badge
<think>User likely wants to read new messages, unread indicator visible</think>
<action>Click(box=(150, 200))</action>
<conclusion></conclusion>
```

#### 3-Shot (Multiple Examples)
Includes diverse examples:
1. Reading notification
2. Completing form submission
3. Navigating to settings

### 5. Error Recovery Guidance

#### None
No error handling instructions provided.

#### Brief
```
If an action fails, try alternative approaches to achieve the goal.
```

#### Explicit
```
Error Recovery Protocol:
- If click fails: Check if element moved, try nearby coordinates
- If typing fails: Ensure field is focused first
- If navigation blocked: Use alternative paths (back button, menu)
- After 2 failed attempts: Try different interpretation of user intent
```

### 6. Termination Policy

#### Strict
```
Use Finished(content) ONLY when you have concrete evidence the task is complete.
Never terminate early. Verify success through visual confirmation.
```

#### Soft
```
Complete the task when you believe the user's goal has been achieved.
Use Finished(content) to provide results or confirm completion.
```

#### Adaptive
```
Termination signals:
- Success indicators visible on screen
- Workflow reaches natural endpoint
- No more productive actions available
- User goal appears satisfied based on context
```

## The 18 Core Variants

### Variant Design Matrix

| ID | Variant Name | Design Philosophy | Key Hypothesis |
|----|--------------|-------------------|----------------|
| v01 | Minimal Baseline | Absolute minimum information | Tests pure emergent behavior |
| v02 | Guided Navigator | Balanced navigation focus | Expected strong performer |
| v03 | Constrained Navigator | Structured thinking + flexibility | Best for complex tasks |
| v04 | Minimal Executor | Task-focused minimalism | Efficient for clear goals |
| v05 | Robust Executor | Executor + error handling | Resilient to failures |
| v06 | Full Executor | Maximum executor guidance | Handling edge cases |
| v07 | Balanced Assistant | User-centric with guidance | Good for ambiguous intent |
| v08 | Adaptive Assistant | Flexible user-focused | Handles uncertainty well |
| v09 | Maximum Navigator | All features for navigator | Upper bound performance |
| v10 | Simple Navigator Plus | Minimal + 1-shot | Tests example impact |
| v11 | Structured Executor | Constraints without examples | Pure reasoning test |
| v12 | Minimal Assistant | Baseline for assistant role | Tests role impact |
| v13 | Smart Navigator | Best practices combination | Production candidate |
| v14 | Verbose Executor | Detailed executor variant | Tests verbosity impact |
| v15 | Explicit Assistant | Maximum clarity | Reduces ambiguity |
| v16 | Adaptive Navigator | Flexible navigation | Handles variety |
| v17 | Lean Executor | Efficient task completion | Speed optimized |
| v18 | Comprehensive Assistant | Full-featured assistant | Maximum capability |

## Detailed Variant Specifications

### v01: Minimal Baseline
```yaml
role: navigator
objective: concise
tools: terse
k_shot: 0
recovery: none
termination: strict

prompt: |
  You are a mobile UI navigator.
  Analyze the screen and take the next logical action based on apparent user intent.
  Actions: Click(box), Drag(start,end), Scroll(start,end,dir), Type(text), Launch(app),
  Wait(), Finished(text), CallUser(text), LongPress(box), PressBack/Home/Enter/Recent()

  Output format:
  <think>reasoning</think>
  <action>single action</action>
  <conclusion>when done</conclusion>
```

### v02: Guided Navigator
```yaml
role: navigator
objective: concise
tools: moderate
k_shot: 1
recovery: brief
termination: strict

prompt: |
  You are a mobile UI navigator with expertise in understanding application interfaces.
  Analyze the screen and take the next logical action based on apparent user intent.

  Available Actions:
  - Click(box=(x,y)): Tap at coordinates
  - Drag(start=(x1,y1), end=(x2,y2)): Swipe between points
  [... moderate tool documentation ...]

  Example:
  Screen: Messaging app with unread badge
  <think>User wants to read new messages</think>
  <action>Click(box=(150, 200))</action>
  <conclusion></conclusion>

  If an action fails, try alternative approaches.

  Use Finished() only with concrete evidence of completion.
```

### v03: Constrained Navigator
```yaml
role: navigator
objective: constraints
tools: moderate
k_shot: 1
recovery: brief
termination: soft

prompt: |
  You are a mobile UI navigator.

  Given the current screen, identify the most likely user goal considering:
  - Common app usage patterns
  - Visual affordances and UI hints
  - Incomplete or in-progress workflows
  Then execute appropriate actions to achieve that goal.

  [moderate tools + 1-shot example + brief recovery + soft termination]
```

[... Specifications continue for all 18 variants ...]

## Prompt Quality Criteria

### Must Have
- Clear output format specification
- Valid action syntax examples
- Unambiguous role definition

### Should Have
- Reasoning guidance
- Error handling consideration
- Termination criteria

### Nice to Have
- Diverse examples
- Progressive complexity
- Adaptive strategies

## Testing Protocol

### Phase 1: Syntax Validation
- Verify all prompts parse correctly
- Ensure action format is consistent
- Validate example correctness

### Phase 2: Behavioral Testing
- Run each variant on 5 canonical tasks
- Verify basic functionality
- Check for systematic failures

### Phase 3: Full Evaluation
- Complete benchmark run
- Statistical analysis
- Performance ranking

## Prompt Evolution Strategy

### After Initial Results:
1. **Identify top 5 performers**
2. **Analyze success patterns**
3. **Create 5 refined variants**
4. **Test refinements**

### Refinement Techniques:
- Combine successful elements
- Adjust verbosity based on performance
- Tune example selection
- Optimize role framing

## Expected Outcomes

### Performance Distribution
- **High Performers (>40% SR)**: v02, v03, v07, v13
- **Moderate (25-40% SR)**: v05, v08, v09, v16
- **Low (<25% SR)**: v01, v12, v04
- **Unknown**: Others

### Key Insights Expected
1. Role framing significantly impacts behavior
2. 1-shot examples provide major improvement
3. Error recovery essential for robustness
4. Moderate verbosity optimal

## Risk Mitigation

### If All Variants Fail (<10% SR):
1. Add semi-explicit goal hints
2. Increase example count
3. Provide workflow templates

### If Parse Errors Dominate:
1. Simplify output format
2. Add format examples
3. Implement recovery parser

### If Behavioral Diversity Low:
1. Increase role contrast
2. Add persona-specific language
3. Vary thinking patterns

## Conclusion

This prompt variation strategy provides systematic coverage of the design space while maintaining scientific rigor. The 18 variants test key hypotheses about autonomous behavior induction through context engineering, setting the foundation for subsequent optimization in Experiment 2.