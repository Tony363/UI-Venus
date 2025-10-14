# Research Proposal: Investigating the Limits of In-Context Learning for Autonomous GUI Task Completion

## 1. Introduction
The recent development of UI-Venus has established a new state-of-the-art in GUI agents, demonstrating exceptional performance in UI grounding and navigation tasks. UI-Venus-Navi, built upon the Qwen2.5-VL model, excels at executing multi-step tasks when provided with an explicit user instruction and a screenshot. However, this proposal pivots from instruction-following to investigate a more fundamental question: To what extent can complex, autonomous behaviors be induced in a powerful, pre-trained model without architectural changes or fine-tuning?

The challenge of creating agents that can proactively assist users by inferring intent from context—formally defined as the inverse problem of UI automation—is a well-established area of research with roots in broader studies of goal inference in fields like robotics and web search. Existing approaches often rely on bespoke inference-time algorithms (e.g., EARL) or modular pipelines that first infer a goal and then pass it to an execution agent. Concurrently, the state-of-the-art in general computer control has moved towards complex, multi-component agentic frameworks (e.g., Agent-S, AutoGLM) that utilize sophisticated techniques like hierarchical planning and reinforcement learning to achieve open-ended, workflow-level autonomy.

This proposal outlines a different, more minimalist paradigm. We will explore whether a frozen, off-the-shelf agent like UI-Venus can be repurposed to perform goal inference and autonomous action for single tasks—a capability we term instruction-less task completion—solely through advanced context engineering. Our core hypothesis is that the sophisticated reasoning required for this task can be unlocked via the manipulation of the model's input context alone. This reframes the objective from building a new state-of-the-art agent to conducting a rigorous scientific investigation into the limits and emergent properties of in-context learning, aiming to characterize the crucial trade-off between the efficiency of a minimalist approach and the raw performance of more complex architectures.

## 2. Research Question and Proposed Approach
The central research question is: Can a large, instruction-following vision-language model be coerced into performing zero-shot goal inference and subsequent autonomous action solely through the manipulation of its input context, and how does this minimalist, context-driven approach compare in performance and efficiency to methods that rely on bespoke inference algorithms, modular pipelines, or complex agentic frameworks?

To investigate this, we propose to extend the UI-Venus-Navi model by shifting its operational paradigm. Instead of providing a task-specific `{problem}` input, we will provide a static, goal-oriented system prompt that instructs the model to infer user intent and autonomously decide the next action based on the visual context and action history. The core model architecture and weights will initially remain unchanged, focusing our investigation on the power of input context to unlock new abilities in a frozen model.

## 3. Proposed Experiments and Methodology
To test our hypothesis, we will conduct a series of experiments designed to systematically enhance the autonomous capabilities of the pre-trained UI-Venus-Navi model.

### 3.1 Experiment 1: Establishing an Autonomous Baseline via System Prompt Engineering
- **Objective:** Determine the most effective system prompt for eliciting autonomous, goal-oriented behavior from the stock UI-Venus-Navi model.
- **Methodology:**
  - Design a set of candidate system prompts grounded in established HCI principles of user goal modeling. Prompts will be varied systematically along axes such as directness (e.g., “Infer and complete the user’s most likely next task”) and persona (e.g., “You are a helpful assistant...”) to explore the design space in a principled manner.
  - Evaluate each prompt using the AndroidWorld benchmark. The model will be provided with the initial screenshot and the candidate system prompt, while the explicit task goal will be withheld.
  - Operate the model autonomously and select the prompt yielding the highest Success Rate (SR) as the baseline for subsequent experiments.

### 3.2 Experiment 2: Automated Context Evolution for Zero-Shot Performance Boost
- **Objective:** Investigate whether Agentic Context Engineering (ACE) can automatically evolve a more effective system prompt, improving autonomous performance without model fine-tuning.
- **Methodology:**
  - Use the best-performing system prompt from Experiment 1 as a seed and apply an evolutionary algorithm as described in the ACE framework.
  - Define the fitness function as the model’s Success Rate on a dedicated subset of AndroidWorld benchmark tasks to test whether algorithmically evolving the prompt guides the model’s internal reasoning more effectively than manual design.

### 3.3 Experiment 3: Guidance via Multi-modal Context Encoding
- **Objective:** Explore whether enriching the input prompt with textualized, non-visual information can further improve the accuracy of autonomous action prediction.
- **Methodology:**
  - Identify and encode supplementary data such as application metadata (e.g., `app_category: "finance"`) or user state (e.g., `user_idle_for: "5 minutes"`).
  - Integrate this information into the evolved system prompt from Experiment 2 and evaluate performance to analyze whether this additional information helps the model disambiguate user intent.

### 3.4 Experiment 4: Niche Task Adaptation via Comparative Preference Optimization
- **Objective:** Investigate and compare the effectiveness of supervised versus unsupervised data generation methods for fine-tuning an autonomous agent for a specialized domain using Direct Preference Optimization (DPO).
- **Methodology:**
  - Define a niche, out-of-distribution domain (e.g., professional creative software).
  - **Approach A (Supervised):** A human labeler creates preference pairs by labeling one of several generated autonomous trajectories as “chosen” and another as “rejected”.
  - **Approach B (Unsupervised):** Implement an automated method inspired by WEPO to generate preference pairs without manual labeling, using heuristics to sample less salient UI elements as “rejected” actions.
  - Use both datasets to fine-tune two separate models via DPO and directly compare their performance, data efficiency, and scalability on the niche domain.

## 4. Evaluation
Our evaluation protocol is designed to rigorously test our central hypothesis by comparing our context-driven methods against strong, established baselines and employing nuanced metrics on diverse benchmarks.

### 4.1 Baselines for Comparison
- **Modular “Infer-then-Act” Pipeline:** A two-stage baseline where a state-of-the-art LMM first infers a natural language goal, which is then fed as a standard prompt to the original UI-Venus-Navi agent. This tests the efficacy of a modular pipeline against our end-to-end approach.
- **Inference-Time Algorithm (EARL):** Implement the EARL algorithm, a state-of-the-art zero-shot baseline designed specifically for inferring latent goals from partial actions, providing a direct comparison between our prompt-engineering method and a dedicated inference algorithm.
- **State-of-the-Art Agentic Framework (SOTA Reference):** Include a leading agentic framework like Agent-S as a third baseline. This system, with its complex hierarchical planning and memory architecture, serves as an upper-bound performance reference and enables a nuanced discussion of performance-versus-efficiency trade-offs.

### 4.2 Benchmarks and Metrics
- **Primary Benchmark (Mobile):** Use the established AndroidWorld benchmark to ensure a direct comparison with the original UI-Venus-Navi’s instructed performance (65.9% success rate).
- **Generalizability Benchmark (Desktop):** Evaluate the best-performing agent on the UI-Vision benchmark, which captures the complexity of professional software with high-density UI elements and specialized interactions, to test whether the approach extends beyond constrained mobile interfaces.
- **Refined Evaluation Metrics:**
  - Task Completion (Success Rate): Measure the rate of autonomous task completion without a ground-truth goal.
  - Goal Inference Quality: Compare the agent’s implicit goal (extracted from its think process) against the ground-truth goal for semantic similarity to decouple inference from execution failure.

### 4.3 Comparative Analysis
Directly compare the autonomous success rates and goal inference quality scores from the experiments against the Modular baseline, EARL, and the SOTA reference framework. This analysis will demonstrate the efficacy and efficiency of the context-engineering paradigm relative to other established methods for solving the goal inference problem.

