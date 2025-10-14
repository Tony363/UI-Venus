# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

UI-Venus is a state-of-the-art visual grounding and UI navigation model that uses Reinforcement Fine-Tuning (RFT) for GUI understanding across mobile, desktop, and web interfaces. The repository contains evaluation code for two model variants (7B and 72B) for both grounding and navigation tasks.

## Development Environment

### Prerequisites
```bash
# Install dependencies (Python 3.10+, CUDA-enabled PyTorch recommended)
pip install -r requirements.txt

# Validate environment setup
python validate_setup.py

# Check GPU setup for model inference
python tests/test_gpu_setup.py
```

### Model Configuration
The repository uses Hugging Face models that must be configured with proper paths:
- **Grounding Models**: `inclusionAI/UI-Venus-Ground-7B` and `inclusionAI/UI-Venus-Ground-72B`
- **Navigation Models**: `inclusionAI/UI-Venus-Navi-7B` and `inclusionAI/UI-Venus-Navi-72B`

Set environment variables for Hugging Face:
```bash
export HF_HOME=<your_cache_directory>
export HF_TOKEN=<your_token>  # If accessing private models
```

## Common Development Commands

### Running Evaluations

#### Grounding Tasks
```bash
# 7B Model Evaluation
bash scripts/run_gd_7b.sh

# 72B Model Evaluation
bash scripts/run_gd_72b.sh
```

Before running, ensure you configure these paths in the scripts:
- `screenspot_imgs`: Screenshot directory (e.g., `Screenspot-pro/images`)
- `screenspot_test`: Annotation files directory (e.g., `Screenspot-pro/annotations`)
- `model_name_or_path`: Model checkpoint path
- `log_path`: Output directory for results

#### Navigation Tasks
```bash
# 7B Model Navigation
bash scripts/run_navi_7b.sh

# 72B Model Navigation
bash scripts/run_navi_72b.sh
```

Required configuration:
- `model_path`: Path to navigation model checkpoint
- `input_file`: JSON file with navigation tasks (default: `examples/trace/trace.json`)
- `output_file`: Path for saving execution history

### Testing

```bash
# Quick smoke test for model inference
python tests/quick_start.py

# Test GPU setup and mixed precision support
python tests/test_gpu_setup.py

# Download evaluation datasets (if needed)
python tests/download_datasets.py
```

### Single Test Execution
For debugging specific functionality:
```python
# Test grounding on a single image
from models.grounding.ui_venus_ground_7b import UI_Venus_Ground_7B
model = UI_Venus_Ground_7B()
model.load_model("inclusionAI/UI-Venus-Ground-7B")
result = model.inference("click the search button", "path/to/screenshot.png")
```

## Code Architecture

### Two-Model System Architecture

The repository implements two distinct model types that work in tandem:

1. **Grounding Models** (`models/grounding/`)
   - `ui_venus_ground_7b.py` / `ui_venus_ground_72b.py`: Core grounding implementations
   - `eval_screenspot_pro.py`: Unified evaluation driver for both model sizes
   - Processes visual elements and returns bounding boxes in [x1,y1,x2,y2] format
   - Normalizes coordinates relative to input dimensions

2. **Navigation Models** (`models/navigation/`)
   - `ui_venus_navi_agent.py`: Main agent class managing action history and state
   - `ui_venus_navi_vllm.py`: VLLM-accelerated inference backend
   - `runner.py`: Orchestration layer for trace execution
   - `utils.py`: Shared utilities for prompt formatting and action parsing
   - Supports 14 action types including click, drag, scroll, type, and workflow control

### Data Flow Pipeline

1. **Input Processing**: Screenshots are resized based on min/max pixel constraints (2M-4.8M for grounding)
2. **Model Inference**: Uses Qwen2.5-VL architecture with specialized prompting
3. **Coordinate Transformation**: Converts model outputs to normalized coordinates
4. **Action Execution**: Navigation agent maintains history and executes sequential actions

### Key Design Patterns

- **Dataclass-based State**: Uses `@dataclass` for structured agent state management
- **Chain-of-Thought**: Navigation models output `<think>` and `<action>` tags for interpretability
- **History Management**: Configurable context window for previous actions in navigation
- **Error Recovery**: Graceful fallback to [0,0,0,0] for invalid grounding outputs

## Dataset Organization

- `Screenspot-pro/`: Professional UI grounding benchmark with domain-specific categories
- `ScreenSpot-v2-variants/`: Multi-platform grounding datasets (mobile, desktop, web)
- `data/`: Additional evaluation datasets (osworld, ui_vision, CAGUI)
- `examples/`: JSON format templates for input/output specifications
- `venus_7b/`: Sample evaluation outputs for reference

## Important Conventions

### Coding Standards
- Follow PEP 8 with 4-space indentation
- Use snake_case for functions, PascalCase for classes
- Maintain type hints and dataclasses for agent state
- Use structured logging via `models/.../utils.py` instead of print statements

### Path Configuration
When modifying evaluation scripts, always use absolute paths and document required overrides inline. Never hardcode credentials or checkpoint paths directly in code.

### Model Inference Settings
- Default temperature: 0.0 (deterministic)
- Max new tokens: 128 for grounding, 2048 for navigation
- Attention implementation: "eager" for 7B, "flash_attention_2" optional

### Testing New Features
- Place new tests in `tests/` with naming pattern `test_<feature>.py`
- Gate GPU-intensive operations behind environment flags
- Keep test fixtures lightweight under `examples/`
- Document required artifacts in docstrings

## Performance Optimization

- Use VLLM backend for navigation models to improve throughput
- Batch processing supported via array inputs to processor
- Configure `min_pixels` and `max_pixels` based on GPU memory
- Enable `flash_attention_2` for supported hardware

## Debugging Tips

1. **Model Loading Issues**: Check CUDA availability and bfloat16 support
2. **Coordinate Misalignment**: Verify image grid dimensions match preprocessing
3. **Navigation Failures**: Inspect action history in agent's `StepData` records
4. **Memory Issues**: Reduce `max_pixels` or use smaller batch sizes