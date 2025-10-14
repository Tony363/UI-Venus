#!/bin/bash
set -e

# Optimized script for UI-Venus-Ground-7B with memory improvements
# This script includes memory optimization settings to prevent OOM errors

echo "Setting up environment for optimized memory usage..."

# Set PyTorch memory configuration to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Clear GPU cache before running
echo "Clearing any existing GPU cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Monitor GPU memory (optional - comment out if not needed)
echo "Current GPU memory status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits || true

echo "Starting evaluation with optimized settings..."

models=("ui_venus_ground_7b")
for model in "${models[@]}"
do
    echo "Running evaluation for model: ${model}"

    # Run ScreenSpot-v2 evaluation
    python models/grounding/eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "ScreenSpot-v2-variants/images"  \
        --screenspot_test "ScreenSpot-v2-variants/annotations"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-7B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "venus_7b/venus_7b_ss2_optimized.json" \
        --inst_style "instruction"

    echo "ScreenSpot-v2 evaluation completed."

    # Clear cache between evaluations
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Run ScreenSpot-pro evaluation
    python models/grounding/eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "Screenspot-pro/images"  \
        --screenspot_test "Screenspot-pro/annotations"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-7B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "venus_7b/venus_7b_pro_optimized.json" \
        --inst_style "instruction"

    echo "ScreenSpot-pro evaluation completed."
done

echo "All evaluations completed successfully!"
echo "Results saved with '_optimized' suffix in the log paths."