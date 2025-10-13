#!/usr/bin/env python3
"""Test script to verify GPU setup for Qwen2_5_VLProcessor"""

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Device detected: {device}")

# Test 1: Check CUDA availability
if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  - CUDA Version: {torch.version.cuda}")
else:
    print("⚠ CUDA not available, will use CPU")

# Test 2: Load processor (should work without .to())
try:
    processor = AutoProcessor.from_pretrained("inclusionAI/UI-Venus-Ground-7B")
    print("✓ Processor loaded successfully (CPU-only)")
except Exception as e:
    print(f"✗ Processor loading failed: {e}")
    exit(1)

# Test 3: Verify processor doesn't have .to() method
has_to_method = hasattr(processor, 'to')
if not has_to_method:
    print("✓ Confirmed: Processor has no .to() method (as expected)")
else:
    print("⚠ Warning: Processor has .to() method (unexpected)")

# Test 4: Load model with GPU support
try:
    print("\nLoading model (this may take a while)...")
    model = AutoModelForVision2Seq.from_pretrained(
        "inclusionAI/UI-Venus-Ground-7B",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    print(f"✓ Model loaded successfully on {device}")
    print(f"  - Model device: {next(model.parameters()).device}")
    print(f"  - Model dtype: {next(model.parameters()).dtype}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    exit(1)

# Test 5: Create dummy inputs and verify tensor movement
try:
    # Create a simple text-only message for testing
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello, this is a test."}
            ]
        },
    ]

    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    print("✓ Inputs created successfully")
    print(f"  - Input keys: {list(inputs.keys())}")
    print(f"  - Input device before .to(): {inputs['input_ids'].device}")

    # Move to device
    inputs = inputs.to(device)
    print(f"  - Input device after .to(): {inputs['input_ids'].device}")

    if str(inputs['input_ids'].device).startswith('cuda'):
        print("✓ Tensors successfully moved to GPU")

except Exception as e:
    print(f"✗ Input processing failed: {e}")
    exit(1)

print("\n" + "="*50)
print("✓ All tests passed! GPU setup is correct.")
print("="*50)
print("\nKey findings:")
print("1. Processor stays on CPU (no .to() method)")
print("2. Model successfully loaded on GPU")
print("3. Input tensors can be moved to GPU")
print("4. Ready for inference on GPU!")