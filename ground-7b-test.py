# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load processor (stays on CPU)
processor = AutoProcessor.from_pretrained("inclusionAI/UI-Venus-Ground-7B")

# Load model with memory optimization
model = AutoModelForVision2Seq.from_pretrained(
    "inclusionAI/UI-Venus-Ground-7B",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
