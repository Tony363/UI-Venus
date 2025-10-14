from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import os
from qwen_vl_utils import process_vision_info


# model path
model_name = "inclusionAI/UI-Venus-Ground-7B"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name)

generation_config = {
    "max_new_tokens": 2048,
    "do_sample": False,
    "temperature": 0.0
}

def inference(instruction, image_path):
    assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
    
    prompt_origin = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'
    full_prompt = prompt_origin.format(instruction)

    min_pixels = 2000000
    max_pixels = 4800000
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels
                },
                {"type": "text", "text": full_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**model_inputs, **generation_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # normalized coordinates
    try:
        box = eval(output_text[0])
        input_height = model_inputs['image_grid_thw'][0][1] * 14
        input_width = model_inputs['image_grid_thw'][0][2] * 14
        abs_x1 = float(box[0]) / input_width
        abs_y1 = float(box[1]) / input_height
        abs_x2 = float(box[2]) / input_width
        abs_y2 = float(box[3]) / input_height
        bbox = [abs_x1, abs_y1, abs_x2, abs_y2]
    except Exception:
        bbox = [0, 0, 0, 0]

    point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    result_dict = {
        "result": "positive",
        "format": "x1y1x2y2",
        "raw_response": output_text,
        "bbox": bbox,
        "point": point
    }
    
    return result_dict

