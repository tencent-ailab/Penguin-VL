import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


# NOTE: transformers==4.51.3 is recommended for this script
model = None
processor = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="tencent/Penguin-VL-8B",
        help="Hugging Face model path or local model directory.",
    )
    return parser.parse_args()


@torch.inference_mode()
def infer(conversation):
    if model is None or processor is None:
        raise RuntimeError("Model and processor must be initialized before calling infer().")

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=True)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


def main():
    global model, processor

    args = parse_args()
    model_path = args.model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Video conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": "./assets/inputs/polar_bear.mp4", "fps": 1, "max_frames": 180}},
                {"type": "text", "text": "Describe this video in detail, then provide annotations with timestamps."},
            ]
        },
    ]
    print("Video conversation:\n", infer(conversation), '\n\n')

    # Image conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": "./assets/inputs/horse_poet.png"}},
                {"type": "text", "text": "Please describe the image in detail."},
            ]
        }
    ]
    print("Image conversation:\n", infer(conversation), '\n\n')

    # Mixed conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write a fairy tale based on the video and the image below:\nVideo\n"},
                {"type": "video", "video": {"video_path": "./assets/inputs/polar_bear.mp4", "fps": 1, "max_frames": 180}},
                {"type": "text", "text": "\n\nImage\n"},
                {"type": "image", "image": {"image_path": "./assets/inputs/horse_poet.png"}},
            ]
        }
    ]
    print("Mixed conversation:\n", infer(conversation), '\n\n')

    # Plain text conversation
    conversation = [
        {
            "role": "user",
            "content": "There are ten birds in a tree. If you shoot and kill one, how many are left?",
        }
    ]
    print("Text conversation:\n", infer(conversation), '\n\n')


if __name__ == "__main__":
    main()
