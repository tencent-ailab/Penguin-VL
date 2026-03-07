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
    output_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True)
    if output_ids.shape[1] > inputs["input_ids"].shape[1]:
        response_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    else:
        response_ids = output_ids
    response = processor.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
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


    # Example1: Single Image conversation (code generation)
    image_question = "please think this problem step by step and give the python code solution"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": "./assets/inputs/leetcode.png"}},
                {"type": "text", "text": image_question},
            ]
        }
    ]
    response = infer(conversation)
    print("Image conversation:")
    print(f"User: {image_question}")
    print(f"Assistant: {response}\n")

    # Example2: Single Image conversation (ocr)
    image_question = "please parse the text content in the paragraphs, from left to right, from top to bottom."
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": "./assets/inputs/newspaper.png"}},
                {"type": "text", "text": image_question},
            ]
        }
    ]
    response = infer(conversation)
    print("Image conversation:")
    print(f"User: {image_question}")
    print(f"Assistant: {response}\n")

    # Example3: Single Image conversation (creative writing)
    image_question = "Write a short poem inspired by this image. Capture the relationship between the man and the horse, as well as the traditional, historical atmosphere of the painting."
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": "./assets/inputs/horse_poet.png"}},
                {"type": "text", "text": image_question},
            ]
        }
    ]
    response = infer(conversation)
    print("Image conversation:")
    print(f"User: {image_question}")
    print(f"Assistant: {response}\n")
    
    
    # Example4: Single Image conversation (Long table parsing)
    image_question = "please output the table content in markdown format."
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": "./assets/inputs/2b_table_result.png"}},
                {"type": "text", "text": image_question},
            ]
        }
    ]
    response = infer(conversation)
    print("Image conversation:")
    print(f"User: {image_question}")
    print(f"Assistant: {response}\n")

    # Example5: Multi-round image conversation
    chart_questions = [
        "Look at the 'Nonmetropolitan' line. In what approximate year does it reach its absolute lowest point on the chart, and what is the approximate percent change at that time?",
        "Compare the three lines over the entire 50-year period. Which demographic category exhibits the highest volatility (the most dramatic peaks and valleys), and which one appears to be the most stable?",
    ]
    conversation = []

    for round_id, question in enumerate(chart_questions, start=1):
        if round_id == 1:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": {"image_path": "./assets/inputs/chart_understanding.png"}},
                        {"type": "text", "text": question},
                    ]
                }
            )
        else:
            conversation.append({"role": "user", "content": question})

        response = infer(conversation)
        conversation.append({"role": "assistant", "content": response})
        print(f"Image conversation - Round {round_id}:")
        print(f"User: {question}")
        print(f"Assistant: {response}\n")

    # Example6: Multi-round video conversation
    conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    video_questions = [
        "please describe the video in details",
        "At what timestamps is the Summar Palace mentioned?",
        "At what timestamps is the CHANG AN AVENUE mentioned?",
        "At what timestamps is the THE FINANCIAL STREET FORUM mentioned?",
    ]

    for round_id, question in enumerate(video_questions, start=1):
        if round_id == 1:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {"video_path": "./assets/inputs/video-example.mp4", "fps": 1, "max_frames": 180}},
                        {"type": "text", "text": question},
                    ]
                }
            )
        else:
            conversation.append({"role": "user", "content": question})

        response = infer(conversation)
        conversation.append({"role": "assistant", "content": response})
        print(f"Video conversation - Round {round_id}:")
        print(f"User: {question}")
        print(f"Assistant: {response}\n")

    # Example7: Mixed conversation
    mixed_question = "Write a fairy tale based on the video and the image below:\nVideo\n"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": mixed_question},
                {"type": "video", "video": {"video_path": "./assets/inputs/polar_bear.mp4", "fps": 1, "max_frames": 180}},
                {"type": "text", "text": "\n\nImage\n"},
                {"type": "image", "image": {"image_path": "./assets/inputs/horse_poet.png"}},
            ]
        }
    ]
    response = infer(conversation)
    print("Mixed conversation:")
    print(f"User: {mixed_question}")
    print(f"Assistant: {response}\n")

    # Example8: Plain text conversation
    text_question = "There are ten birds in a tree. If you shoot and kill one, how many are left?"
    conversation = [
        {
            "role": "user",
            "content": text_question,
        }
    ]
    response = infer(conversation)
    print("Text conversation:")
    print(f"User: {text_question}")
    print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
