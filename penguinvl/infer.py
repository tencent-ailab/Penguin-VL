import sys
sys.path.append('./')

import torch
from penguinvl import disable_torch_init, model_init, mm_infer


def main():
    disable_torch_init()

    # modal = "text"
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": "There are ten birds in a tree. If you shoot and kill one, how many are left?",
    #     }
    # ]

    modal = "image"
    image_path = "assets/inputs/horse_poet.png"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."},
            ]
        }
    ]

    # modal = "video"
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "video", "video": {"video_path", "assets/inputs/polar_bear.mp4", "fps": 1, "max_frames": 180}},
    #             {"type": "text", "text": "What is the cat doing?"},
    #         ]
    #     }
    # ]

    model_path = "../pg-vl-2b"
    model, processor = model_init(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device, dtype=torch.bfloat16)
    images = processor.load_images(image_path)
    image_inputs = processor.process_images(images, merge_size=1, return_tensors="pt")
    prompt = processor.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_system_prompt=True,
        add_generation_prompt=True,
    )
    text_inputs = processor.process_text(
        text=prompt,
        image_inputs=image_inputs,
        return_tensors="pt",
    )
    output = mm_infer(
        {**text_inputs, **image_inputs},
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal=modal,
        max_new_tokens=1024,
    )
    print(output)


if __name__ == "__main__":
    main()