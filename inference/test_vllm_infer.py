"""
PenguinVL vLLM deployment demo script.

Shows how to load PenguinVL with vLLM and run text, image, video, and batch inference.

Usage:
    # Single GPU, run all demos
    CUDA_VISIBLE_DEVICES=0 python inference/test_vllm_infer.py \
        --model-path /path/to/pg-vl-2b

    # Text-only demo
    python inference/test_vllm_infer.py --model-path /path/to/pg-vl-2b --demo text

    # Image demo (requires --image-path)
    python inference/test_vllm_infer.py --model-path /path/to/pg-vl-2b \
        --demo image --image-path assets/demo.png

    # Video demo (requires --video-path)
    python inference/test_vllm_infer.py --model-path /path/to/pg-vl-2b \
        --demo video --video-path assets/cat_and_chicken.mp4 --max-video-frames 16

    # Multi-GPU tensor parallelism (e.g. 8B model)
    CUDA_VISIBLE_DEVICES=0,1 python inference/test_vllm_infer.py \
        --model-path /path/to/pg-vl-8b --tensor-parallel-size 2

    # Custom max tokens and GPU memory
    python inference/test_vllm_infer.py --model-path /path/to/pg-vl-2b \
        --max-new-tokens 256 --gpu-memory-utilization 0.8
"""

import os
import sys
sys.path.insert(0, ".")

# Fix FlashInfer JIT linking: add common libcuda paths so ld can find -lcuda
_libcuda_paths = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/lib64/stubs",
]
_existing = os.environ.get("LIBRARY_PATH", "")
_extra = ":".join(p for p in _libcuda_paths if os.path.exists(p))
if _extra:
    os.environ["LIBRARY_PATH"] = f"{_extra}:{_existing}" if _existing else _extra

import argparse
import time
import numpy as np
from PIL import Image

import penguinvl  # noqa: F401 — registers PenguinVLQwen3ForCausalLM in vLLM ModelRegistry

from vllm import LLM, SamplingParams


IMAGE_TOKEN = "<image>"


def build_prompt(user_content: str, num_images: int = 0, num_video_frames: int = 0) -> str:
    """Build a single-turn prompt in PenguinVL Qwen3 chat format."""
    prefix = ""
    if num_images > 0:
        prefix += "\n".join([IMAGE_TOKEN] * num_images) + "\n"
    if num_video_frames > 0:
        prefix += ",".join([IMAGE_TOKEN] * num_video_frames) + "\n"
    return (
        f"<|im_start|>user\n{prefix}{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def load_video_frames(video_path: str, max_frames: int = 16, fps: float = 1.0) -> list:
    """Load video frames as a list of PIL Images using decord."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())

    sample_interval = max(1, int(video_fps / fps))
    indices = list(range(0, total_frames, sample_interval))

    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]

    frames_np = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames_np]


def make_text_request(question: str) -> dict:
    """Build a text-only request (no multi_modal_data)."""
    return {"prompt": build_prompt(question)}


def make_image_request(image_path: str, question: str) -> dict:
    """Build a request with image: prompt + multi_modal_data.image."""
    image = Image.open(image_path).convert("RGB")
    return {
        "prompt": build_prompt(question, num_images=1),
        "multi_modal_data": {"image": image},
    }


def make_video_request(video_path: str, question: str,
                       max_frames: int = 16, fps: float = 1.0) -> dict:
    """Build a request with video frames passed as images."""
    frames = load_video_frames(video_path, max_frames=max_frames, fps=fps)
    return {
        "prompt": build_prompt(question, num_video_frames=len(frames)),
        "multi_modal_data": {"image": frames},
    }


def run_demo_text(llm: LLM, sampling_params: SamplingParams) -> None:
    """Demo 1: Text-only conversation."""
    print("\n--- Demo: Text-only ---")
    request = make_text_request("What is the color of bananas?")
    outputs = llm.generate([request], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()
    print(f"Input:  What is the color of bananas?")
    print(f"Output: {text}\n")


def run_demo_image(llm: LLM, sampling_params: SamplingParams, image_path: str) -> None:
    """Demo 2: Single image description."""
    print("\n--- Demo: Single image description ---")
    request = make_image_request(image_path, "Please describe this image in detail.")
    outputs = llm.generate([request], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()
    print(f"Image:  {image_path}")
    print(f"Output: {text}\n")


def run_demo_qa(llm: LLM, sampling_params: SamplingParams, image_path: str) -> None:
    """Demo 3: Image question answering."""
    print("\n--- Demo: Image QA ---")
    request = make_image_request(image_path, "According to the image, which model performs best?")
    outputs = llm.generate([request], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()
    print(f"Image:  {image_path}")
    print(f"Output: {text}\n")


def run_demo_video(
    llm: LLM,
    sampling_params: SamplingParams,
    video_path: str,
    max_frames: int = 16,
    fps: float = 1.0,
) -> None:
    """Demo 4: Video understanding."""
    print("\n--- Demo: Video understanding ---")
    request = make_video_request(
        video_path,
        "Describe what is happening in this video.",
        max_frames=max_frames,
        fps=fps,
    )
    num_frames = len(request["multi_modal_data"]["image"])
    print(f"Video:  {video_path} ({num_frames} frames @ {fps} fps)")
    outputs = llm.generate([request], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()
    print(f"Output: {text}\n")


def run_demo_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    image_path: str,
) -> None:
    """Demo 5: Batch inference (mixed text + image)."""
    print("\n--- Demo: Batch (text + image) ---")
    requests = [
        make_text_request("What is 2 + 3? Answer with just the number."),
        make_image_request(image_path, "Describe this image in one sentence."),
        make_text_request("Write a haiku about artificial intelligence."),
    ]
    t0 = time.time()
    outputs = llm.generate(requests, sampling_params=sampling_params)
    elapsed = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    labels = ["math", "image", "haiku"]
    for label, out in zip(labels, outputs):
        print(f"  [{label}] {out.outputs[0].text.strip()}")
    print(f"  Time: {elapsed:.2f}s, total tokens: {total_tokens}, "
          f"~{total_tokens / elapsed:.1f} tok/s\n")


def main():
    parser = argparse.ArgumentParser(
        description="PenguinVL vLLM deployment demo: load model and run example inference."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/x2robot_v2/cyril/pg-vl-2b",
        help="PenguinVL model path (HF or local dir)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        nargs="+",
        default=["assets/results_image_2b.png"],
        help="Image path(s) for image/qa/batch demos (first one is used)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="assets/cat_and_chicken.mp4",
        help="Video path for video demo",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=16,
        help="Max frames to sample from video",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="Target FPS for video frame sampling",
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["text", "image", "qa", "video", "batch", "all"],
        default="all",
        help="Demo to run: text, image, qa, video, batch, or all",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (e.g. 2 for 8B on 2 GPUs)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens per generation",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max context length (use >= 8192 when using images)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory for vLLM (0-1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PenguinVL vLLM Deployment Demo")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Demo:  {args.demo}")
    print(f"  Image: {args.image_path[0] if args.image_path else '-'}")
    print(f"  Video: {args.video_path}")
    print(f"  TP:    {args.tensor_parallel_size}")
    print(f"  Max tokens: {args.max_new_tokens}")
    print("=" * 60)

    print("\nLoading vLLM model...")
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    image_path = args.image_path[0] if args.image_path else None

    if args.demo == "text" or args.demo == "all":
        run_demo_text(llm, sampling_params)
    if args.demo == "image" or args.demo == "all":
        if image_path:
            run_demo_image(llm, sampling_params, image_path)
        else:
            print("\n--- Demo: Single image --- Skipped (no --image-path)\n")
    if args.demo == "qa" or args.demo == "all":
        if image_path:
            run_demo_qa(llm, sampling_params, image_path)
        else:
            print("\n--- Demo: Image QA --- Skipped (no --image-path)\n")
    if args.demo == "video" or args.demo == "all":
        if args.video_path:
            run_demo_video(
                llm, sampling_params, args.video_path,
                max_frames=args.max_video_frames, fps=args.video_fps,
            )
        else:
            print("\n--- Demo: Video --- Skipped (no --video-path)\n")
    if args.demo == "batch" or args.demo == "all":
        if image_path:
            run_demo_batch(llm, sampling_params, image_path)
        else:
            print("\n--- Demo: Batch --- Skipped (no --image-path)\n")

    print("Done.")


if __name__ == "__main__":
    main()
