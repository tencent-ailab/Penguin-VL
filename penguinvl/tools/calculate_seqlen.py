from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import tempfile
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Please install Pillow first: pip install pillow")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_PATCH_SIZE = 14
_VIDEO_PATCH_SIZE = 28
_MAX_VISUAL_TOKENS = 10_240


# ---------------------------------------------------------------------------
# Multiprocessing worker helpers
# ---------------------------------------------------------------------------

def _init_worker(model_path: str) -> None:
    global _tokenizer, _model_path
    _model_path = model_path
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"[pid {os.getpid()}] Failed to load tokenizer: {e}")
        _tokenizer = None


def _ensure_tokenizer() -> None:
    """Lazily initialise the tokenizer if the worker init failed."""
    global _tokenizer, _model_path
    if globals().get("_tokenizer") is None:
        print(f"[pid {os.getpid()}] Tokenizer not ready — retrying…")
        _tokenizer = AutoTokenizer.from_pretrained(_model_path)


# ---------------------------------------------------------------------------
# Sequence-length computation
# ---------------------------------------------------------------------------

def _visual_token_count(h: int, w: int, frames: int | None) -> int:
    patch = _VIDEO_PATCH_SIZE if frames is not None else _IMAGE_PATCH_SIZE
    tokens = math.ceil(h / patch) * math.ceil(w / patch) * (frames or 1)
    return min(tokens, _MAX_VISUAL_TOKENS)


def _process_chunk(chunk_file: str) -> list[tuple[int, int]]:
    _ensure_tokenizer()

    try:
        with open(chunk_file, "rb") as f:
            chunk_data: list[tuple[int, str]] = pickle.load(f)
    except Exception as e:
        print(f"[pid {os.getpid()}] Cannot load {chunk_file}: {e}")
        return []

    results: list[tuple[int, int]] = []
    for line_idx, line_content in chunk_data:
        try:
            data = json.loads(line_content)
            text_tokens = sum(
                len(_tokenizer.encode(conv["value"]))
                for conv in data.get("conversations", [])
            )
            h, w = data.get("height"), data.get("width")
            vis_tokens = (
                _visual_token_count(h, w, data.get("frames"))
                if h is not None and w is not None
                else 0
            )
            results.append((line_idx, text_tokens + vis_tokens))
        except Exception as e:
            print(f"[pid {os.getpid()}] Error on line {line_idx}: {e}")
            results.append((line_idx, 0))

    return results


def _print_stats(tensor: torch.Tensor) -> None:
    print(
        f"  min={tensor.min().item()}"
        f"  max={tensor.max().item()}"
        f"  mean={tensor.float().mean().item():.1f}"
        f"  median={tensor.median().item()}"
    )


def compute_lengths(
    jsonl_path: str,
    model_path: str,
    output_path: str,
    num_processes: int | None = None,
    chunk_size: int = 50_000,
) -> None:
    """Tokenize every sample in *jsonl_path* and save per-line token lengths."""
    num_processes = num_processes or min(cpu_count(), 16)
    print(f"Processes: {num_processes}  |  chunk size: {chunk_size:,}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")

    tmp_dir = tempfile.mkdtemp(prefix="seqlen_chunks_")
    chunk_files: list[str] = []

    try:
        # Split input into temporary pickle chunks for parallel processing.
        with open(jsonl_path, "r", encoding="utf-8") as f:
            chunk: list[tuple[int, str]] = []
            for line_idx, line in enumerate(f):
                chunk.append((line_idx, line.strip()))
                if len(chunk) >= chunk_size:
                    _flush_chunk(chunk, tmp_dir, chunk_files)
                    chunk = []
            if chunk:
                _flush_chunk(chunk, tmp_dir, chunk_files)

        print(f"Chunks: {len(chunk_files)}")

        with Pool(
            processes=num_processes,
            initializer=_init_worker,
            initargs=(model_path,),
        ) as pool:
            chunk_results = list(
                tqdm(
                    pool.imap(_process_chunk, chunk_files),
                    total=len(chunk_files),
                    desc="Processing chunks",
                )
            )
    finally:
        for path in chunk_files:
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    lengths = [0] * total_lines
    for chunk_result in chunk_results:
        for line_idx, length in chunk_result:
            lengths[line_idx] = length

    tensor = torch.tensor(lengths, dtype=torch.long)
    torch.save(tensor, output_path)
    print(f"Saved → {output_path}  shape={tensor.shape}")
    _print_stats(tensor)


def _flush_chunk(
    chunk: list[tuple[int, str]],
    tmp_dir: str,
    chunk_files: list[str],
) -> None:
    path = os.path.join(tmp_dir, f"chunk_{len(chunk_files)}.pkl")
    with open(path, "wb") as f:
        pickle.dump(chunk, f)
    chunk_files.append(path)


def get_length(
    jsonl_path: str,
    model_path: str,
    output_path: str,
    num_processes: int = 1,
) -> None:
    print(f"Input : {jsonl_path}\nModel : {model_path}\nOutput: {output_path}")
    t0 = time.time()
    try:
        compute_lengths(jsonl_path, model_path, output_path, num_processes=num_processes)
    except Exception:
        import traceback
        traceback.print_exc()
    print(f"Done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Metadata enrichment (image / video dimensions)
# ---------------------------------------------------------------------------

def _resolve_path(paths: list[str] | str | None, root: str) -> str | None:
    if not paths:
        return None
    first = paths[0] if isinstance(paths, list) else paths
    return os.path.join(root, first) if root else first


def _ffprobe(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,width,height,avg_frame_rate",
        "-of", "json", video_path,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "height": int(stream.get("height", 0)),
        "width": int(stream.get("width", 0)),
        "duration": round(float(stream.get("duration", 0)), 2),
    }


def _enrich_one(line: str, root: str, fps: int, max_frames: int) -> str | None:
    """Attach width/height (and frames for video) to a single JSONL record."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return None

    if "image" in obj:
        img_path = _resolve_path(obj["image"], root)
        if not img_path:
            print(f"No image path in record: {line}")
            return None
        try:
            with Image.open(img_path) as im:
                obj["width"], obj["height"] = im.size
        except Exception as e:
            print(f"Cannot open image {img_path}: {e}")
            return None

    elif "video" in obj:
        video_path = _resolve_path(obj["video"], root)
        if not video_path:
            print(f"No video path in record: {line}")
            return None
        try:
            info = _ffprobe(video_path)
            obj["width"] = info["width"]
            obj["height"] = info["height"]
            obj["frames"] = max(max_frames, math.floor(info["duration"] * fps))
        except Exception as e:
            print(f"Cannot probe video {video_path}: {e}")
            return None

    # text-only samples pass through unchanged
    return json.dumps(obj) + "\n"


def get_meta(args: argparse.Namespace, num_processes: int) -> str:
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    process_fn = partial(_enrich_one, root=args.root, fps=args.fps, max_frames=args.max_frames)
    output_path = args.input.replace(".jsonl", "_meta.jsonl")

    with open(args.input, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with (
        open(args.input, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
        Pool(processes=num_processes) as pool,
    ):
        for line in tqdm(
            pool.imap_unordered(process_fn, fin, chunksize=args.chunksize),
            total=total_lines,
            desc="Enriching metadata",
            unit="line",
        ):
            if line is not None:
                fout.write(line)

    print(f"Metadata saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute per-sample sequence lengths for a JSONL dataset."
    )
    p.add_argument("--input", "-i", required=True, help="Input JSONL file")
    p.add_argument("--root", "-r", default="", help="Root directory for media files")
    p.add_argument("--fps", type=int, default=1, help="Frames per second for video sampling")
    p.add_argument("--max-frames", type=int, default=180, help="Minimum frame count for videos")
    p.add_argument("--tk-path", default="Qwen/Qwen3-0.6B", help="Tokenizer model path or HuggingFace ID")
    p.add_argument("--chunksize", type=int, default=100, help="imap_unordered chunk size")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    num_processes = min(cpu_count(), 128)
    print(f"CPU cores: {cpu_count()}  →  using {num_processes} processes")

    meta_path = get_meta(args, num_processes)

    lengths_dir = args.root if args.root else os.path.dirname(os.path.abspath(args.input))
    lengths_path = os.path.join(lengths_dir, "lengths.pt")

    get_length(meta_path, args.tk_path, lengths_path, num_processes=num_processes)

    lengths = torch.load(lengths_path)
    torch.save(lengths.argsort(), lengths_path)
    print(f"Sorted indices saved → {lengths_path}")
