import os
import json
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import math
import pickle
import time
import torch
import subprocess
from transformers import AutoProcessor, AutoTokenizer


try:
    from PIL import Image
except ImportError:
    raise SystemExit("Please install Pillow first: pip install pillow")


def init_worker(model_path):
    global _tokenizer, _model_path
    _model_path = model_path
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Process {os.getpid()}: Tokenizer loaded successfully")
    except Exception as e:
        print(f"Process {os.getpid()}: Failed to load tokenizer: {e}")
        _tokenizer = None


def print_statistics(lengths_tensor):
    print(f"Statistics:")
    print(f"  Min length: {lengths_tensor.min().item()}")
    print(f"  Max length: {lengths_tensor.max().item()}")
    print(f"  Mean length: {lengths_tensor.float().mean().item():.2f}")
    print(f"  Median length: {lengths_tensor.median().item()}")


def process_jsonl_lengths_streaming_multiprocess(jsonl_path, model_path, output_path,
                                                num_processes=None, chunk_size=100000):
    """
    Streaming multiprocess processing version - large file friendly
    The file is divided into chunks, and each process handles one chunk

    Args:
        jsonl_path: JSONL file path
        model_path: tokenizer model path
        output_path: output tensor save path
        num_processes: number of processes
        chunk_size: number of lines per chunk
    """
    
    if num_processes is None:
        num_processes = min(cpu_count(), 16)
    
    print(f"Using streaming multiprocess with {num_processes} processes")
    print(f"Chunk size: {chunk_size:,} lines per chunk")
    
    print("Counting total lines...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")
    
    print("Creating chunks...")
    chunk_files = []
    chunk_count = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            current_chunk = []
            start_idx = 0
            
            for line_idx, line in enumerate(f):
                current_chunk.append((line_idx, line.strip()))
                
                if len(current_chunk) >= chunk_size or line_idx == total_lines - 1:
                    chunk_file = f"temp_chunk_{chunk_count}.pkl"
                    with open(chunk_file, 'wb') as cf:
                        pickle.dump(current_chunk, cf)
                    chunk_files.append((chunk_file, start_idx, len(current_chunk)))
                    
                    current_chunk = []
                    start_idx = line_idx + 1
                    chunk_count += 1
        
        print(f"Created {len(chunk_files)} chunks")
        
        print("Processing chunks...")
        with Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_chunk_file, [(cf[0], cf[1]) for cf in chunk_files]),
                total=len(chunk_files),
                desc="Processing chunks"
            ))
    
    finally:
        for chunk_file, _, _ in chunk_files:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except Exception as e:
                    print(f"Warning: Failed to remove {chunk_file}: {e}")
    
    print("Merging results...")
    lengths = [0] * total_lines
    
    for chunk_result in chunk_results:
        if chunk_result:
            for line_idx, length in chunk_result:
                lengths[line_idx] = length
    
    print("Saving final tensor...")
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    torch.save(lengths_tensor, output_path)
    
    print(f"Lengths saved to: {output_path}")
    print(f"Tensor shape: {lengths_tensor.shape}")
    print_statistics(lengths_tensor)
    
    print("Merging results...")
    lengths = [0] * total_lines
    
    for chunk_result in chunk_results:
        for line_idx, length in chunk_result:
            lengths[line_idx] = length
    
    print("Saving final tensor...")
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    torch.save(lengths_tensor, output_path)
    
    print(f"Lengths saved to: {output_path}")
    print(f"Tensor shape: {lengths_tensor.shape}")
    print_statistics(lengths_tensor)

def process_chunk_file(args):
    """
    Process a chunk file
    Args:
        args: (chunk_file_path, start_idx)
    Returns:
        [(line_idx, length), ...]
    """
    global _tokenizer, _model_path
    
    chunk_file, start_idx = args
    
    if '_tokenizer' not in globals() or _tokenizer is None:
        print(f"Warning: Tokenizer not initialized in process {os.getpid()}, initializing now...")
        _tokenizer = AutoTokenizer.from_pretrained(_model_path)
    
    if not os.path.exists(chunk_file):
        print(f"Error: Chunk file does not exist: {chunk_file}")
        return []
    
    try:
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading chunk file {chunk_file}: {e}")
        return []
    
    results = []
    for line_idx, line_content in chunk_data:
        try:
            data = json.loads(line_content)
            length = 0
            for conv in data.get('conversations', []):
                conversation_value = conv['value']
                length += len(_tokenizer.encode(conversation_value))
            
            if not data.get('conversations', []):
                length = 0

            h, w = data.get('height', None), data.get('width', None)
            t = data.get('frames', None)
            img_length = 0
            if h is not None and w is not None:
                if t is not None:
                    img_length = min(10240, math.ceil(h/28)*math.ceil(w/28)*t)
                else:
                    img_length = min(10240, math.ceil(h/14)*math.ceil(w/14))
            length += img_length
            results.append((line_idx, length))
        except Exception as e:
            print(f"Error processing line {line_idx}: {e}")
            results.append((line_idx, 0))
    
    return results


def get_length(jsonl_path, model_path, output_path, num_processes=1):
    print(f"Processing file: {jsonl_path}")
    print(f"Using model: {model_path}")
    print(f"Output path: {output_path}")
    
    try:
        start_time = time.time()
        
        process_jsonl_lengths_streaming_multiprocess(
            jsonl_path, model_path, output_path,
            num_processes=num_processes,
            chunk_size=50000
        )
            
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("Processing completed!")


def resolve_path(img_list, root):
    if not img_list:
        return None
    p = img_list[0]
    if root:
        return os.path.join(root, p)
    return p

def fast_ffprobe(video_path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration,width,height,avg_frame_rate',
        '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        stream = json.loads(result.stdout)['streams'][0]
        return {
            "file_path": video_path,
            "height": int(stream.get("height", 0)),
            "width": int(stream.get("width", 0)),
            "duration": round(float(stream.get("duration", 0)), 2),
        }
    except Exception as e:
        return {"file_path": video_path, "video_error": str(e)}

def get_wh_one(line: str, root: str, modal: str = "image", processor=None, fps=1, max_frames=128):
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except Exception as e:
        print(f"json loads failed: {e}, line: {line}")
        return None

    if "image" in obj:
        modal = "image"
    elif "video" in obj:
        modal = "video"
    else:
        modal = "text"

    if modal == "image":
        img_path = resolve_path(obj.get("image", []), root)
        if not img_path:
            print(f"no image find, line: {line}")
            return None

        try:
            with Image.open(img_path) as im:
                w, h = im.size
            obj["width"] = int(w)
            obj["height"] = int(h)
        except Exception as e:
            print(f"open_image_failed: {e}")
            return None

        return json.dumps(obj) + "\n"
    elif modal == "video":
        video_path = resolve_path(obj.get("video", []), root)
        if not video_path:
            print(f"no video find, line: {line}")
            return None

        try:
            info = fast_ffprobe(video_path)
            h, w = info.get("height", 0), info.get("width", 0)
            obj["width"] = int(w)
            obj["height"] = int(h)
            obj["frames"] = max(max_frames, math.floor(info.get("duration", 0) * fps))
        except Exception as e:
            print(f"load_video_failed: {e}, path: {video_path}")
            return None

        return json.dumps(obj) + "\n"
    elif modal == "text":
        return json.dumps(obj) + "\n"
    

def get_meta(num_processes, args):
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file does not exist: {args.input}")

    process_fn = partial(get_wh_one, root=args.root, fps=args.fps, max_frames=args.max_frames)

    with open(args.input, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    output_filename = args.input.replace(".jsonl", "_meta.jsonl")

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(output_filename, "w", encoding="utf-8") as fout, \
         Pool(processes=num_processes) as pool:

        for out_line in tqdm(
            pool.imap_unordered(process_fn, fin, chunksize=args.chunksize),
            total=total_lines,
            desc="Processing",
            unit="line"
        ):
            if out_line is not None:
                fout.write(out_line)

    print(f"Complete! Results are written in {output_filename}")
    return output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="input jsonl file")
    parser.add_argument("--root", "-r", default="", help="image root directory (optional)")
    parser.add_argument("--fps", type=int, default=1,
                        help="fps")
    parser.add_argument("--max-frames", type=int, default=180,
                        help="max-frames")
    parser.add_argument("--tk-path", "-o", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--chunksize", type=int, default=100,
                        help="chunksize of imap_unordered")
    args = parser.parse_args()


    num_processes = min(cpu_count(), 128)
    print(f"Detected {cpu_count()} CPU cores, using {num_processes} processes")

    output_filename = get_meta(num_processes, args)

    lengths_path = os.path.join(args.root, "lengths.pt")
    get_length(output_filename, args.tk_path, lengths_path, num_processes=num_processes)

    length = torch.load(lengths_path)
    torch.save(length.argsort(), lengths_path)