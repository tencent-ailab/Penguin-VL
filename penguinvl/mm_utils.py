# Copyright (c) Penguin-VL team at Tencent AI Lab
import ast
import os
import re
import math
import base64
import traceback
from io import BytesIO
from typing import Optional

import torch
import torchvision.transforms.functional as VF
import numpy as np
from transformers import StoppingCriteria

import cv2
import requests
import imageio
import ffmpeg
from PIL import Image
from decord import VideoReader, cpu
from einops import rearrange
from torch import nn

from .constants import NUM_FRAMES, MAX_FRAMES, NUM_FRAMES_PER_SECOND, MODAL_INDEX_MAP, DEFAULT_IMAGE_TOKEN, MIN_FRAME_SIMILARITY



def load_images(image_path):
    if isinstance(image_path, str) and os.path.isfile(image_path):
        images = [Image.open(image_path).convert('RGB')]
    elif isinstance(image_path, str) and os.path.isdir(image_path):
        images = [Image.open(os.path.join(image_path, f)).convert('RGB') for f in sorted(os.listdir(image_path))]
    elif isinstance(image_path, str) and (image_path.startswith("http://") or image_path.startswith("https://")):
        images = Image.open(requests.get(image_path, stream=True).raw)
    elif isinstance(image_path, list) and isinstance(image_path[0], str):
        images = [Image.open(f).convert('RGB') for f in image_path]
    elif isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
        images = image_path
    elif isinstance(image_path, Image.Image):
        images = [image_path]
    else:
        raise ValueError(f"Unsupported image path type: {type(image_path)}")

    return images

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_pad_image(image, padding_value=(0, 0, 0)):
    image = expand2square(image, padding_value)

    return [image]


def process_images(image_path, processor, aspect_ratio='pad', image_size=384, use_thumbnail=True):
    images = load_images(image_path)

    padding_value = tuple(int(x*255) for x in processor.image_mean)

    image_grids = []
    for image in images:
        if aspect_ratio == 'pad':
            image_grid = process_pad_image(image, padding_value=padding_value)
        else:
            image_grid = [image]

        image_grid = [processor.preprocess(image_row, return_tensors='pt', num_images=len(images)) for image_row in image_grid]
        image_grids.append(image_grid)

    return image_grids


def frame_sample(duration, mode='uniform', num_frames=None, vid_fps=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        if duration <= num_frames:
            return np.arange(duration).astype(int)
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        # if duration <= num_frames:
        #     return np.arange(duration).astype(int)
        # seg_size = float(duration - 1) / num_frames

        # frame_ids = []
        # for i in range(num_frames):
        #     # Calculate the start and end indices of each segment
        #     start = seg_size * i
        #     end   = seg_size * (i + 1)
        #     # Append the middle index of the segment to the list
        #     frame_ids.append((start + end) / 2)

        # return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert vid_fps is not None, "FPS must be provided for FPS sampling."
        fps = fps if fps is not None else NUM_FRAMES_PER_SECOND
        segment_len = min(vid_fps // fps, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def load_video_from_ids(video_path, s=None, e=None, fps=None, max_frames=None, temporal_factor=1):
    if s is not None and e is not None:
        s = s if s >= 0. else 0.
        e = e if e >= 0. else 0.
        if s > e:
            s, e = e, s
        elif s == e:
            e = s + 1

    # 1. Loading Video
    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))

        vid_fps = 3
        num_frames_of_video = len(frame_files)
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)

        vid_fps = 25
        num_frames_of_video = len(gif_reader)
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        # vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

    # 2. Determine frame range & Calculate frame indices
    f_start = 0                       if s is None else max(int(s * vid_fps) - 1, 0)
    f_end   = num_frames_of_video - 1 if e is None else min(int(e * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))

    duration = len(frame_indices)
    # 3. Sampling frame indices
    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    if fps is not None and duration / vid_fps < max_frames:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', vid_fps=vid_fps, fps=fps)]
    else:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=max_frames)]

    # 4. Acquire frame data
    if os.path.isdir(video_path):
        frames = [cv2.cvtColor(cv2.imread(os.path.join(video_path, frame_files[frame_idx])), cv2.COLOR_BGR2RGB) for frame_idx in sampled_frame_indices]
    elif video_path.endswith('.gif'):
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
    else:
        frames = vreader.get_batch(sampled_frame_indices).asnumpy()

    # frames = frames.transpose(0, 3, 1, 2)
    timestamps = [x / vid_fps for x in sampled_frame_indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    frames = np.array(frames)
    frames = frames.transpose(0, 3, 1, 2)
    frames_tensor = torch.from_numpy(frames.copy()).float()
    frame_types = extract_ki_frames(frames_tensor)

    frames = [frame for frame in frames]

    # NOTE: pad the video with black frames
    # while num_frames is not None and len(video_data) < num_frames:
    #     video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    return frames, timestamps, frame_types


def smart_resize(
    height: int,
    width: int,
    factor: int = 14,
    min_pixels: int = 0,
    max_pixels: int = 16384,
):
    """
    Compute target (height, width) such that:
    - Both dimensions are divisible by factor.
    - Total pixels lie in [min_pixels, max_pixels].
    - Aspect ratio is preserved as closely as possible.
    """
    def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor
    def ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor
    def floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    max_ratio = 200
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"Aspect ratio must be < {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h = max(factor, round_by_factor(height, factor))
    w = max(factor, round_by_factor(width, factor))
    if h * w > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        h = floor_by_factor(height / scale, factor)
        w = floor_by_factor(width / scale, factor)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = ceil_by_factor(height * scale, factor)
        w = ceil_by_factor(width * scale, factor)
    return max(h, factor), max(w, factor)


def get_frame_sim(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    patch_size: int = 14,
    threshold: float = 0.7,
    epsilon: float = 1e-8,
) -> float:
    """Cosine similarity between two frames in HSV, averaged over patches. Returns mean similarity in [0, 1]."""
    assert frame1.dim() == 3 and frame2.dim() == 3, "Frames must be 3D tensors [C, H, W]"

    def to_hsv_tensor(tensor: torch.Tensor) -> torch.Tensor:
        arr = tensor.cpu().permute(1, 2, 0).numpy()
        if arr.dtype in (np.float32, np.float64):
            arr = arr.astype(np.uint8)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        return torch.from_numpy(hsv).permute(2, 0, 1).to(tensor.device).float()

    f1 = to_hsv_tensor(frame1)
    f2 = to_hsv_tensor(frame2)
    patch1 = rearrange(f1, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()
    patch2 = rearrange(f2, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon
    cos_sim = (patch1 / norm1 * patch2 / norm2).sum(dim=-1)

    both_near_zero = (norm1.squeeze() < 0.01) & (norm2.squeeze() < 0.01)
    similar = torch.ones_like(cos_sim)
    similar[~both_near_zero] = (cos_sim[~both_near_zero] > threshold).float()
    return similar[~both_near_zero].float().mean().item()


# KI: keyframe indices (formerly slow/fast). 0 = key frame, 1 = intermediate frame.
K_PATCH = 14
K_MIN_PIXELS = 10 * 14 * 14
K_MAX_PIXELS = 10240 * 14 * 14


def extract_ki_frames(
    frames: torch.Tensor,
    threshold: float = MIN_FRAME_SIMILARITY,
) -> list:
    """
    Label each frame as keyframe (0) or non-keyframe (1) by comparing to the previous keyframe.
    First frame is always a keyframe; a new keyframe is chosen when similarity drops below threshold.
    """
    assert frames.dim() == 4, "Frames must be 4D tensor [N, C, H, W]"

    def _keyframe_indices(f: torch.Tensor) -> list:
        indices = [0]
        key = f[0]
        for i in range(1, f.size(0)):
            if get_frame_sim(key, f[i]) < threshold:
                indices.append(i)
                key = f[i]
        return indices

    _, _, h, w = frames.shape
    rh, rw = smart_resize(h, w, factor=K_PATCH, min_pixels=K_MIN_PIXELS, max_pixels=K_MAX_PIXELS)
    resized = nn.functional.interpolate(frames, (rh, rw), mode="bilinear", antialias=True).float()
    k_indices = _keyframe_indices(resized)
    frame_types = torch.ones(frames.size(0), dtype=torch.int32)
    frame_types[k_indices] = 0
    return frame_types.tolist()


def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    size: Optional[int] = None,
    size_divisible: int = 1,
    precise_time: bool = False,
    verbose: bool = False,
    temporal_factor: int = 1
):
    """
    Load and process a video file and return the frames and the timestamps of each frame.

    Args:
        video_path (str): Path to the video file.
        start_time (float, optional): Start time in seconds. Defaults to None.
        end_time (float, optional): End time in seconds. Defaults to None.
        fps (float, optional): Frames per second. Defaults to None.
        num_frames (float, optional): Number of frames to sample. Defaults to None.
        size (int, optional): Size of the shortest side. Defaults to None.
        size_divisible (int, optional): Size divisible by this number. Defaults to 1.
        precise_time (bool, optional): Whether to use precise time. Defaults to False.
        verbose (bool, optional): Print ffmpeg output. Defaults to False.

    Returns:
        frames (List[PIL.Image]): List of frames.
        timestamps (List[float]): List of timestamps.
    """
    if video_path.endswith('png') or video_path.endswith('jpg') or video_path.endswith('jpeg'):
        return load_images(video_path), [], [0]
    if start_time is not None and end_time is not None and end_time - start_time < 1:
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if os.path.isdir(video_path):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if video_path.endswith('.gif'):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    w, h = int(video_stream['width']), int(video_stream['height'])

    kwargs, input_kwargs, output_kwargs = {}, {}, {}
    do_trim = start_time is not None or end_time is not None
    if start_time is not None:
        new_start_time = max(float(video_stream['start_time']), start_time)
        duration -= new_start_time - start_time
        start_time = new_start_time
    else:
        start_time = float(video_stream['start_time'])
    if end_time is not None:
        duration = min(duration, end_time - start_time)
    else:
        duration = duration
    if do_trim:
        kwargs = {'ss': start_time, 't': duration}
    if precise_time:
        output_kwargs.update(kwargs)
    else:
        input_kwargs.update(kwargs)

    if size is not None:
        scale_factor = size / min(w, h)
        new_w, new_h = round(w * scale_factor), round(h * scale_factor)
    else:
        new_w, new_h = w, h
    new_w = new_w // size_divisible * size_divisible
    new_h = new_h // size_divisible * size_divisible

    # NOTE: It may result in unexpected number of frames in ffmpeg
    # if calculate the fps directly according to max_frames
    # NOTE: the below lines may hurt the performance
    # if max_frames is not None and (fps is None or duration * fps > 2 * max_frames):
    #     fps = max_frames / duration * 2

    stream = ffmpeg.input(video_path, **input_kwargs)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    if new_w != w or new_h != h:
        stream = ffmpeg.filter(stream, 'scale', new_w, new_h)
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

    frames = np.frombuffer(out, np.uint8).reshape([-1, new_h, new_w, 3]).transpose([0, 3, 1, 2])

    if fps is not None:
        timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)[:len(frames)]
    else:
        timestamps = np.linspace(start_time, start_time + duration, len(frames))

    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    if max_frames is not None and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
        timestamps = [timestamps[i] for i in indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    frames_tensor = torch.from_numpy(frames.copy()).float()
    frame_types = extract_ki_frames(frames_tensor)

    frames = [frame for frame in frames]

    return frames, timestamps, frame_types


def process_video(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=None):
    fps = 1 if num_frames is None else None
    # FFmpeg
    frames, timestamps = load_video(video_path, s, e, fps=fps, max_frames=num_frames)
    # Decord
    # frames, timestamps = load_video_from_ids(video_path, s, e, fps=fps, max_frames=num_frames)

    assert len(frames) == len(timestamps), "Number of frames and timestamps must match."

    if aspect_ratio == 'pad':
        frames = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in frames]

    if aspect_ratio == 'avt':
        frames = [processor.preprocess(frame, return_tensors='pt', image_num=len(frames)) for frame in frames]
        grid_frames = [frames]
    else:
        frames = processor.preprocess(frames, return_tensors='pt', image_num=len(frames))
        grid_frames = [[frames]]

    return grid_frames, timestamps


def load_audio(
    audio_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    verbose: bool = False,
    sample_rate: int = 16000,
):
    """
    Load and process an audio file and return the wave and the timestamps of each frame.

    Args:
        audio_path (str): Path to the audio file.
        start_time (float, optional): Start time in seconds. Defaults to None.
        end_time (float, optional): End time in seconds. Defaults to None.
        verbose (bool, optional): Print ffmpeg output. Defaults to False.

    Returns:
        wave (List[PIL.Image]): List of wave.
        timestamps (List[float]): List of timestamps.
    """

    audio_stream_ff = (
        ffmpeg
        .input(audio_path)
        .output(
            "pipe:",
            format="s16le",
            acodec="pcm_s16le",
            ac=1,
            ar=sample_rate,
        )
    )
    audio_out, audio_err = ffmpeg.run(audio_stream_ff, capture_stdout=True, quiet=not verbose)
    audio = np.frombuffer(audio_out, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / sample_rate
    if duration > 600:
        duration = 600
    if duration > 30:
        audio = [audio[i*30*sample_rate: (i+1)*30*sample_rate] for i in range(int(duration // 30) + 1)]
    else:
        audio = [audio]
    timestamps = [t for n, chunk in enumerate(audio) for t in range(n*30, n*30 + math.ceil(len(chunk) / sample_rate), 2)]
    return audio, timestamps


def tokenizer_multimodal_token(prompt, tokenizer, multimodal_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    """Tokenize text and multimodal tag to input_ids.

    Args:
        prompt (str): Text prompt (w/ multimodal tag), e.g., '<video>\nDescribe the video.'
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        multimodal_token (int): Token index corresponding to the multimodal tag.
    """
    multimodal_token_index = MODAL_INDEX_MAP.get(multimodal_token, None)
    if multimodal_token_index is None:
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for idx, chunk in enumerate(prompt.split(multimodal_token))]

        input_ids = []
        for i in range(1, 2 * len(prompt_chunks)):
            if i % 2 == 1:
                input_ids.extend(prompt_chunks[i // 2])
            else:
                input_ids.append(multimodal_token_index)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
