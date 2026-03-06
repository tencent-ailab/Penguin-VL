"""PenguinVL multimodal processor for vLLM.

Handles image preprocessing, prompt tokenization, and dummy input generation
for PenguinVL models within the vLLM serving framework.
"""

import math
import warnings
from typing import Mapping, Optional, Any
from collections.abc import Sequence

import numpy as np
import torch
from transformers import BatchFeature, ProcessorMixin
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs, MultiModalDataDict
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor, BaseProcessingInfo, PromptReplacement, PromptUpdate
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.transformers_utils.processor import cached_image_processor_from_config
from penguinvl.constants import DEFAULT_IMAGE_TOKEN
from penguinvl.mm_utils import extract_ki_frames


class PenguinVLProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(
        self,
        *,
        image_merge_size: int = 1,
        video_merge_size: int = 2,
        min_tokens: int = 16,
        max_tokens: int = 16384,
        **kwargs
    ):
        image_processor = self.get_image_processor(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        return self.ctx.get_hf_processor(
            ProcessorMixin,
            image_processor=image_processor,
            image_merge_size=image_merge_size,
            video_merge_size=video_merge_size,
        )

    def get_image_processor(
        self,
        *,
        min_tokens: int = 16,
        max_tokens: int = 16384,
        **kwargs
    ):
        return cached_image_processor_from_config(
            self.ctx.model_config,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

    def get_max_image_tokens(self) -> int:
        image_processor = self.get_image_processor()
        return image_processor.max_tokens

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        num_images = mm_counts.get("image", 0)
        if num_images == 0:
            return {"image": 0}

        hf_processor = self.get_hf_processor()
        image_processor = hf_processor.image_processor
        max_tokens = image_processor.max_tokens
        patch_size = image_processor.patch_size
        merge_size = hf_processor.image_merge_size
        target_height = int(math.sqrt(max_tokens // num_images)) * patch_size * merge_size
        num_tokens = (target_height // patch_size // merge_size) ** 2

        return {"image": num_tokens}


class PenguinVLDummyInputsBuilder(BaseDummyInputsBuilder[PenguinVLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        if num_images > 0:
            return "\n".join([DEFAULT_IMAGE_TOKEN] * num_images)
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        if num_images > 0:
            hf_processor = self.info.get_hf_processor()
            image_processor = hf_processor.image_processor
            max_tokens = image_processor.max_tokens
            patch_size = image_processor.patch_size
            merge_size = hf_processor.image_merge_size
            target_height = target_width = (
                int(math.sqrt(max_tokens // num_images)) * patch_size * merge_size
            )
            return {
                "image": self._get_dummy_images(
                    width=target_width,
                    height=target_height,
                    num_images=num_images,
                ),
            }
        return {}


class PenguinVLMultiModalProcessor(BaseMultiModalProcessor[PenguinVLProcessingInfo]):

    def _extract_ki_frame_types(self, frames: list) -> Optional[list]:
        """Extract keyframe (0) vs intermediate (1) labels using KI strategy."""
        if len(frames) <= 1:
            return None
        arr = np.stack([np.array(f) for f in frames])
        if arr.shape[-1] == 3:
            arr = arr.transpose(0, 3, 1, 2)
        frames_tensor = torch.from_numpy(arr.astype(np.float32))
        return extract_ki_frames(frames_tensor)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        has_images = bool(mm_data.get("image") or mm_data.get("images"))
        if not has_images:
            tokenizer = self.info.get_tokenizer()
            encoding = tokenizer(prompt, return_tensors="pt")
            return BatchFeature(data=dict(encoding))

        images = mm_data.get("images") or mm_data.get("image")
        if not isinstance(images, list):
            images = [images]

        if len(images) <= 1:
            return self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(text=prompt, images=images),
                dict(**mm_kwargs, **tok_kwargs),
            )

        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        clips = []
        merge_sizes = []
        clip_frame_types = []
        for img in images:
            if isinstance(img, (list, tuple)):
                frames = list(img)
                clips.append(frames)
                merge_sizes.append(hf_processor.video_merge_size)
                frame_types = self._extract_ki_frame_types(frames)
                clip_frame_types.append(frame_types)
            else:
                clips.append([img])
                merge_sizes.append(hf_processor.image_merge_size)
                clip_frame_types.append(None)

        image_inputs = hf_processor.image_processor(
            images=clips,
            merge_size=merge_sizes,
            frame_types=clip_frame_types if any(ft is not None for ft in clip_frame_types) else None,
        )

        data = {}
        for k, v in image_inputs.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v)
            elif isinstance(v, torch.Tensor):
                data[k] = v
            else:
                data[k] = v

        tokenizer = self.info.get_tokenizer()
        encoding = tokenizer(prompt, return_tensors="pt")
        data.update(dict(encoding))
        return BatchFeature(data=data)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:

        def _get_downsampled_grid_sizes(image_inputs):
            grid_sizes = []
            raw_grid_sizes = [item.get("grid_sizes", None) for item in image_inputs["image"]]
            raw_merge_sizes = [item.get("merge_sizes", None) for item in image_inputs["image"]]
            for grid_size, merge_size in zip(raw_grid_sizes, raw_merge_sizes):
                if grid_size is None or merge_size is None:
                    continue
                grid_size = grid_size.data
                merge_size = merge_size.data
                if not torch.all(grid_size[1:] % merge_size == 0):
                    warnings.warn(
                        f"Grid size {grid_size} is not divisible by merge size."
                    )
                if grid_size[0] == 1:
                    grid_sizes.append(grid_size[1:] / merge_size)
                elif grid_size[0] > 1:
                    grid_sizes.extend([grid_size[1:] / merge_size] * grid_size[0])
            return grid_sizes

        if "image" not in out_mm_kwargs:
            return []

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        placeholder = tokenizer.get_vocab()[DEFAULT_IMAGE_TOKEN]
        grid_sizes = _get_downsampled_grid_sizes(out_mm_kwargs)

        def get_replacement(item_idx: int):
            num_tokens = hf_processor._get_visual_seq_len(grid_sizes[item_idx])
            return [placeholder] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[placeholder],
                replacement=get_replacement,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_sizes = hf_inputs.get("grid_sizes", torch.empty((0, 3))).prod(-1)
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            grid_sizes=MultiModalFieldConfig.batched("image"),
            merge_sizes=MultiModalFieldConfig.batched("image"),
        )
