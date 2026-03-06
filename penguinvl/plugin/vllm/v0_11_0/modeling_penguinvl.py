"""PenguinVL Qwen3 model for vLLM inference.

Integrates the PenguinVL vision encoder, MLP projector, and Qwen3 LLM backbone
for efficient multimodal inference through vLLM's serving framework.
"""

import re
from typing import List, Optional, Tuple, Set, Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig, AutoConfig
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig, QuantizationConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.models.interfaces import SupportsMultiModal, MultiModalEmbeddings
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .modeling_penguinvl_encoder import PenguinVLVisionEncoderModel
from .processing_penguinvl import (
    PenguinVLMultiModalProcessor, PenguinVLProcessingInfo,
    PenguinVLDummyInputsBuilder,
)


class MlpGeluProjector(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        projector_type = getattr(
            config, 'vision_projector_type',
            getattr(config, 'mm_projector_type', 'mlp2x_gelu')
        )
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))
        assert mlp_depth == 2, "Only support 2-layer MLP for now"

        self.readout = nn.Sequential(
            ColumnParallelLinear(
                config.vision_encoder_config.hidden_size,
                config.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.readout.0",
            ),
            nn.GELU(),
            RowParallelLinear(
                config.hidden_size,
                config.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.readout.2",
            ),
        )

    def forward(self, x):
        x, _ = self.readout[0](x)
        x = self.readout[1](x)
        x, _ = self.readout[2](x)
        return x


@MULTIMODAL_REGISTRY.register_processor(
    processor=PenguinVLMultiModalProcessor,
    info=PenguinVLProcessingInfo,
    dummy_inputs=PenguinVLDummyInputsBuilder,
)
class PenguinVLQwen3ForCausalLM(Qwen3ForCausalLM, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config.hf_config

        vision_encoder_prefix = "vision_encoder"
        mm_projector_prefix = "mm_projector"
        if prefix:
            vision_encoder_prefix = f"{prefix}.{vision_encoder_prefix}"
            mm_projector_prefix = f"{prefix}.{mm_projector_prefix}"

        if getattr(model_config, "vision_encoder", None) is not None:
            from penguinvl.model.penguinvl_encoder import PenguinVLVisionEncoderConfig
            model_config.vision_encoder_config = (
                PenguinVLVisionEncoderConfig.from_pretrained(model_config.vision_encoder)
            )
            self.hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
                "model.vision_encoder.vision_encoder.embeddings.": "vision_encoder.embeddings.",
                "model.vision_encoder.vision_encoder.encoder.": "vision_encoder.encoder.",
                "model.vision_encoder.vision_encoder.norm.": "vision_encoder.norm.",
                "model.vision_projector.": "mm_projector.",
            })
        else:
            if not isinstance(model_config.vision_encoder_config, PretrainedConfig):
                from penguinvl.model.penguinvl_encoder import PenguinVLVisionEncoderConfig
                model_config.vision_encoder_config = (
                    PenguinVLVisionEncoderConfig.from_dict(model_config.vision_encoder_config)
                )
            self.hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
                "model.vision_encoder.embeddings.": "vision_encoder.embeddings.",
                "model.vision_encoder.encoder.": "vision_encoder.encoder.",
                "model.vision_encoder.norm.": "vision_encoder.norm.",
                "model.vision_projector.": "mm_projector.",
            })

        self.vision_encoder = PenguinVLVisionEncoderModel(
            model_config.vision_encoder_config,
            quant_config=vllm_config.quant_config,
            prefix=vision_encoder_prefix,
        )

        self.mm_projector = MlpGeluProjector(
            model_config,
            quant_config=vllm_config.quant_config,
            prefix=mm_projector_prefix,
        )

    def get_multimodal_embeddings(
        self,
        pixel_values: torch.Tensor = None,
        grid_sizes: torch.Tensor = None,
        merge_sizes: torch.Tensor = None,
    ) -> Optional[MultiModalEmbeddings]:
        if pixel_values is None:
            return None

        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.ndim == 3:
                pixel_values = torch.cat(list(pixel_values))
            elif pixel_values.ndim != 2:
                raise ValueError(
                    f"Unexpected pixel_values ndim: {pixel_values.ndim}"
                )
        elif isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values)
        else:
            raise ValueError(f"Unexpected pixel_values type: {type(pixel_values)}")

        pixel_values = pixel_values.type(self.vision_encoder.dtype)
        grid_sizes = torch.cat([x for x in grid_sizes])
        merge_sizes = torch.cat([x for x in merge_sizes])

        mm_features = self.vision_encoder(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )

        return mm_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        image_selected = (input_ids == self.config.image_token_index)
        input_ids[image_selected] = 0
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds[image_selected] = self.mm_projector(
                torch.cat(multimodal_embeddings)
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None,
        grid_sizes: torch.Tensor = None,
        merge_sizes: torch.Tensor = None,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            if pixel_values is None:
                inputs_embeds = None
            else:
                multimodal_embeddings = self.get_multimodal_embeddings(
                    pixel_values=pixel_values,
                    grid_sizes=grid_sizes,
                    merge_sizes=merge_sizes,
                )
                inputs_embeds = self.get_input_embeddings(
                    input_ids,
                    multimodal_embeddings=multimodal_embeddings,
                )
                input_ids = None

        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    _IGNORED_PREFIXES = (
        "model.audio_encoder.",
        "model.audio_projector.",
        "model.vision_encoder.vision_encoder.encoder.rotary_emb.",
        "model.vision_encoder.encoder.rotary_emb.",
    )

    def _filter_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if any(name.startswith(p) for p in self._IGNORED_PREFIXES):
                continue
            yield name, tensor

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            self._filter_weights(weights), mapper=self.hf_to_vllm_mapper
        )
