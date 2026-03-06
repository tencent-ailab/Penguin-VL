# Copyright (c) Penguin-VL team at Tencent AI Lab
# Adapted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from copy import deepcopy

from .encoder import build_vision_encoder
from .projector import build_vision_projector


class VLMMetaModel:

    def __init__(self, config):
        super(VLMMetaModel, self).__init__(config)

        if hasattr(config, "vision_encoder"):
            delay_load = not hasattr(config, "mm_vision_encoder")
            self.vision_encoder = build_vision_encoder(config, delay_load=delay_load)
            if getattr(config, "use_vision_teacher", False):
                temp_config = deepcopy(config)
                temp_config.vision_encoder = temp_config.vision_encoder_teacher
                print("Building vision teacher:", temp_config.vision_encoder)
                self.vision_encoder_teacher = build_vision_encoder(temp_config, delay_load=delay_load)
                del temp_config
            self.vision_projector = build_vision_projector(config, self.vision_encoder.hidden_size)

    def get_vision_encoder(self):
        vision_encoder = getattr(self, 'vision_encoder', None)
        if type(vision_encoder) is list:
            vision_encoder = vision_encoder[0]
        return vision_encoder

    def get_vision_teacher(self):
        vision_encoder_teacher = getattr(self, 'vision_encoder_teacher', None)
        return vision_encoder_teacher

    def get_vision_projector(self):
        return self.vision_projector

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_encoder = model_args.vision_encoder
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        self.config.mm_vision_encoder = vision_encoder

        if self.get_vision_encoder() is None:
            vision_encoder = build_vision_encoder(model_args)
            if getattr(model_args, "use_vision_teacher", False):
                temp_args = deepcopy(model_args)
                temp_args.vision_encoder = temp_args.vision_encoder_teacher
                print("Building vision teacher:", temp_args.vision_encoder)
                self.vision_encoder_teacher = build_vision_encoder(temp_args)
                del temp_args

            if fsdp is not None and len(fsdp) > 0:
                self.vision_encoder = [vision_encoder]
            else:
                self.vision_encoder = vision_encoder
        else:
            self.vision_encoder.load_model(self.vision_encoder.cfg_only)
            if getattr(model_args, "use_vision_teacher", False):
                print("Loading vision teacher...")
                self.vision_encoder_teacher.load_model(self.vision_encoder_teacher.cfg_only)
            vision_encoder = self.vision_encoder

        self.config.vision_projector_type = getattr(model_args, 'vision_projector_type', 'mlp2x_gelu')
        self.config.vision_hidden_size = vision_encoder.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'vision_projector', None) is None:
            self.vision_projector = build_vision_projector(self.config, self.vision_encoder.hidden_size)
        else:
            # In case it is frozen by LoRA
            for p in self.vision_projector.parameters():
                p.requires_grad = True


class VLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_vision_teacher(self):
        return self.get_model().get_vision_teacher()

    def get_vision_projector(self):
        return self.get_model().get_vision_projector()

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
    ) -> torch.FloatTensor:
        mm_features = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        mm_features_teacher = None
        if self.get_model().get_vision_teacher() is not None and self.training:
            with torch.no_grad():
                mm_features_teacher = self.get_model().get_vision_teacher()(
                    pixel_values=pixel_values,
                    grid_sizes=grid_sizes,
                    merge_sizes=merge_sizes,
                )
        mm_features_projector = self.get_model().vision_projector(mm_features)
        if mm_features_teacher is not None:
            mm_features_teacher = mm_features_teacher.detach()
        return mm_features_projector, (mm_features, mm_features_teacher)

    def _get_valid_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
    ):
        valid_masks = []
        for num_patches, modal in zip(batched_num_patches, modals):
            valid_mask = torch.full((num_patches, ), modal != "text", dtype=torch.bool, device=mm_features.device)
            valid_masks.append(valid_mask)
        mm_features = mm_features[torch.cat(valid_masks)]
        return mm_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
    ):
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels, None, None

        # 1. flatten text inputs
        B, N = input_ids.shape
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. embed visual tokens
        image_selected, mm_features_teacher = None, None
        if pixel_values is not None:
            # 2.1 encode images
            batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()
            mm_features, mm_features_teacher = self.encode_images(pixel_values, grid_sizes, merge_sizes)
            mm_features = mm_features.to(input_ids.device)
            mm_features_teacher = [(item.to(input_ids.device) if item is not None else None) for item in mm_features_teacher]
            mm_features = self._get_valid_visual_tokens(mm_features, batched_num_patches, modals)

            # 2.2 get image selected
            image_selected = (input_ids == self.config.image_token_index)
            input_ids[image_selected] = 0

            num_vision_tokens = image_selected.sum()
            if mm_features.size(0) != num_vision_tokens:
                print(f"Number of vision_features ({mm_features.size(0)}) does not match the number of image tokens ({num_vision_tokens}). Please check the inputs.")

        # 3. replace multimodal tokens with features
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()
        if image_selected is not None:
            inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features

        # 4. reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)
        image_selected = image_selected.view(B, -1) if image_selected is not None else None

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels, image_selected, mm_features_teacher
