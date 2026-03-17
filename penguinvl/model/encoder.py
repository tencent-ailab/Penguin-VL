# Copyright (c) Penguin-VL team at Tencent AI Lab
import os

import torch
import torch.nn as nn
import math
from transformers import (CLIPImageProcessor, CLIPVisionConfig,
                          CLIPVisionModel, SiglipImageProcessor,
                          SiglipVisionConfig, SiglipVisionModel,
                          AutoConfig, AutoModel, AutoImageProcessor)

from .penguinvl_encoder import (PenguinVLVisionEncoderConfig, PenguinVLVisionEncoderModel, PenguinVLImageProcessor)


class CLIPVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model()
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = CLIPVisionModel.from_pretrained(self.vision_encoder_name,
                                                            attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        images = torch.cat(images)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model()
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = SiglipVisionModel.from_pretrained(self.vision_encoder_name,
                                                              attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        images = torch.cat(images)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class PenguinVLVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.args = args

        self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
        if not delay_load:
            self.load_model(self.args)
        else:
            self.cfg_only = PenguinVLVisionEncoderConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self, args):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = PenguinVLImageProcessor.from_pretrained(self.vision_encoder_name)

        # merge_size is fixed to 1 for STAGE1, STAGE1.5, STAGE2, STAGE3 in encoder and can be modified in connector.
        self.cfg_only = PenguinVLVisionEncoderConfig.from_pretrained(self.vision_encoder_name)

        try:
            self.vision_encoder = PenguinVLVisionEncoderModel.from_pretrained(
                self.vision_encoder_name,
                torch_dtype=args.torch_dtype,
                attn_implementation=self.attn_implementation)
        except Exception as e:
            print(f"Error loading PenguinVLVisionEncoderModel: {e}, trying to create a new model from config.")
            # Create new model with the configuration
            self.vision_encoder = PenguinVLVisionEncoderModel(self.cfg_only)
            self.vision_encoder.apply(self.vision_encoder._init_weights)
            self.vision_encoder = self.vision_encoder.to(dtype=args.torch_dtype)

        self.is_loaded = True

    def forward(self, pixel_values, grid_sizes, merge_sizes, **kwargs):
        image_features = self.vision_encoder(pixel_values, grid_sizes, merge_sizes)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return -1

    @property
    def num_patches_per_side(self):
        return -1

    @property
    def image_size(self):
        return -1
    

class Videollama3VisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.args = args

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model(self.args)
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = AutoConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self, args):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_encoder_name)

        # merge_size is fixed to 1 for STAGE1, STAGE1.5, STAGE2, STAGE3 in encoder and can be modified in connector.
        self.cfg_only = AutoConfig.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = AutoModel.from_pretrained(
            self.vision_encoder_name,
            trust_remote_code=True,
            torch_dtype=args.torch_dtype,
            attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def forward(self, pixel_values, grid_sizes, merge_sizes, **kwargs):
        image_features = self.vision_encoder(pixel_values, grid_sizes, merge_sizes)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return -1

    @property
    def num_patches_per_side(self):
        return -1

    @property
    def image_size(self):
        return -1
    

def build_vision_encoder(vision_encoder_cfg, **kwargs):
    vision_encoder = getattr(vision_encoder_cfg, 'mm_vision_encoder', getattr(vision_encoder_cfg, 'vision_encoder', None))

    if  'clip' in vision_encoder:
        vision_encoder = CLIPVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
    elif 'siglip' in vision_encoder:
        vision_encoder = SiglipVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
    elif 'penguin' in vision_encoder.lower():
        vision_encoder = PenguinVLVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
    elif 'videollama3' in vision_encoder.lower() or 'vl3-siglip-navit' in vision_encoder.lower():
        vision_encoder = Videollama3VisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision encoder: {vision_encoder}')

    return vision_encoder
