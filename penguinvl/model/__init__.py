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


import os
import warnings
import shutil

import torch
from transformers import PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor

from .projector import load_mm_projector
from .penguinvl_encoder import PenguinVLVisionEncoderModel, PenguinVLVisionEncoderConfig
from .penguinvl_qwen3 import PenguinVLQwen3ForCausalLM, PenguinVLQwen3Config


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None

    # NOTE: auto device_map by default
    # if want to put model into a single device, you can set device_map={"": "cuda:0"}
    kwargs = {"device_map": device_map, **kwargs}

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = kwargs.pop('attn_implementation', "flash_attention_2") # default to flash_attention_2

    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else kwargs.pop('torch_dtype', torch.float16)

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # NOTE: High-version Transformers will report: """ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time."""
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype

    model_type = config.model_type if hasattr(config, "model_type") else kwargs.pop('model_type', "penguinvl_qwen3")

    if 'penguinvl_qwen3' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

        model = PenguinVLQwen3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)

    processor = None

    if "penguinvl_qwen3" in model_type:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False, token=token)
        tokenizer = processor.tokenizer

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
