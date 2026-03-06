# Copyright (c) Penguin-VL team at Tencent AI Lab
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


from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoImageProcessor,
                          Qwen3Config, Qwen3ForCausalLM, Qwen3Model)
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .loss import cross_entropy_loss
from .processor import PenguinVLBaseProcessor
from .vlm_arch import VLMMetaForCausalLM, VLMMetaModel
from .penguinvl_encoder import PenguinVLImageProcessor
from penguinvl.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN

CHAT_TEMPLATE = """
{%- set identifier = 'im' %}
{% for message in messages %}
    {% if message['role'] == 'stream' %}
        {% set identifier = 'stream' %}
    {% else %}
        {% set identifier = 'im' %}
    {% endif %}
    {% if message['role'] is not none %}
        {{- '<|' + identifier + '_start|>' + message['role'] + '\n' -}}
    {% endif %}
    {% if message['content'] is string %}
        {{- message['content'] + '<|' + identifier + '_end|>\n' -}}
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% if 'time' in content %}
                    {{- 'Time ' + content['time'] | round(1) | string + 's: ' -}}
                {% endif %}
                {{- image_token + '\n' -}}
            {% elif content['type'] == 'video' or 'video' in content or 'video_url' in content %}
                {% for i in range(content['num_frames']) %}
                    {% if 'timestamps' in content and content['timestamps']|length > 0 %}
                        {{- 'Time ' + content['timestamps'][i] | round(1) | string + 's:' -}}
                    {% endif %}
                    {% if i < content['num_frames'] - 1 %}
                        {{- image_token + ',' -}}
                    {% else %}
                        {{- image_token + '\n' -}}
                    {% endif %}
                {% endfor %}
            {% elif content['type'] == 'text' or 'text' in content %}
                {{- content['text'] -}}
            {% endif %}
        {% endfor %}
        {% if message['role'] is not none %}
            {{- '<|' + identifier + '_end|>\n' -}}
        {% endif %}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' -}}
    {% if not add_think_prompt %}
        {{- '<think>\n\n</think>\n\n' -}}
    {% endif %}
{% endif %}
"""


class PenguinVLQwen3Processor(PenguinVLBaseProcessor):

    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    chat_template = CHAT_TEMPLATE

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_merge_size: int = 1,
        video_merge_size: int = 2,
        fps=1,
        max_frames=180,
        **kwargs
    ):
        super().__init__(image_processor, tokenizer, chat_template, **kwargs)
        self.generation_prompt = self._infer_generation_prompt(add_think_prompt=False)
        self.generation_prompt_ids = self.tokenizer.encode(self.generation_prompt, return_tensors="pt")
        self.generation_prompt_length = len(self.generation_prompt_ids[0])

        self.generation_prompt_think = self._infer_generation_prompt(add_think_prompt=True)
        self.generation_prompt_think_ids = self.tokenizer.encode(self.generation_prompt_think, return_tensors="pt")
        self.generation_prompt_think_length = len(self.generation_prompt_think_ids[0])

        self.non_thinking_prompt = self.generation_prompt.replace(self.generation_prompt_think, "")

    def _infer_generation_prompt(self, add_think_prompt=False):
        pseudo_message = [{"role": "user", "content": ""}]
        instruction = self.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=True, add_think_prompt=add_think_prompt)
        conversation = self.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=False)
        return instruction.replace(conversation, "")

    def _is_non_thinking_response(self, message):
        if message["role"] != "assistant":
            return False
        assert isinstance(message["content"], str), f"Response must be a string, but got: {message['content']}"
        return self.think_start_token not in message["content"]

    def _process_text_with_label(
        self,
        text: List[Dict],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        assert kwargs.pop("return_tensors", "pt") == "pt", "Only PyTorch tensors are supported when return_labels=True."
        assert isinstance(text[0], dict), "When return_labels=True, text must be a list of messages."

        input_ids_list = []
        targets_list = []
        image_idx = 0

        for message_idx, message in enumerate(text):
            # NOTE: For non-thinking response, we need to add the '<think>\n\n</think>\n\n' prefix
            if self._is_non_thinking_response(message):
                message["content"] = self.non_thinking_prompt + message["content"]
                generation_prompt_length = self.generation_prompt_length
            else:
                generation_prompt_length = self.generation_prompt_think_length

            # 1. set chat template
            prompt = self.apply_chat_template([message], tokenize=False, add_generation_prompt=False)

            # 2. append image tokens
            prompt_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
            prompt = []
            for chunk_idx in range(len(prompt_chunks) - 1):
                prompt.append(prompt_chunks[chunk_idx])
                thw = grid_sizes[image_idx]
                prompt.append(DEFAULT_IMAGE_TOKEN * thw.prod().long())
                image_idx += 1
            prompt.append(prompt_chunks[-1])
            prompt = "".join(prompt)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            input_ids_list.append(input_ids)

            targets = torch.full_like(input_ids, IGNORE_INDEX)
            if message["role"] == "assistant" or message["role"] is None:
                targets[generation_prompt_length:-1] = input_ids[generation_prompt_length:-1].clone()

                # NOTE: mask out image tokens
                vision_mask = input_ids == self.image_token_id
                targets[vision_mask] = IGNORE_INDEX
                vision_indices = torch.nonzero(vision_mask, as_tuple=True)[0]
                targets[vision_indices + 1] = IGNORE_INDEX

                # NOTE: mask out <think> or <think>\n
                think_mask = targets == self.think_start_token_id
                targets[think_mask] = IGNORE_INDEX
                think_indices = torch.nonzero(think_mask, as_tuple=True)[0]
                newline_mask = torch.zeros_like(think_mask)
                newline_mask[think_indices + 1] = targets[think_indices + 1] == self.newline_token_id
                targets[newline_mask] = IGNORE_INDEX

            targets_list.append(targets)

        assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        text_inputs = {
            "input_ids": torch.cat(input_ids_list),
            "labels": torch.cat(targets_list),
        }

        return text_inputs


class PenguinVLQwen3Config(Qwen3Config):
    model_type = "penguinvl_qwen3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "penguinvl_qwen3"


class PenguinVLQwen3Model(VLMMetaModel, Qwen3Model):
    config_class = PenguinVLQwen3Config

    def __init__(self, config: PenguinVLQwen3Config):
        super(PenguinVLQwen3Model, self).__init__(config)


class PenguinVLQwen3ForCausalLM(Qwen3ForCausalLM, VLMMetaForCausalLM):
    config_class = PenguinVLQwen3Config

    def __init__(self, config, **kwargs):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = PenguinVLQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if self.config.use_vision_teacher:
            self.vision_distill_layer = nn.Linear(self.model.vision_encoder.hidden_size, self.model.vision_encoder_teacher.hidden_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # NOTE: arguments are copied from transformers==4.51.3
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        vision_mask = None

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                vision_mask,
                mm_features_teacher,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        loss, logits, image_loss, teacher_loss, relation_loss, caption_loss = None, None, None, None, None, None
        if labels is not None:
            loss = cross_entropy_loss(
                hidden_states=hidden_states,
                lm_head=self.lm_head,
                position_ids=position_ids,
                labels=labels,
                reduction_scope=self.config.loss_reduction_scope,
                **kwargs,
            )
            caption_loss = loss.detach()
            if self.config.use_reconstruct:
                student_features = mm_features_teacher[0]
                teacher_features = mm_features_teacher[1]
                # rgb_pred = self.vision_head(student_features)
                # image_loss = torch.nn.functional.l1_loss(pixel_values.detach(), rgb_pred)
                # loss = loss + image_loss * 2
                if self.config.use_vision_teacher:
                    student_features = self.vision_distill_layer(student_features)
                    student_norm = torch.nn.functional.normalize(student_features, p=2, dim=-1)
                    teacher_norm = torch.nn.functional.normalize(teacher_features, p=2, dim=-1)
                    teacher_loss = 1 - torch.nn.functional.cosine_similarity(student_features, teacher_features, dim=-1).mean()
                    teacher_loss = teacher_loss + torch.nn.functional.smooth_l1_loss(student_features, teacher_features, beta=0.001)
                    
                    relation_loss = torch.nn.functional.smooth_l1_loss(
                        torch.einsum('id,jd->ij', student_norm, student_norm),
                        torch.einsum('id,jd->ij', teacher_norm, teacher_norm),
                        beta=0.001
                    )

                    loss = loss + (teacher_loss + relation_loss) * (3 - kwargs.get("current_epoch", 0)) / kwargs.get("num_items_in_batch", 1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        output.caption_loss = caption_loss
        if image_loss is not None:
            output.image_loss = image_loss
        if teacher_loss is not None:
            output.teacher_loss = teacher_loss
        if relation_loss is not None:
            output.relation_loss = relation_loss

        return output

    @torch.no_grad()
    def generate(
        self,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                vision_mask,
                mm_features_teacher,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("penguinvl_qwen3", PenguinVLQwen3Config)
AutoModelForCausalLM.register(PenguinVLQwen3Config, PenguinVLQwen3ForCausalLM)
AutoProcessor.register(PenguinVLQwen3Config, PenguinVLQwen3Processor)
AutoImageProcessor.register(PenguinVLQwen3Config, PenguinVLImageProcessor)
