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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from modeling_qwen import *
from configuration_qwen import Qwen2Config

from transformers.modeling_outputs import CausalLMOutputWithPast

from llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class GroundQwenConfig(Qwen2Config):
    model_type = "ground_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = GroundQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)

class GroundQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = GroundQwenConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaQwenModel):
            module.gradient_checkpointing = value

    def forward_grounding(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        qs_ids: Optional[torch.LongTensor] = None,
        qs_mask: Optional[torch.Tensor] = None,
        time_labels: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        projector: Optional[torch.LongTensor] = None,
        select_layer: Optional[int] = None,
        return_simi: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
    
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                clip_embeds,
                qs_embeds,
                qs_mask,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                qs_ids,
                qs_mask,
                past_key_values,
                labels,
                images,
                projector
            )
        if isinstance(labels, tuple):
            labels, indicators = labels
        else:
            indicators = None

        loss, similarity, global_memory, clip_memory = super().forward_grounding_hm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            clip_embeds=torch.stack(clip_embeds, dim=0),
            qs_embeds=qs_embeds,
            qs_mask=qs_mask,
            labels=labels,
            time_labels=time_labels,
            indicators=indicators,
            return_simi=return_simi,
            select_layer=100,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return similarity, global_memory, clip_memory

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        qs_ids: Optional[torch.LongTensor] = None,
        qs_mask: Optional[torch.Tensor] = None,
        time_labels: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        projector: Optional[torch.LongTensor] = None,
        select_layer: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
    
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                clip_embeds,
                qs_embeds,
                qs_mask,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                qs_ids,
                qs_mask,
                past_key_values,
                labels,
                images,
                projector
            )
        if isinstance(labels, tuple):
            labels, indicators = labels
        else:
            indicators = None

        loss, similarity = super().forward_grounding_hm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            clip_embeds=torch.stack(clip_embeds, dim=0),
            qs_embeds=qs_embeds,
            qs_mask=qs_mask,
            labels=labels,
            time_labels=time_labels,
            indicators=indicators,
            select_layer=100,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return CausalLMOutputWithPast(loss=loss, past_key_values=past_key_values)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, indicators=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, indicators=indicators, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("ground_qwen", GroundQwenConfig)
AutoModelForCausalLM.register(GroundQwenConfig, GroundQwenForCausalLM)
