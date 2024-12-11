# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from lmdeploy.pytorch.kernels import fill_kv_cache, paged_attention_fwd
from lmdeploy.pytorch.weight_loader.dist_utils import (
    colwise_parallelize_linear, colwise_split_parallelize_linear,
    rowwise_parallelize_linear)

class PatchedQwenModel(nn.Module):

    def forward(self,
                input_ids: torch.LongTensor,
                past_key_values: Optional[Union[torch.FloatTensor]] = None,
                **kwargs) -> torch.FloatTensor:
        """rewrite forward of QwenModel."""
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        inputs_embeds = self.embed_tokens(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            # multi-modality
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        hidden_states = inputs_embeds

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                past_key_values=past_key_values[i],
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        return hidden_states


class PatchedGroundQwenForCausalLM(nn.Module):

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        qwen_model = self.model
        hidden_states = qwen_model(input_ids,
                                  past_key_values=past_key_values)
        return CausalLMOutputWithPast(logits=hidden_states)
