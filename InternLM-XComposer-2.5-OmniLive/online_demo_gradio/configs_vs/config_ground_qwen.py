# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import ModelConfig

from lmdeploy.pytorch.configurations.builder import AutoModelConfigBuilder


class GroundQwenModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] == 'GroundQwenForCausalLM'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build llava qwen."""
        from transformers import AutoConfig
        from grounding_qwen import GroundQwenForCausalLM

        groundqwen_hf_cfg = hf_config
        hidden_size = groundqwen_hf_cfg.hidden_size
        num_attention_heads = groundqwen_hf_cfg.num_attention_heads
        head_dim = hidden_size // num_attention_heads
        num_key_value_heads = groundqwen_hf_cfg.num_key_value_heads or num_attention_heads
        bos_token_id = 151643
        eos_token_id = 151645
        hf_config.torch_dtype = 'float16'
        return ModelConfig(
            hidden_size=hidden_size,
            num_layers=groundqwen_hf_cfg.num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            head_dim=head_dim,
            vocab_size=groundqwen_hf_cfg.vocab_size,
            hf_config=hf_config,
            auto_model_cls=GroundQwenForCausalLM,
            unused_modules=[
                'lm_head',
                'model.vision_tower',
                'model.mm_projector'
            ])
