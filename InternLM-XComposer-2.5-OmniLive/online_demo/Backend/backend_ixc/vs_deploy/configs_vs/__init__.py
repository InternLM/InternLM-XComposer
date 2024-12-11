from lmdeploy.pytorch.models.module_map import MODULE_MAP

from .config_ground_qwen import GroundQwenModelConfigBuilder


def setup4lmdeploy():
    # llava_qwen from video streaming
    MODULE_MAP.update({
        'modeling_qwen.Qwen2Model':
        'configs_vs.rewrite_ground_qwen.PatchedQwenModel',
        'grounding_qwen.LlavaQwenModel':
        'configs_vs.rewrite_ground_qwen.PatchedQwenModel',
        'grounding_qwen.GroundQwenForCausalLM':
        'configs_vs.rewrite_ground_qwen.PatchedGroundQwenForCausalLM',
    })


__all__ = ['setup4lmdeploy', 'GroundQwenModelConfigBuilder']
