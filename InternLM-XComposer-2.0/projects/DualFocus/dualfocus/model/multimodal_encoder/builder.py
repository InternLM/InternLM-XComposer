import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    if hasattr(vision_tower_cfg, 'architectures') and 'Share4V' in vision_tower_cfg.architectures[0]:
        vision_tower = '/mnt/petrelfs/caoyuhang/LLaVA/models/ShareGPT4V-7B_Pretrained_vit-large336-l12'
    else:
        vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
