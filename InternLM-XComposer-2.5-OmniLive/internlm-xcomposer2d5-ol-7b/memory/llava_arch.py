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

import numpy as np
import torch
import torch.nn as nn

from builder_encoder import build_vision_tower
from builder_projector import build_vision_projector

from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector = build_vision_projector(self.config)
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, base_mode=False):
        clip_features = self.get_model().get_vision_tower()(images)
        if not base_mode:
            clip_features = self.mix_spatial_tokens(clip_features)
        else:
            clip_features = self.mix_spatial_tokens(clip_features)
        image_features = self.get_model().mm_projector(clip_features)
        return clip_features, image_features
    
    def extract_images(self, images):
        image_features_list = []
        block_size = 16
        for i in range(0, images.shape[0], block_size):
            image_features = self.get_model().get_vision_tower()(images[i: i+block_size])
            image_features_list.append(image_features)
        image_features = torch.cat(image_features_list, dim=0)
        assert image_features.shape[0] == images.shape[0]
        return image_features
    
    def project_features(self, features):
        proj_features = self.get_model().mm_projector(features)
        return proj_features
    
    def mix_spatial_tokens(self, features):
        # features b n c
        # output b n//4 4c
        b, n, c = features.shape
        h = int(np.sqrt(n))
        features = features.view(b, h//2, 2, h//2, 2, c).permute(0, 1, 3, 2, 4, 5).contiguous()
        features = features.view(b, n//4, 4*c).contiguous()
        return features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, qs_ids, qs_mask, past_key_values, labels, images, projector
    ):
        vision_tower = self.get_vision_tower()
        if hasattr(self.get_model().mm_projector, 'num_slot'):
            base_mode = False
            num_slot = self.get_model().mm_projector.num_slot
        elif hasattr(self.get_model().mm_projector, 'resolution'):
            base_mode = False
            pool_num = self.get_model().mm_projector.pool_num
            resolution = self.get_model().mm_projector.resolution + pool_num
        else:
            base_mode = True
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if isinstance(past_key_values, tuple) and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            elif past_key_values is not None and past_key_values.seqlen_offset>0:
                target_shape = past_key_values.seqlen_offset + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, None, None, None, labels

        ''' using pre-extraced video features
        if type(images) is list:
            concat_images = []
            concat_features = []
            modality_indicators = []
            for image, projector_type in zip(images, projector):
                if image.ndim == 2: # pre-extracted feature
                    concat_features.append(image)
                    modality_indicators.append(2)
                elif image.ndim == 3: # single image
                    concat_images.append(image.unsqueeze(0))
                    modality_indicators.append(1)
                elif image.ndim == 4: # multiple frames
                    concat_images.append(image)
                    modality_indicators.append(2)
            concat_images = torch.cat(concat_images, dim=0)
            concat_features = torch.stack(concat_features, dim=0)
            concat_images = self.extract_images(concat_images)
            concat_images, concat_features = self.project_features([concat_images, concat_features])
            # concat_combine = torch.cat([concat_images.reshape(-1, concat_images.shape[-1]), concat_features.reshape(-1, concat_features.shape[-1])], dim=0)
            # concat_combine = self.project_features(concat_combine)
            # concat_images = concat_combine[:concat_images.shape[0]*concat_images.shape[1]].contiguous().view(*concat_images.shape[:2], -1)
            # concat_features = concat_combine[concat_images.shape[0]*concat_images.shape[1]:].contiguous().view(*concat_features.shape[:2], -1)
            image_features = []
            image_index = 0
            feature_index = 0
            for image in images:
                if image.ndim == 2:
                    image_features.append(concat_features[feature_index])
                    feature_index += 1
                elif image.ndim == 3:
                    image_features.append(concat_images[image_index])
                    image_index += 1
                elif image.ndim == 4:
                    image_features.append(concat_images[image_index: image_index+image.shape[0]].flatten(0, 1))
                    image_index += image.shape[0]
            image_features = [x.to(self.device) for x in image_features]
        '''

        if qs_ids is not None:
            qs_embeds = self.get_model().embed_tokens(qs_ids)
        else:
            qs_embeds = None

        assert len(images) == len(input_ids)
        if type(images) is list:
            concat_images = []
            concat_videos = []
            modality_indicators = []
            for image in images:
                if image.ndim == 3: # single image
                    concat_images.append(image.unsqueeze(0))
                    modality_indicators.append(1)
                elif image.ndim == 4: # multiple frames
                    concat_videos.append(image)
                    modality_indicators.append(2)
            concat_images = torch.cat(concat_images, dim=0) # n c h w
            concat_videos = torch.stack(concat_videos, dim=0) # n t c h w
            mix_image_video = torch.cat([concat_images, concat_videos.view(-1, *concat_videos.shape[2:])], dim=0) # m c h w
            mix_image_video = self.extract_images(mix_image_video) # m k c
            if not base_mode:
                mix_image_video = self.mix_spatial_tokens(mix_image_video)
                concat_images = mix_image_video[:concat_images.shape[0]].contiguous() # n k c
                concat_videos = mix_image_video[concat_images.shape[0]:].contiguous().view(
                    concat_videos.shape[0], concat_videos.shape[1]*mix_image_video.shape[1], mix_image_video.shape[2]) # n, tk, c
            else:
                mix_image_video = self.mix_spatial_tokens(mix_image_video)
                concat_images = mix_image_video[:concat_images.shape[0]].contiguous() # n k c
                concat_videos = mix_image_video[concat_images.shape[0]:].contiguous().view(
                    concat_videos.shape[0], concat_videos.shape[1], mix_image_video.shape[1], mix_image_video.shape[2]) # n, t, k, c
            clip_features = []
            image_index = 0
            video_index = 0
            for image in images:
                if image.ndim == 3:
                    clip_features.append(concat_images[image_index])
                    image_index += 1
                elif image.ndim == 4:
                    clip_features.append(concat_videos[video_index])
                    video_index += 1
            clip_features = [x.to(self.device) for x in clip_features]

            concat_images, concat_videos = self.project_features([concat_images, concat_videos])
            image_features = []
            image_index = 0
            video_index = 0
            for image in images:
                if image.ndim == 3:
                    image_features.append(concat_images[image_index])
                    image_index += 1
                elif image.ndim == 4:
                    image_features.append(concat_videos[video_index])
                    video_index += 1
            image_features = [x.to(self.device) for x in image_features]
        elif images.ndim == 5:
            modality_indicators = [2 for _ in range(images.shape[0])]
            concat_images = images.view(-1, *images.shape[2:]) # nt c h w
            image_features = self.extract_images(concat_images)

            # image_features = image_features.view(images.shape[0], images.shape[1], image_features.shape[1], image_features.shape[2]) # n t k c
            # time_token = torch.mean(image_features, dim=2) # n t c
            # spatial_token = torch.mean(image_features, dim=1) # n k c
            # token = torch.cat([time_token, spatial_token], dim=1)
            # output = self.project_features(token) # n t+k c
            # image_features = [x.to(self.device) for x in output]
            
            if not base_mode:
                image_features = self.mix_spatial_tokens(image_features) # nt k c
                image_features = image_features.view(images.shape[0], images.shape[1]*image_features.shape[1], image_features.shape[2]) # n tk c
            else:
                image_features = self.mix_spatial_tokens(image_features) # nt k c
                image_features = image_features.view(images.shape[0], images.shape[1], image_features.shape[1], image_features.shape[2]) # n t k c
            clip_features = [x.to(self.device) for x in image_features]
            image_features = self.project_features(image_features)
            image_features = [x.to(self.device) for x in image_features]
        elif images.ndim == 3:
            modality_indicators = [2 for _ in range(images.shape[0])]
            image_features = self.project_features(images).to(self.device)
        else:
            modality_indicators = [1 for _ in range(images.shape[0])]
            clip_features, image_features = self.encode_images(images, base_mode)
            clip_features = [x.to(self.device) for x in clip_features]
            image_features = [x.to(self.device) for x in image_features]

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        indicators = torch.zeros_like(input_ids)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        indicators = [cur_indicators[cur_attention_mask] for cur_indicators, cur_attention_mask in zip(indicators, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_indicators = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_indicators.append(indicators[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_indicators = indicators[batch_idx]
            cur_indicators_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_indicators_noim.append(cur_indicators[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_indicators = []

            if True: # stage 2
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    cur_new_indicators.append(cur_indicators_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        if hasattr(self.get_model().mm_projector, 'resolution'):
                            assert (cur_image_features.shape[0]-1) % resolution == 0
                            num_slot = (cur_image_features.shape[0]-1) // resolution * pool_num
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        tmp = modality_indicators[batch_idx]*torch.ones((cur_image_features.shape[0],), device=cur_indicators.device, dtype=cur_indicators.dtype)
                        try:
                            tmp[-num_slot-1: -1] = 100
                            tmp[-1] = 200
                        except:
                            pass
                        cur_new_indicators.append(tmp)
                        # cur_new_indicators.append(modality_indicators[batch_idx]*torch.ones((cur_image_features.shape[0],), device=cur_indicators.device, dtype=cur_indicators.dtype))
                        # cur_new_indicators.append(torch.ones((self.config.n_slot,), device=cur_indicators.device, dtype=cur_indicators.dtype)+1)

            if False: # stage 1
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    cur_new_indicators.append(cur_indicators_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        if hasattr(self.get_model().mm_projector, 'resolution'):
                            assert cur_image_features.shape[0] % resolution == 0
                            num_slot = cur_image_features.shape[0] // resolution * pool_num
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        tmp = modality_indicators[batch_idx]*torch.ones((cur_image_features.shape[0],), device=cur_indicators.device, dtype=cur_indicators.dtype)
                        tmp[-num_slot:] = 100
                        cur_new_indicators.append(tmp)
                        # cur_new_indicators.append(modality_indicators[batch_idx]*torch.ones((cur_image_features.shape[0],), device=cur_indicators.device, dtype=cur_indicators.dtype))
                        # cur_new_indicators.append(torch.ones((self.config.n_slot,), device=cur_indicators.device, dtype=cur_indicators.dtype)+1)


            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_indicators = torch.cat(cur_new_indicators)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_indicators.append(cur_new_indicators)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_indicators = [x[:tokenizer_model_max_length] for x in new_indicators] 

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_indicators_padded = torch.zeros((batch_size, max_len), dtype=new_indicators[0].dtype, device=new_indicators[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_indicators) in enumerate(zip(new_input_embeds, new_labels, new_indicators)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_indicators_padded[i, -cur_len:] = cur_new_indicators
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_indicators_padded[i, :cur_len] = cur_new_indicators
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_indicators = new_indicators_padded
        # print('finish preparing labels multimodal')

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if base_mode:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
        else:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, clip_features, qs_embeds, qs_mask, (new_labels, new_indicators)

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
