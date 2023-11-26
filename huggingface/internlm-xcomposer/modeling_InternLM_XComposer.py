import copy
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

import contextlib

import torch.utils.checkpoint
from torch.nn import LayerNorm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from .modeling_perceive_sampler import BertConfig, BertLMHeadModel
from .modeling_vit import *
from .modeling_InternLM import *
from .modeling_utils import *

from transformers.utils import logging
logger = logging.get_logger(__name__)


class InternLMXComposerForCausalLM(PreTrainedModel):
    config_class = InternLMXComposerConfig
    _auto_class = "AutoModelForCausalLM"

    meta_instruction = """meta instruction
You are an AI assistant whose name is 浦语.
- 浦语 is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- 浦语 can understand and communicate fluently in the language chosen by the user such as English and 中文.
conversation
"""

    gen_config = dict(
        num_beams=5,
        do_sample=False,
        min_length=1,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1.0,
        max_new_tokens=500,
    )

    def __init__(self, config):
        super().__init__(config)

        print('Init VIT ... ', end='')
        self.visual_encoder = create_eva_vit_g()
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)
        print('Done')

        print('Init Perceive Sampler ... ', end='')
        with all_logging_disabled():
            self.Qformer, self.query_tokens = self.init_qformer(
                config.num_query_token, self.visual_encoder.num_features)
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.Qformer.cls = None
        print('Done')

        print('Init InternLM ... ', end='')
        self.flag_image_start = nn.Parameter(torch.zeros([1, 1, 4096]))
        self.flag_image_end = nn.Parameter(torch.zeros([1, 1, 4096]))
        self.flag_image_start.requires_grad = False
        self.flag_image_end.requires_grad = False

        internlm_lora = config.internlm_lora
        self.internlm_lora = internlm_lora
        setattr(InternLMForCausalLM, 'lora_cfg', internlm_lora)

        if int(torch.__version__[0]) == 1:
            self.internlm_model = InternLMForCausalLM._from_config(config).to(
                torch.float16)
        else:
            assert int(torch.__version__[0]) == 2
            # speed up init llm
            with torch.device('meta'):
                self.internlm_model = InternLMForCausalLM._from_config(config)
            self.internlm_model.to_empty(device='cpu').to(torch.float16)
            self.internlm_model.to(config.device)
        for n, m in self.internlm_model.named_modules():
            if 'lora' in n:
                m.float()

        self.internlm_proj = nn.Linear(self.Qformer.config.hidden_size,
                                       self.internlm_model.config.hidden_size)
        print('Done')

        self.vis_processor = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.tokenizer = None

        self.eoh = '<TOKENS_UNUSED_0>'  # end of human
        self.eoa = '<TOKENS_UNUSED_1>'  # end of assistant
        stop_words_ids = [
            torch.tensor([103027]).to(config.device),
            torch.tensor([103028]).to(config.device),
        ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])
        self.gen_config['stopping_criteria'] = stopping_criteria

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_qformer(cls,
                     num_query_token,
                     vision_width,
                     cross_attention_freq=2,
                     pretrain=True):
        encoder_config = BertConfig()
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0,
                                  std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def encode_img(self, image):
        if image is None:
            return None
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            image = self.vis_processor(image).unsqueeze(0).to(self.device)
        else:
            assert isinstance(image, torch.Tensor)
        device = image.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(
                self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1],
                                    dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1,
                                                    -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            inputs_internlm = self.internlm_proj(
                query_output.last_hidden_state)
            inputs_internlm = torch.cat([
                self.flag_image_start.expand(inputs_internlm.shape[0], -1, -1),
                inputs_internlm,
                self.flag_image_end.expand(inputs_internlm.shape[0], -1, -1)
            ],
                                        dim=1)
        return inputs_internlm

    def encode_text(self, text, add_special_tokens=False):
        text_token_ids = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=add_special_tokens,
        ).input_ids.to(self.device)
        text_embeds = self.internlm_model.model.embed_tokens(text_token_ids)
        return text_embeds

    def decode_text(self, out_embeds):
        out_text = self.tokenizer.batch_decode(out_embeds,
                                               skip_special_tokens=True)[0]
        out_text = out_text.split(self.eoa)[0]
        return out_text

    def wrap_text(self, user_text, bot_text='', add_special=True):
        if add_special:
            eoh = self.eoh
        else:
            eoh = ''
        text = f' <|User|>:{user_text} \n{eoh} <|Bot|>:{bot_text}'
        return text

    def get_gen_args(self, **kwargs):
        new_kargs = copy.deepcopy(self.gen_config)
        new_kargs.update(kwargs)
        return new_kargs

    def generate(self, text, image=None, **kwargs):
        text_embeds = self.encode_text(text)
        img_embeds = self.encode_img(image)
        prompt_embeds = self.wrap_prompt(text_embeds, img_embeds)
        out_embeds = self.internlm_model.generate(
            inputs_embeds=prompt_embeds, **self.get_gen_args(**kwargs))
        out_text = self.decode_text(out_embeds)
        return out_text

    def chat(self, text, image=None, history=None, **kwargs):
        text_embeds = self.encode_text(text)
        img_embeds = self.encode_img(image)
        prompt_embeds = self.wrap_prompt(text_embeds,
                                         img_embeds,
                                         history=history)
        out_embeds = self.internlm_model.generate(
            inputs_embeds=prompt_embeds, **self.get_gen_args(**kwargs))
        out_text = self.decode_text(out_embeds)

        # trunc at eoh and eoa
        clean_out_text_token_ids = self.tokenizer(
            out_text, return_tensors='pt').input_ids.to(self.device)
        clean_out_text_embeds = self.internlm_model.model.embed_tokens(
            clean_out_text_token_ids)
        clean_prompt_embeds = self.wrap_prompt(text_embeds,
                                               img_embeds,
                                               add_special=False)
        cur_history = torch.cat([clean_prompt_embeds, clean_out_text_embeds],
                                dim=1)
        if history is None:
            history = []
        history.append(cur_history)
        return out_text, history

    def wrap_prompt(self,
                    text_embeds,
                    img_embeds=None,
                    history=None,
                    add_special=True):
        if add_special:
            if history is None:
                prompt_segs = [
                    self.meta_instruction + ' <|User|>:',
                    f'\n{self.eoh} <|Bot|>:'
                ]
            else:
                prompt_segs = [' <|User|>:', f'\n{self.eoh} <|Bot|>:']
        else:
            prompt_segs = [' <|User|>:', ' <|Bot|>:']  # used in wrap history
        prompt_seg_embeds = []
        for i, seg in enumerate(prompt_segs):
            if history is not None:
                add_special_tokens = False
            else:
                add_special_tokens = i == 0
            seg_embeds = self.encode_text(
                seg, add_special_tokens=add_special_tokens)
            prompt_seg_embeds.append(seg_embeds)
        if img_embeds is None:
            img_embeds = text_embeds.new_empty(text_embeds.size(0), 0,
                                               text_embeds.size(-1))
        prompt_seg_embeds = [
            prompt_seg_embeds[0], img_embeds, text_embeds, prompt_seg_embeds[1]
        ]
        prompt_embeds = torch.cat(prompt_seg_embeds, dim=1)
        if history is not None:
            prompt_embeds = torch.cat([*history, prompt_embeds], dim=1)
        return prompt_embeds
