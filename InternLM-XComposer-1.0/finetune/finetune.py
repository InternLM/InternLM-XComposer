# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
from dataclasses import dataclass, field
import json
import math
import logging
import os
import random
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from torchvision import transforms
from typing import Dict, Optional, Sequence, List
from data_mix import Mix_dataset


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/petrelfs/share_data/dongxiaoyi/share_models/debug_ft")

@dataclass
class DataArguments:
    vl_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    txt_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    img_size: int = 224
    inner_batch_size: int = 4


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = False
    label_names: List[str] = field(
        default_factory=lambda: ['samples']
    )

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['mlp.up_proj', 'mlp.down_proj', 'mlp.gate_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance['samples'] for instance in instances]
        text_input, data_type = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "data_type"))
        
        batch = dict(
            text_input = text_input,
            data_type = data_type,
        )
        batch['sp_token'] = instances[0]['sp_token']
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image'] = images

            data_type = [instance['data_type'] for instance in instances]
            batch['data_type'] = data_type


        return dict(
            samples=batch
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print("Loading data...")

    train_dataset = Mix_dataset(data_args.vl_data_path, data_args.txt_data_path, data_args.inner_batch_size, img_size=data_args.img_size, local_rank=local_rank)
    print (str(len(train_dataset)) + 'samples is loaded')
    eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
               )


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length

    # Load model and tokenizer
    print(f'Load model from: {model_args.model_name_or_path}')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model.tokenizer = tokenizer
    if not training_args.use_lora:
        #model.requires_grad_(True)
        if training_args.fix_vit and hasattr(model,'visual_encoder'):
            print ('Freeze ViT')
            model.visual_encoder.requires_grad_(False)
        if training_args.fix_sampler and hasattr(model,'Qformer'):
            print ('Freeze Qformer')
            model.Qformer.requires_grad_(False)
            model.internlm_proj.requires_grad_(False)
            model.query_tokens.requires_grad = False

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args
    )

    # Start trainner
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
