# InternLM-XComposer2.5 Finetuning

<div align="center">

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

We offer the official scripts for easy finetuning of the pretrained [InternLM-XComposer2.5](https://huggingface.co/internlm/internlm-xcomposer2d5-7b) model on downstream tasks. Our finetune scripts use DeepSpeed and FSDP by default, and please refer to the [installation instructions](../docs/install.md) for installation details.

Our fine-tuning scripts are based on the following environment:

```
torch==2.0.1
transformers==4.33.2
peft==0.8.2
deepspeed==0.12.3
```

> \[!WARNING\]
> The data format of InternLM-XComposer2.5 has been changed compared with the previous [InternLM-XComposer1.0](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-1.0/finetune)、[InternLM-XComposer2.0](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.0/finetune/). Please refer to the latest version for fine-tuning 2.5.

### Data preparation

We provide three examples illustrating three different formats of fine-tuning data:

- `data/only_text_example.json`: text only, on images
- `data/single_turn_single_image_example.json`：single turn, single image conversation
- `data/multi_turn_multi_images_example.json` : multi turn, multi images conversation

Your fine-tuning data should follow the following format:

1. Saved as a list in json format, each conversation corresponds to an element of the list
2. Plain text conversation includes two keys: `id` and `conversation`；The image-text conversation contains three keys: `id`, `conversation` and `image`.
3. `image` is the file path of the image or video
   - single image：string
   - multi images：\[string, string, ……\]
4. conversation is in list format

```
# An example of a single image and two rounds of conversations
temp = {
 'id': 0,
 'conversations': [
     {'from': 'human',   'value': 'Q'},
     {'from': 'bot',   'value': 'A'}，
     {'from': 'human',   'value': 'Q'},
     {'from': 'bot',   'value': 'A'}
  ],
 'image': 'path'
}
```

5. image placeholder `<ImageHere>`
   - single image：**no placeholder required**
   - multi images：use `'Image x <ImageHere>; '` in the instruction to specify the image order and position, x starts counting from 1

```
# single image, no placeholder required
[
    {'from': 'human',   'value': 'Q'},
    {'from': 'bot',   'value': 'A'},
    {'from': 'human',   'value': 'Q'},
    {'from': 'bot',   'value': 'A'},
]
# multi image, please use 'Image x <ImageHere>; '
[
    {'from': 'human',   'value': 'Image1 <ImageHere>; Image2 <ImageHere>; Question'},
    {'from': 'bot',   'value': 'A'},
    {'from': 'human',   'value': 'Question. Image3 <ImageHere>; Image4 <ImageHere>; '},
    {'from': 'bot',   'value': 'A'},
]
```

After pre-pareing the JSON files, you are required to define all the JSON file paths in a text file (e.g., `data.txt`) using the format:

```
<json path> <sample number (k)>
```

For example:

```
data/only_text_example.json 0.02
data/single_turn_single_image_example.json 0.01
data/multi_turn_multi_images_example.json 0.01
```

This means the model will sample 20 samples from `data/only_text_example.json`, 10 samples from `data/single_turn_single_image_example.json` and 10 samples from `data/multi_turn_multi_images_example.json` per fine-tuning epoch. The sample counts will be automatically adjusted (either up-sampled or down-sampled) to meet the specified quantities.

If you want to sample 2,000 samples from `data/single_turn_single_image_example.json`，You can manually modify the second line of `data.txt`：

```
data/single_turn_single_image_example.json 2
```

After data preparation, you can use the provided bash scripts (`finetune.sh` or `finetune_lora.sh`) to finetune the model. Remember to specify the pre-train model path ($MODEL) and the txt data file path ($DATA) in the bash script.

### Full-parameter fine-tuning

Full-parameter parameter finetuning requires updating all parameters of LLM in the whole training process. To launch your training, run the following script:

```
sh finetune.sh
```

### LoRA fine-tuning

The LoRA allows light-weight model tuning with only a small subset of parameters updated. We provide the LoRA implementation based on `peft`. To launch your training, run the following script:

```
sh finetune_lora.sh
```

After training, you could load the model with the path to the adapter. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load.

```
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    # path to the output directory
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()
```

After training, you can also use `merge_peft_adapter.py` to merge the LoRA weights with the original model weights:

```
python3 merge_peft_adapter.py \
    --adapter_model_name=path_to_adapter \
    --base_model_name=path_to_base_model \
    --output_name=path_to_output_name \
```

### Training script parameters explanation

The following is an explanation of some of the key hyper-parameters we defined in the training script (`finetune.sh` or `finetune_lora.sh`):

- `model_name_or_path`: model path, the default is `internlm/internlm-xcomposer2d5-7b`
- `data_path`：defines the path for all fine-tuning json data, the default is `data.txt`
- `fix_vit`: whether to freeze the ViT encoder parameters. The default for full-parameter fine-tuning is `False`, and the default for LoRA is `True`
- `fix_sampler`：Whether to freeze the parameters of the projection layer after ViT. The default for full-parameter fine-tuning is `False`, and the default for LoRA is `True`
- `use_lora`：Whether to use LoRA fine-tuning. The default for full-parameter fine-tuning is `False`, and the default for LoRA is `True`
- `hd_num`: the number of sub-image patches in Dynamic Image Partition and Global-Local Format, the default is 18. If you encounter the GPU out of memory problem, you can reduce the value of this parameter
- `output_dir`: the path to save the fine-tuned weights, for example `output/finetune`
- `max_length`: The maximum number of tokens per conversation, the default is 16384, 80G A100 can support up to 24000 in the flash_attention2 environment. If you encounter the GPU out of memory problem, you can reduce the value of this parameter
