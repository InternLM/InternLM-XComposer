# 浦语·灵笔2.5微调

<div align="center">

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

我们官方提供了将[浦语·灵笔2.5](https://huggingface.co/internlm/internlm-xcomposer2d5-7b)应用到下游任务中的微调代码。我们的微调代码默认使用了 DeepSpeed 和 FSDP, 请参考[安装指南](../docs/install_CN.md)进行安装.

我们的微调代码基于以下环境：

```
torch==2.0.1
transformers==4.33.2
peft==0.8.2
deepspeed==0.12.3
```

> \[!WARNING\]
> 浦语·灵笔2.5的微调代码相比之前的 [浦语·灵笔1.0](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-1.0/finetune)、[浦语·灵笔2.0](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.0/finetune/) 数据格式有改动，微调2.5请以最新的版本为准。

### 微调数据准备

我们提供了三个示例来说明三种不同微调数据的格式：

- `data/only_text_example.json`: 纯文本，不带图片
- `data/single_turn_single_image_example.json`：单图单轮对话
- `data/multi_turn_multi_images_example.json` ：多图多轮对话

您的微调数据应该遵循以下的格式：

1. 保存为 json 格式的列表，每条对话对应一个列表的元素
2. 纯文本对话包括 id, conversation 两个 key；图文对话包含 id, conversation，image 三个 key
3. image 为图像或者视频的路径
   - 单图：string
   - 多图：\[string, string, ……\]
4. conversation 为 list 格式

```
# 一个单图、两轮对话的例子
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

5. 图片占位符 `<ImageHere>`
   - 单图：**无需占位符**，仅包含指令文本
   - 多图：在指令中用 `'Image x <ImageHere>; '` 来指定图像顺序和位置,  x 从 1 开始计数

```
# 单图无需占位符
[
    {'from': 'human',   'value': 'Q'},
    {'from': 'bot',   'value': 'A'},
    {'from': 'human',   'value': 'Q'},
    {'from': 'bot',   'value': 'A'},
]
# 多图用 'Image x <ImageHere>; '
[
    {'from': 'human',   'value': 'Image1 <ImageHere>; Image2 <ImageHere>; Question'},
    {'from': 'bot',   'value': 'A'},
    {'from': 'human',   'value': 'Question. Image3 <ImageHere>; Image4 <ImageHere>; '},
    {'from': 'bot',   'value': 'A'},
]
```

准备好 JSON 文件后，您需要使用以下格式在一个文本文件（例如 `data.txt`）中定义所有 JSON 文件的路径：

```
<json path> <sample number (k)>
```

例如：

```
data/only_text_example.json 0.02
data/single_turn_single_image_example.json 0.01
data/multi_turn_multi_images_example.json 0.01
```

这意味着模型将在每个微调周期（epoch）从 `data/only_text_example.json` 中采样 20个样本， `data/single_turn_single_image_example.json` 中采样 10 个样本，从 `data/multi_turn_multi_images_example.json` 中采样 10 个样本。 样本计数将自动调整（上采样或下采样）以满足指定的数量。

如果你想从 `data/single_turn_single_image_example.json` 中采样 2000 个样本，你可以手动修改 `data.txt` 的第二行：

```
data/single_turn_single_image_example.json 2
```

数据准备完毕后，您可以使用提供的 bash 脚本（`finetune.sh` 或 `finetune_lora.sh`）来微调模型。请记住在 bash 脚本中指定预训练模型路径（$MODEL）和数据路径（$DATA）。

### 全参数微调

全参数微调需要更新 LLM 的所有参数。要启动全参数微调，请运行以下脚本：

```
sh finetune/finetune.sh
```

### LoRA 微调

LoRA 是一种轻量级、允许仅更新一小部分参数的微调方法。 我们提供基于 `peft` 的 LoRA 微调。要启动 LoRA 微调，请运行以下脚本：

```
sh finetune/finetune_lora.sh
```

训练后，您可以使用保存 adapter 的路径加载模型。我们建议您在 configuration json file 中使用绝对路径定义预训练模型。

```
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    # 保存 adapter 的路径
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()
```

训练后，您也可以用 `merge_peft_adapter.py` 合并 LoRA 权重与原模型权重：

```
python3 merge_peft_adapter.py \
    --adapter_model_name=path_to_adapter \
    --base_model_name=path_to_base_model \
    --output_name=path_to_output_name \
```

### 训练脚本参数解释

以下是我们在训练脚本 （`finetune.sh` 或 `finetune_lora.sh`）定义的一些关键参数的解释：

- `model_name_or_path`: 模型的路径，默认是 `internlm/internlm-xcomposer2d5-7b`
- `data_path`：定义所有微调 json 数据的路径，默认是 `data.txt`
- `fix_vit`: 是否要冻结 ViT encoder 的参数。全参数放开默认是 `False`，LoRA 默认是 `True`
- `fix_sampler`：是否要冻结 ViT 之后连接层的参数。全参数放开默认是 `False`，LoRA 默认是 `True`
- `use_lora`：是否要使用 LoRA 微调。全参数放开默认是 `False`，LoRA 默认是 `True`
- `hd_num`: 对于 Dynamic Image Partition 和 Global-Local Format 中图像切块的数量，默认是 18。如果你遇到爆显存的问题，可以调小
- `output_dir`: 保存微调后权重的路径，例如 `output/finetune`
- `max_length`: 每条对话最长的 tokens 数量，默认是 16384，80G A100 在 flash_attention2 的环境下可以支持到 24000。如果你遇到爆显存的问题，可以调小
