# 浦语·灵笔2微调

<div align="center">

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

我们官方提供了将浦语·灵笔2应用到下游任务中的微调代码。我们的微调代码默认使用了 DeepSpeed 和 FSDP, 请参考[安装指南](../docs/install_CN.md)进行安装.

### Data preparation

为了准备微调数据，您应该（1）将每个样本制定为一个字典，其中包含一个 id、一个包含多个图像的图像路径列表（可选，对于纯语言样本不需要）和一个对话列表，（2）并将数据样本保存在 JSON 文件中。

对于带有图像的样本，您需要定义占位符 <ImageHere> 来定义插入图像特征的位置。

<details>
  <summary>
    <b>图文数据示例 (vl_data.json)</b>
  </summary>

```
  [
    {
      "id": "0",
      "image": ['path/to/image_0.jpg', 'path/to/image_1.jpg']
      "conversations": [
        {
          "from": "user",
          "value": "<ImageHere> <ImageHere>图中是什么"
        },
        {
          "from": "assistant",
          "value": "这张图中包含了......"
        }
      ]
    },
    {
      "id": "1",
      "image": ['path/to/image_1.jpg']
      "conversations": [
        {
          "from": "user",
          "value": "<ImageHere> what is the color of the dog"
        },
        {
          "from": "assistant",
          "value": "it is ...."
        }
      ]
    }
  ]
```

</details>

<details>
  <summary>
    <b>纯文本数据示例 (text_data.json)</b>
  </summary>

```
  [
    {
      "id": "0",
      "conversations": [
        {
          "from": "user",
          "value": "你好"
        },
        {
          "from": "assistant",
          "value": "你好，我是浦语·灵笔，一个支持图文创作的多模态大模型。"
        }
      ]
    },
    {
      "id": "1",
      "conversations": [
        {
          "from": "user",
          "value": "Tell me something about Albert Einstein."
        },
        {
          "from": "assistant",
          "value": "Albert Einstein was a German-born theoretical physicist who developed .... "
        }
      ]
    }
  ]
```

</details>

准备好 JSON 文件后，您需要使用以下格式在一个文本文件（例如 `data.txt`）中定义所有 JSON 文件的路径：

```
<json path> <sample number (k)>
```

例如：

```
path/to/vl_data.json 10
path/to/text_data.json 5
```

这意味着模型将在每个微调周期从“vl_data.json”中采样 10k 个样本，从“text_data.json”中采样 5k 个样本。 样本计数将自动调整（上采样或下采样）以满足指定的数量。

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
