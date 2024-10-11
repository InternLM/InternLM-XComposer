# 浦语·灵笔2微调

<div align="center">

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

我们官方提供了将浦语·灵笔2应用到下游任务中的微调代码。我们的微调代码默认使用了 DeepSpeed 和 FSDP, 请参考[安装指南](../docs/install_CN.md)进行安装.

请确保您已从 [huggingface](https://huggingface.co/openai/clip-vit-large-patch14-336) 下载 `openai/clip-vit-large-patch14-336` 模型。

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
          "value": "<ImageHere> <ImageHere>这两张图中有什么"
        },
        {
          "from": "assistant",
          "value": "第一张图中包含了......"
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

如果你想微调 `internlm/internlm-xcomposer2-7b` 模型, 请设置 `--img_size 224` 和 `--hd_num -1`.

如果你想微调 `internlm/internlm-xcomposer2-vl-7b` 模型, 请设置 `--img_size 490` 和 `--hd_num -1`.

如果你想微调 `internlm/internlm-xcomposer2-4khd-7b` 模型, 请设置 `hd_num`为正整数，例如 `--hd_num 16`. 参数 `img_size` 在4khd 模型中未使用，可以设为任意数字.

### LoRA 微调

LoRA 是一种轻量级、允许仅更新一小部分参数的微调方法。 我们提供基于 `peft` 的 LoRA 微调。要启动 LoRA 微调，请运行以下脚本：

```
sh finetune/finetune_lora.sh
```

参数 `img_size` 的取值和全参数微调一致 (7b 设成 224， vl-7b 设成 490).

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

### 微调常见问题

> Q: batch_size 要怎么设置？

A: 目前的微调代码只支持 batch_size = 1. 如果你想要支持 batch size > 1，要自行在 [这个函数](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/blob/main/modeling_internlm_xcomposer2.py#L208) 加入 padding。

> Q: 为什么微调时我的 loss 是 0？

A: 这是由于数据格式不对。对于 `-vl-7b` 模型，可以在[这里](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/blob/main/modeling_internlm_xcomposer2.py#L214)设断点查看 `text` 变量的值。对于 `-7b` 和 `-4khd-7b` 也是在该函数对应的位置查看。

> Q: 微调代码支持多张图片的输入吗？

A: 支持。微调时多张图片请按照下面的格式准备：

```
{
    "id": "0",
    "image": ['path/to/image_0.jpg', 'path/to/image_1.jpg']
    "conversations": [
      {
        "from": "user",
        "value": "<ImageHere> <ImageHere>这两张图中有什么"
      },
      {
        "from": "assistant",
        "value": "第一张图中包含了......"
      }
    ]
},
```

测试时，请参考下面的代码进行多张图片的输入:

```
model = AutoModelForCausalLM.from_pretrained('your model path').cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('your model path')

images = ["./a.png", "./b.png"]
image1 = model.encode_img(images[0])
image2 = model.encode_img(images[1])
image = torch.cat((image1, image2), dim=0)

query = ""First picture:<ImageHere>, second picture:<ImageHere>. Describe the subject of these two pictures?"""

response, _ = model.interleav_wrap_chat(tokenizer, query, image, history=[])
print(response)
```
