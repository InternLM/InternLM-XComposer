# InternLM-XComposer2 Finetuning

<div align="center">

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

We offer the official scripts for easy finetuning of the pretrained internlm-xcomposer2 model on downstream tasks. Our finetune scripts use DeepSpeed and FSDP by default, and please refer to the [installation instructions](../docs/install.md) for installation details.

Please make sure you have downloaded the `openai/clip-vit-large-patch14-336` model from [huggingface](https://huggingface.co/openai/clip-vit-large-patch14-336).

### Data preparation

To prepare your finetuning data, you should (1) formulate each sample as a dictionary consisting of an id, an image path list with multiple images (optional, not required for pure-text example), and a list of conversations, and (2) save data samples in JSON files.

For the vision-language example with image(s), you are required to define placeholder(s) <ImageHere> to define the position to insert the image embeddings.

<details>
  <summary>
    <b>vision-language example (vl_data.json) with 2 samples.</b>
  </summary>

```
  [
    {
      "id": "0",
      "image": ['path/to/image_0.jpg', 'path/to/image_1.jpg']
      "conversations": [
        {
          "from": "user",
          "value": "<ImageHere> <ImageHere>Please describe these two images in detail."
        },
        {
          "from": "assistant",
          "value": "The first image......"
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
    <b>pure-text example list (text_data.json) with 2 samples.</b>
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

After pre-pareing the JSON files, you are required to define all the JSON file paths in a text file (e.g., `data.txt`) using the format:

```
<json path> <sample number (k)>
```

For example:

```
path/to/vl_data.json 10
path/to/text_data.json 5
```

This means the model will sample 10k samples from `vl_data.json` and 5k samples from `text_data.json` per finetuning epoch. The sample counts will be automatically adjusted (either up-sampled or down-sampled) to meet the specified quantities.

After data preparation, you can use the provided bash scripts (`finetune.sh` or `finetune_lora.sh`) to finetune the model. Remember to specify the pre-train model path ($MODEL) and the txt data file path ($DATA) in the bash script.

### Full-parameter finetuning

Full-parameter parameter finetuning requires updating all parameters of LLM in the whole training process. To launch your training, run the following script:

```
sh finetune.sh
```

If you want to finetune the `internlm/internlm-xcomposer2-7b` model, please set the `--img_size 224` and `--hd_num -1`.

If you want to finetune the `internlm/internlm-xcomposer2-vl-7b` model, please set the `--img_size 490` and `--hd_num -1`.

If you want to finetune the `internlm/internlm-xcomposer2-4khd-7b` model, please set `hd_num` to a positive integer, e.g., `--hd_num 16`. The parameter `img_size` is not used in the 4khd model and can by any number.

### LoRA finetuning

The LoRA allows light-weight model tuning with only a small subset of parameters updated. We provide the LoRA implementation based on `peft`. To launch your training, run the following script:

```
sh finetune_lora.sh
```

The value of the `img_size` parameter is consistent with full parameter fine-tuning (224 for the 7b model and 490 for the vl-7b model).

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

### Finetuning FAQs

> Q: How to set the `batch_size` parameter?

A: The current fine-tuning code only supports batch_size = 1. If you want to support batch size > 1, you have to add the padding yourself in \[this function\](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/blob/main/ modeling_internlm_xcomposer2.py#L208).

> Q: Why my loss is 0 during the fine-tuning?

A: This is due to the incorrect SFT data format. For the `-vl-7b` model, you can set a breakpoint to view the value of the `text` variable in [here](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/blob/main/modeling_internlm_xcomposer2.py#L214). For `-7b` and `-4khd-7b`, also check the corresponding position of this function.

> Q: Does the fine-tuning code support multi-image inputs?

A: Yes. The finetuning SFT data format for multi-image inputs is:

```
{
    "id": "0",
    "image": ['path/to/image_0.jpg', 'path/to/image_1.jpg']
    "conversations": [
      {
        "from": "user",
        "value": "<ImageHere> <ImageHere>Please describe these two images in detail."
      },
      {
        "from": "assistant",
        "value": "The first image......"
      }
    ]
},
```

When testing, please refer to the following code using multiple image inputs:

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
