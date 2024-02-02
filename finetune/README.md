# InternLM-XComposer2 Finetuning

<div align="center">

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

We offer the official scripts for easy finetuning of the pretrained internlm-xcomposer2 model on downstream tasks. Our finetune scripts use DeepSpeed and FSDP by default, and please refer to the [installation instructions](../docs/install.md) for installation details.

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

If you want to finetune the `internlm/internlm-xcomposer-7b` model, please set the `--img_size 224`.

If you want to finetune the `internlm/internlm-xcomposer-vl-7b` model, please set the `--img_size 490`.

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
