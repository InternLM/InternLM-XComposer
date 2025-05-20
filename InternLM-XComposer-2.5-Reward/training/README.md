# InternLM-XComposer-2.5-Reward (IXC-2.5-Reward) Training Code

We offer the official scripts for finetuning the [InternLM-XComposer-2.5-Reward](https://huggingface.co/internlm/internlm-xcomposer2d5-7b-reward) model on downstream tasks or training from scratch. Our training scripts use DeepSpeed and FSDP by default, and please refer to the [installation instructions](../../docs/install.md) for installation details.

Our fine-tuning scripts are based on the following environment:

```
torch==2.0.1
transformers==4.33.2
peft==0.8.2
deepspeed==0.12.3
```

### Data preparation

IXC-2.5-Reward is trained on preference data. We've open-sourced a portion of our mutli-modal instruction-following training data, [MMIF-23k](https://huggingface.co/datasets/ChrisDing1105/MMIF-23k), on Hugging Face. You can find technical details on how we built the multi-modal instruction-following preference data in [this report](https://arxiv.org/abs/2504.07957). Other preference data used in IXC-2.5-Reward is detailed in Table 1 and Table 2 of the [main paper](https://arxiv.org/abs/2501.12368).

We provide the `data/example.json` illustrating the training data formats. 

Your training data should follow the following format:

1. Saved as a list in json format, each conversation corresponds to an element of the list
2. Plain text conversation includes three keys: `id`, `conversations_a`, and `conversations_b`；The image-text conversation contains four keys: `id`, `conversations_a`, ``conversations_b`, and `image`.
3. `image` is the file path of the image or video
   - single image：string
   - multi images：\[string, string, ……\]
4. conversations_a means the chosen sample, and conversations_b refers to the rejected sample.

```
# An example of a single image and two rounds of conversations
# conversations_a: chosen
# conversations_b: rejected
temp = {
 'id': 0,
 'conversations_a': [
     {'from': 'human',   'value': 'Q'},
     {'from': 'bot',   'value': 'A'}，
  ],
  'conversations_b': [
     {'from': 'human',   'value': 'Q'},
     {'from': 'bot',   'value': 'A'}，
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
]
# multi image, please use 'Image x <ImageHere>; '
[
    {'from': 'human',   'value': 'Image1 <ImageHere>; Image2 <ImageHere>; Question'},
    {'from': 'bot',   'value': 'A'},
]
```

After pre-pareing the JSON files, you are required to define all the JSON file paths in a text file (e.g., `data.txt`) using the format:

```
<json path> <sample number (k)>
```

For example:

```
data/example.json 1
```

This means the model will sample 1,000 samples from `data/example.json` per training epoch. The sample counts will be automatically adjusted (either up-sampled or down-sampled) to meet the specified quantities.

If you want to sample 2,000 samples from `data/example.json`，You can manually modify the second line of `data.txt`：

```
data/example.json 2
```

After data preparation, you can use the provided bash scripts (`script_train.sh` or `script_train_lora.sh`) to finetune the model. Remember to specify the pre-train model path ($MODEL) and the txt data file path ($DATA) in the bash script.

### Full-parameter training/fine-tuning

Full-parameter parameter finetuning requires updating all parameters of LLM in the whole training process. To launch your training, run the following script:

```
sh script_train.sh
```

### LoRA training/fine-tuning

The LoRA allows light-weight model tuning with only a small subset of parameters updated. We provide the LoRA implementation based on `peft`. To launch your training, run the following script:

```
sh script_train_lora.sh
```

After training, you could load the model with the path to the adapter. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load.

```
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "internlm/internlm-xcomposer2d5-7b-reward", 
    device_map="cuda", 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-xcomposer2d5-7b-reward", trust_remote_code=True)
model.tokenizer = tokenizer

adapter_path = './output/ixc_reward_lora'
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
```

### Training script parameters explanation

The following is an explanation of some of the key hyper-parameters we defined in the training script (`script_train.sh` or `script_train_lora.sh`):

- `model_name_or_path`: model path, the default is `internlm/iinternlm-xcomposer2d5-7b-reward`
- `data_path`：defines the path for all fine-tuning json data, the default is `data.txt`
- `fix_vit`: whether to freeze the ViT encoder parameters. The default for full-parameter fine-tuning is `False`, and the default for LoRA is `True`
- `fix_sampler`：Whether to freeze the parameters of the projection layer after ViT. The default for full-parameter fine-tuning is `False`, and the default for LoRA is `True`
- `use_lora`：Whether to use LoRA fine-tuning. The default for full-parameter fine-tuning is `False`, and the default for LoRA is `True`
- `hd_num`: the number of sub-image patches in Dynamic Image Partition and Global-Local Format, the default is 18. If you encounter the GPU out of memory problem, you can reduce the value of this parameter
- `output_dir`: the path to save the fine-tuned weights, for example `output/finetune`
- `max_length`: The maximum number of tokens per conversation, the default is 16384, 80G A100 can support up to 24000 in the flash_attention2 environment. If you encounter the GPU out of memory problem, you can reduce the value of this parameter
