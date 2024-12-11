<p align="center">
    <img src="assets/logo_en.png" width="650"/>
</p>
<p align="center">
    <b><font size="6">InternLM-XComposer 2.5 OmniLive</font></b>
</p>


<div align="center">
        InternLM-XComposer2.5-OmniLive <a href="https://huggingface.co/internlm/internlm-xcomposer2d5-ol-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b"><img src="../assets/modelscope_logo.png" width="20px"></a> &nbsp｜ XComposer2.5 OmniLive Technical Report <a href="https://arxiv.org/abs/2407.03320">  📄 </a>  
 

[English](./README.md) | [简体中文](./README_CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/5245" target="_blank"><img src="https://trendshift.io/api/badge/repositories/5245" alt="InternLM%2FInternLM-XComposer | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<br>


## Demo Video
🔥 For the best experience, please keep the audio on while enjoying the video.

[https://github.com/user-attachments/assets/fd340f06-5586-452c-ae7b-59c983d4adcc](https://github.com/user-attachments/assets/fd340f06-5586-452c-ae7b-59c983d4adcc)


## Requirements

- python 3.8 and above
- pytorch 1.12 and above, 2.0 and above are recommended
- CUDA 11.4 and above are recommended (this is for GPU users)
- [flash-attention2](https://github.com/Dao-AILab/flash-attention) is required for high-resolution usage of InternLM-XComposer2.5.
  <br>

## Installation

Before running the code, make sure you have set up the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.
Please refer to the [installation instructions](../docs/install.md)

## Docker Image

We have also created a Docker image to simplify your setup process. You can find it here: [ixc-ol Docker Image](https://hub.docker.com/repository/docker/yhcao6/ixc2.5-ol/general). You can pull the image via
```shell
docker pull yhcao6/ixc2.5-ol:latest
```

## Quickstart

We provide simple examples below to show how to use InternLM-XComposer-2.5-OL with 🤗 Transformers. For complete guide, please refer to [here](examples/README.md).
 

<details>
  <summary>
    <b>Audio Understanding</b>
  </summary>

```python
import os
os.environ['USE_HF'] = 'True'

import torch
from swift.llm import (
    get_model_tokenizer, get_template, ModelType,
    get_default_template_type, inference
)
from swift.utils import seed_everything

model_type = ModelType.qwen2_audio_7b_instruct
model_id_or_path = 'internlm/internlm-xcomposer2d5-ol-7b'
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_id_or_path=model_id_or_path, model_dir='audio',
                                       model_kwargs={'device_map': 'cuda:0'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

# Chinese ASR
query = '<audio>Detect the language and recognize the speech.'
response, _ = inference(model, template, query, audios='examples/audios/chinese.mp3')
print(f'query: {query}')
print(f'response: {response}')
```

</details>


<details>
  <summary>
    <b>Image Understanding</b>
  </summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-ol-7b', model_dir='base', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval().half()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-ol-7b', model_dir='base', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Analyze the given image in a detail manner'
image = ['examples/images/dubai.png']
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, _ = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
print(response)
```

</details>


<details>
  <summary>
    <b>Video Understanding</b>
  </summary>

Please refer to [infer_llm_with_memory.py](examples/infer_llm_with_memory.py).
</details>


## Interactive Demo Deploy

Please refer to [Demo Setup Guide](online_demo/README.md) for guidelines.


## Evaluation

We evaluate InternLM-XComposer-2.5-OL on multimodal benchmarks, including audio, video and streaming benchmarks. For complete comparisons, please fer to our technique report.

### ASR benchmarks **WenetSpeech** and **LibriSpeech**.

| Method       | LLM           | Wenetspeech |               | Librispeech |            |             |             |
|--------------|---------------|-------------|---------------|-------------|------------|-------------|-------------|
|              |               | Test\_Net   | Test\_Meeting | Dev\_Clean  | Dev\_Other | Test\_Clean | Test\_Other |
| Qwen2\-Audio | Qwen2\-7B     | 7\.8        | 8\.4          | 1\.3        | 3\.4       | 1\.6        | 3\.6        |
| Mini\-Omni   | Qwen2\-0\.5B  | \-          | \-            | 4\.5        | 9\.7       | 4\.6        | 9\.2        |
| VITA         | Mixtral\-8x7B | 12\.2       | 16\.5         | 7\.6        | 16\.6      | 8\.1        | 18\.4       |
| IXC2\.5\-OL  | Qwen2\-1\.5B  | 9\.0        | 9\.2          | 2\.5        | 5\.7       | 2\.6        | 5\.8        |

### Video benchmark **MLVU**

| Method                 | Params | Topic Rea. | Anomaly Recog. | Needle QA | Ego Rea. | Plot QA | Action Or. | Action Co. | M-Avg |
|------------------------|--------|------------|----------------|-----------|----------|---------|------------|------------|-------|
| **Closed-source APIs** |
| Claude-3-Opus          | -      | 67.2       | 43.5           | 21.6      | 40.2     | 47.8    | 18.2       | 16.7       | 36.5  |
| Qwen-VL-Max            | -      | 67.4       | 63.5           | 40.3      | 40.9     | 43.3    | 25.0       | 14.8       | 42.2  |
| GPT-4 Turbo            | -      | 79.5       | 68.0           | 45.9      | 47.4     | 60.6    | 26.5       | 16.1       | 49.2  |
| GPT-4o                 | -      | 87.4       | 74.5           | 64.8      | 57.1     | 65.1    | 56.7       | 46.3       | 64.6  |
| **Open-source models** |
| MovieChat              | 7B     | 29.5       | 25.0           | 24.2      | 24.7     | 25.8    | 28.6       | 22.8       | 25.8  |
| LLaMA-VID              | 7B     | 50.8       | 34.5           | 30.1      | 32.7     | 32.5    | 23.9       | 27.8       | 33.2  |
| LLaVA-1.6              | 7B     | 60.6       | 41.0           | 43.1      | 38.4     | 41.0    | 25.5       | 25.7       | 39.3  |
| ShareGPT4Video         | 7B     | 75.8       | 51.5           | 47.6      | 43.2     | 48.4    | 34.0       | 23.3       | 46.4  |
| VideoLlaMA2            | 7B     | 74.6       | 64.5           | 49.9      | 43.8     | 45.1    | 34.0       | 27.4       | 48.5  |
| LongVA                 | 7B     | 83.3       | 58.5           | 69.3      | 50.0     | 67.2    | 38.6       | 27.2       | 56.3  |
| IXC2.5                 | 7B     | -          | -              | -         | -        | -       | -          | -          | 58.8  |
| InternVL2              | 8B     | -          | -              | -         | -        | -       | -          | -          | 64.0  |
| LLaVA-OneVision        | 7B     | -          | -              | -         | -        | -       | -          | -          | 64.7  |
| Video-XL               | 7B     | -          | -              | -         | -        | -       | -          | -          | 64.9  |
| IXC2.5-OL              | 7B     | 84.1       | 68.5           | 76.6      | 60.8     | 75.1    | 57.1       | 41.3       | 66.2  |

### Video benchmark **Video-MME**

| Method                 | Params | Short Video | Medium Video | Long Video | Overall |
|------------------------|--------|-------------|--------------|------------|---------|
| **Closed-source APIs** |
| GPT-4V                 | -      | 70.5        | 55.8         | 53.5       | 59.9    |
| Claude 3.5 Sonnet      | -      | 71.0        | 57.4         | 51.2       | 60.0    |
| GPT-4o mini            | -      | 72.5        | 63.1         | 58.6       | 64.8    |
| GPT-4o                 | -      | 80.0        | 70.3         | 65.3       | 71.9    |
| Gemini 1.5 Pro         | -      | 81.7        | 74.3         | 67.4       | 75.0    |
| **Open-source models** |
| ShareGPT4Video         | 7B     | 48.3        | 36.3         | 35.0       | 39.9    |
| VideoLlaMA2            | 7B     | -           | -            | -          | 47.9    |
| LongVA                 | 7B     | 61.1        | 50.4         | 46.2       | 52.6    |
| Video-XL               | 7B     | 64.0        | 53.2         | 49.2       | 55.5    |
| VITA                   | 8×7B   | 65.9        | 52.9         | 48.6       | 55.8    |
| IXC2.5                 | 7B     | -           | -            | -          | 55.8    |
| InternVL2              | 8B     | -           | -            | -          | 56.3    |
| LLaVA-OneVision        | 7B     | -           | -            | -          | 58.2    |
| mPLUG-Owl3             | 7B     | 70.0        | 57.7         | 50.1       | 59.3    |
| MiniCPM-V 2.6          | 8B     | -           | -            | -          | 60.9    |
| IXC2.5-OL              | 7B     | 72.7        | 58.2         | 50.8       | 60.6    |

### Streaming benchmark **StreamingBench**

| Method                 | Params | OP    | CR    | CS    | ATP   | EU    | TR    | PR    | SU    | ACP   | CT    | Overall |
|------------------------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| Human                  | -      | 89.47 | 92.00 | 93.60 | 91.47 | 95.65 | 92.52 | 88.00 | 88.75 | 89.74 | 91.30 | 91.46   |
| **Closed-source APIs** |
| Claude 3.5 Sonnet      | -      | 80.49 | 77.34 | 82.02 | 81.73 | 72.33 | 75.39 | 61.11 | 61.79 | 69.32 | 43.09 | 72.44   |
| GPT-4o                 | -      | 77.11 | 80.47 | 83.91 | 76.47 | 70.19 | 83.80 | 66.67 | 62.19 | 69.12 | 49.22 | 73.28   |
| Gemini 1.5 Pro         | -      | 79.02 | 80.47 | 83.54 | 79.67 | 80.00 | 84.74 | 77.78 | 64.23 | 71.95 | 48.70 | 75.69   |
| **Open-source models** |
| VideoLLM-online        | 8B     | 39.07 | 40.06 | 34.49 | 31.05 | 45.96 | 32.40 | 31.48 | 34.16 | 42.49 | 27.89 | 35.99   |
| VideoLLaMA2            | 7B     | 55.86 | 55.47 | 57.41 | 58.17 | 52.80 | 43.61 | 39.21 | 42.68 | 45.61 | 35.23 | 49.52   |
| VILA-1.5               | 8B     | 53.68 | 49.22 | 70.98 | 56.86 | 53.42 | 53.89 | 54.63 | 48.78 | 50.14 | 17.62 | 52.32   |
| LongVA                 | 7B     | 70.03 | 63.28 | 61.20 | 70.92 | 62.73 | 59.50 | 61.11 | 53.66 | 54.67 | 34.72 | 59.96   |
| InternVL2              | 8B     | 68.12 | 60.94 | 69.40 | 77.12 | 67.70 | 62.93 | 59.26 | 53.25 | 54.96 | 56.48 | 63.72   |
| Kangaroo               | 7B     | 71.12 | 84.38 | 70.66 | 73.20 | 67.08 | 61.68 | 56.48 | 55.69 | 62.04 | 38.86 | 64.60   |
| MiniCPM-V 2.6          | 8B     | 71.93 | 71.09 | 77.92 | 75.82 | 64.60 | 65.73 | 70.37 | 56.10 | 62.32 | 53.37 | 67.44   |
| Qwen2-VL               | 7B     | 75.20 | 82.81 | 73.19 | 77.45 | 68.32 | 71.03 | 72.22 | 61.19 | 69.04 | 46.11 | 69.04   |
| LLaVA-OneVision        | 7B     | 80.38 | 74.22 | 76.03 | 80.72 | 72.67 | 71.65 | 67.59 | 65.45 | 65.72 | 45.08 | 71.12   |
| IXC2.5-OL              | 7B     | 82.83 | 73.77 | 78.66 | 82.95 | 72.50 | 76.01 | 61.11 | 60.67 | 71.59 | 58.85 | 73.79   |


## Citation

If you find our models / code / papers useful in your research, please consider giving ⭐ and citations 📝, thx :)

```BibTeX
@article{internlmxcomposer2_5,
      title={InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output}, 
      author={Pan Zhang and Xiaoyi Dong and Yuhang Zang and Yuhang Cao and Rui Qian and Lin Chen and Qipeng Guo and Haodong Duan and Bin Wang and Linke Ouyang and Songyang Zhang and Wenwei Zhang and Yining Li and Yang Gao and Peng Sun and Xinyue Zhang and Wei Li and Jingwen Li and Wenhai Wang and Hang Yan and Conghui He and Xingcheng Zhang and Kai Chen and Jifeng Dai and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2407.03320},
      year={2024}
}
```

```BibTeX
@article{internlmxcomposer2_4khd,
      title={InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD},
      author={Xiaoyi Dong and Pan Zhang and Yuhang Zang and Yuhang Cao and Bin Wang and Linke Ouyang and Songyang Zhang and Haodong Duan and Wenwei Zhang and Yining Li and Hang Yan and Yang Gao and Zhe Chen and Xinyue Zhang and Wei Li and Jingwen Li and Wenhai Wang and Kai Chen and Conghui He and Xingcheng Zhang and Jifeng Dai and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2404.06512},
      year={2024}
}
```

```BibTeX
@article{internlmxcomposer2,
      title={InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model},
      author={Xiaoyi Dong and Pan Zhang and Yuhang Zang and Yuhang Cao and Bin Wang and Linke Ouyang and Xilin Wei and Songyang Zhang and Haodong Duan and Maosong Cao and Wenwei Zhang and Yining Li and Hang Yan and Yang Gao and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2401.16420},
      year={2024}
}
```

```BibTeX
@article{internlmxcomposer,
      title={InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition},
      author={Pan Zhang and Xiaoyi Dong and Bin Wang and Yuhang Cao and Chao Xu and Linke Ouyang and Zhiyuan Zhao and Shuangrui Ding and Songyang Zhang and Haodong Duan and Wenwei Zhang and Hang Yan and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2309.15112},
      year={2023}
}
```

<br>

## License & Contact Us

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [application form (English)](https://wj.qq.com/s2/12727483/5dba/)/[申请表（中文）](https://wj.qq.com/s2/12725412/f7c1/). For other questions or collaborations, please contact <internlm@pjlab.org.cn>.
