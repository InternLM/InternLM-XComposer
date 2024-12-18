<p align="center">
    <img src="assets/logo_cn.png" width="650"/>
</p>
<p align="center">
    <b><font size="6"> 浦语·灵笔 2.5 OmniLive</font></b>
</p>


<div align="center">
        浦语·灵笔 2.5 OmniLive <a href="https://huggingface.co/internlm/internlm-xcomposer2d5-ol-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b"><img src="../assets/modelscope_logo.png" width="20px"></a> &nbsp｜ 浦语·灵笔 2.5 OmniLive 技术报告 <a href="https://arxiv.org/abs/2407.03320">  📄 </a>  
 

[English](./README.md) | [简体中文](./README_CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/5245" target="_blank"><img src="https://trendshift.io/api/badge/repositories/5245" alt="InternLM%2FInternLM-XComposer | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<br>


## 演示视频
🔥 为了您更好的体验，请观看视频时打开音频选项

[https://github.com/user-attachments/assets/fd340f06-5586-452c-ae7b-59c983d4adcc](https://github.com/user-attachments/assets/fd340f06-5586-452c-ae7b-59c983d4adcc)


## 环境要求

- python 3.8 and above
- pytorch 1.12 and above, 2.0 and above are recommended
- CUDA 11.4 and above are recommended (this is for GPU users)
- [flash-attention2](https://github.com/Dao-AILab/flash-attention) is required for high-resolution usage of InternLM-XComposer2.5.
  <br>

## 安装教程

在运行代码之前，请先按照要求配置环境。请确认你的设备符合以上环境需求，然后安装环境。 请参考[安装教程](../docs/install.md)

## Docker 镜像

我们还创建了一个 Docker 镜像来简化您的安装过程。您可以在此处找到它: [ixc-ol Docker Image](https://hub.docker.com/repository/docker/yhcao6/ixc2.5-ol/general). 您可以通过以下命令来拉取该镜像：
```shell
docker pull yhcao6/ixc2.5-ol:latest
```

## 快速开始

我们提供了几个简单实用的 🤗 Transformers 版本 InternLM-XComposer-2.5-OL 系列的使用案例。 请参考 [这里](examples/README.md) 来查看完整的教程。
 

<details>
  <summary>
    <b>声音理解</b>
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
    <b>图片理解</b>
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
    <b>视频理解</b>
  </summary>

请参考 [infer_llm_with_memory.py](examples/infer_llm_with_memory.py).
</details>


## 交互 Demo 部署

请参考 [Demo Setup Guide](online_demo/README.md).


## 评测

我们在多模态基准测试上评估了 InternLM-XComposer-2.5-OL，包括音频、视频和流媒体基准测试。有关完整的比较，请参阅我们的技术报告。

### 人声识别基准测试 **WenetSpeech** and **LibriSpeech**.

| Method       | LLM           | Wenetspeech |               | Librispeech |            |             |             |
|--------------|---------------|-------------|---------------|-------------|------------|-------------|-------------|
|              |               | Test\_Net   | Test\_Meeting | Dev\_Clean  | Dev\_Other | Test\_Clean | Test\_Other |
| Qwen2\-Audio | Qwen2\-7B     | 7\.8        | 8\.4          | 1\.3        | 3\.4       | 1\.6        | 3\.6        |
| Mini\-Omni   | Qwen2\-0\.5B  | \-          | \-            | 4\.5        | 9\.7       | 4\.6        | 9\.2        |
| VITA         | Mixtral\-8x7B | 12\.2       | 16\.5         | 7\.6        | 16\.6      | 8\.1        | 18\.4       |
| IXC2\.5\-OL  | Qwen2\-1\.5B  | 9\.0        | 9\.2          | 2\.5        | 5\.7       | 2\.6        | 5\.8        |

### 视频基准测试 **MLVU**

<details>
  <summary>
    <b>测试代码</b>
  </summary>


```plaintext
下载MLVU的视频数据，并放在某个文件夹下，如'./video/mlvu'

└── video/                
   └── mlvu/           
      ├── 1_plotQA/ 
      │    ├──1.mp4
      │    ...
      ├── 2_needle/ 
      ├── 3_ego/ 
      ├── 4_count/ 
      ├── 5_order/
      ├── 6_anomaly_reco/  
      └── 7_topic_reasoning/  
```

```bash
sh benchmarks/mlvu/mlvu.sh ./video/mlvu
```

</details>

#### 结果

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

### 视频基准测试 **Video-MME**

<details>
  <summary>
    <b>测试代码</b>
  </summary>


```plaintext
下载VideoMME的视频数据，并放在某个文件夹下，如'./video/video_mme'

└── video/                
   └── video_mme/           
      ├── 026dzf-vc5g.mp4
      ├── 068rdc75mHM.mp4
      ├── 08km9Yqbt-A.mp4
      ├── 0ag_Qi5OEd0.mp4
          ...      
```

```bash
sh benchmarks/video_mme/video_mme.sh ./video/video_mme
```

</details>

#### 结果

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

### 流基准测试 **StreamingBench**

<details>
  <summary>
    <b>测试代码</b>
  </summary>


```plaintext
下载StreamingBench的视频数据，并放在某个文件夹下，如'./video/StreamingBench'

└── video/                
   └── StreamingBench/           
      └── real/ 
          ├──sample_1/
          │    └── video.mp4
          ├──sample_10/
          │    └── video.mp4
          ├──sample_12/
          ...    
```

```bash
sh benchmarks/streamingbench/eval.sh ./video/StreamingBench
```

</details>

#### 结果

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


### 视频基准测试 **MVBench**
<details>
  <summary>
    <b>测试代码</b>
  </summary>


```plaintext
下载MVBench的视频数据，并放在某个文件夹下，如'./video/mvbench'

└── video/                
   └── mvbench/           
      ├── clevrer/ 
      │   └── video_validation/
      │         ├── video_10009.mp4
      │         ├── video_10016.mp4
      │         ├── video_10017.mp4
      │         ...
      ├── FunQA_test/
      │   └── test/
      │         ├──test_creative/
      │         │  ├── C_KT_10_6402_6422.mp4
      │         │  ├── C_KT_12_1452_1602.mp4
      │         │  ├── C_KT_12_5112_5200.mp4
      │         │  ...
      │         ├──test_humor/
      │         │  ├── H_A_101_1433_1631.mp4
      │         │  ├── H_A_112_0436_0691.mp4
      │         │  ├── H_A_125_2078_2286.mp4
      │         │  ... 
      │         ...
      ...  
```

```bash
sh benchmarks/mvbench/mvbench.sh ./video/mvbench
```

</details>

#### 结果

| Method                 | Params | AS   | AP   | AA   | FA   | UA   | OE   | OI   | OS   | MD   | AL   | ST   | AC   | MC   | MA   | SC   | FP   | CO   | EN   | ER   | CI   | Avg  |
|------------------------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| **Closed-source APIs** |
| GPT-4V                 | -      | 55.5 | 63.5 | 72.0 | 46.5 | 73.5 | 18.5 | 59.0 | 29.5 | 12.0 | 40.5 | 83.5 | 39.0 | 12.0 | 22.5 | 45.0 | 47.5 | 52.0 | 31.0 | 59.0 | 11.0 | 43.5 |
| GPT-4o                 | -      | 61.5 | 56.5 | 72.0 | 54.0 | 82.0 | 62.5 | 66.5 | 44.0 | 36.5 | 33.5 | 93.0 | 54.5 | 33.5 | 54.5 | 53.5 | 74.5 | 71.5 | 32.5 | 71.0 | 42.5 | 57.5 |
| **Open-source models** |
| VideoLLaMA             | 7B     | 27.5 | 25.5 | 51.0 | 29.0 | 39.0 | 48.0 | 40.5 | 38.0 | 22.5 | 22.5 | 43.0 | 34.0 | 22.5 | 32.5 | 45.5 | 32.5 | 40.0 | 30.0 | 21.0 | 37.0 | 34.1 |
| VideoChat              | 7B     | 33.5 | 26.5 | 56.0 | 33.5| 40.5| 53.0| 40.5| 30.0| 25.5| 27.0| 48.5| 35.0| 20.5| 42.5| 46.0| 26.5| 41.0| 23.5| 23.5| 36.0|35.5 |
|MiniCPM-V 2.6| 7B | 38.0|43.0|63.0|35.5|67.5|55.5|46.0|35.5|25.5|33.0|77.5|48.0|37.0|54.0|42.5|40.0|31.0|38.0|43.0|40.5|44.7|
|VideoChat2 | 7B | 66.0| 47.5| 83.5| 49.5| 60.0 |58.0| 71.5| 42.5| 23.0| 23.0| 88.5| 39.0| 42.0| 58.5| 44.0| 49.0| 36.5| 35.0| 40.5| 65.5|51.1| 
|Qwen2-VL | 7B  | 51.0|58.0|77.5|47.0|64.0|63.0|65.5|40.0|25.5|35.5|77.0|43.5|47.0|62.0|42.0|61.5|49.5|41.5|47.5|41.5|52.0|
|PLLaVA | 34B  | 65.0|53.0|83.5|45.0|77.5|70.0|64.5|38.5|37.5|49.0|89.5|41.5|43.5|70.0|53.0|52.5|65.0|39.5|60.5|58.0|57.8|
|LLaVA-OneVision | 72B | 63.0|58.0|84.5|46.5|85.5|64.0|73.5|41.5|37.0|69.0|95.0|47.5|47.5|75.5|53.5|52.0|70.5|34.0|64.0|54.5|60.8|
|InternVL2 | 8B | 75.0 |62.0|83.5|40.5|69.5|96.0|72.0|29.5|58.0|53.0|88.5|39.5|83.0|97.0|51.0|78.5|65.0|33.0|48.0|67.0|64.5|
|IXC2.5-OL | 7B  | 84.5| 81.0| 75.0| 46.0| 81.0| 92.0| 79.5| 36.5| 83.0| 47.0| 90.0| 60.5| 75.0, | 93.0| 58.0| 60.5| 74.0| 42.0| 53.0| 62.0 | 68.7|


### 视频基准测试 **MMBench-Video**
<details>
  <summary>
    <b>测试代码</b>
  </summary>


我们使用VLMEvalKit来评测MMBench-Video，请参考[VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md).

```bash
# 将vlmeval/config.py里，XComposer2d5的model_path从internlm/internlm-xcomposer2d5-7b改为internlm-xcomposer2d5-ol-7b/base
torchrun --nproc-per-node=8 run.py --data MMBench-Video --model XComposer2d5 --nframe 64
```

</details>

#### 结果

| Method                 | Params | CP    | FP-S  | FP-C  | HL    | LR    | AR    | RR    | CSR   | TP    | Overall  | 
|------------------------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|----------|
| **Closed-source APIs** |
|Claude 3.5 Sonnet | - | 1.57 | 1.39 | 1.07 | 1.40 | 1.13 | 1.70 | 1.48 | 1.54 | 1.04  | 1.38|
|Gemini 1.0 Pro | - | 1.61 | 1.56 | 1.30 | 0.65  | 1.15 | 1.57 | 1.55 | 1.36 | 1.33  | 1.48|
|Gemini 1.5 Pro | - | 1.99 | 2.04 | 1.70 | 1.90  | 1.98 | 2.02 | 1.92 | 1.78 | 1.63  | 1.94|
|GPT-4V | - | 1.83 | 1.65 | 1.40 | 1.76 | 1.66  | 1.91 | 1.86 | 1.83 | 1.53  | 1.68|
|GPT-4o | - | 2.23 | 2.24 | 2.01 | 1.90 | 2.19  | 2.12 | 2.17 | 1.94 | 1.97  | 2.15|
| **Open-source APIs** |
|MovieLLM | 7B  |  0.95 |0.82  |0.70 | 0.15 |0.52 |1.12 | 1.22 |0.54 |1.05 | 0.87|
|LLaVA-OneVision | 72B  |  1.22 |1.07  |0.90 | 0.21| 0.76 |0.96 | 0.55 |0.81 |0.48 | 0.94|
|PLLaVA | 7B  |  1.08 |1.06  |0.86 | 0.52 |0.64 |1.25 | 1.17 |0.98  |1.01 | 1.03|
|ShareGPT4Video | 7B  |  1.20 |1.05 |1.00 | 0.32 |0.89 |1.06 | 1.19 |1.01|0.99 | 1.05|
|VideoStreaming | 7B  |  1.38 |1.13 |0.8 | 0.32 |0.77 |1.27 | 1.11 |1.01|1.10 | 1.12|
|LLaVA-NeXT-Video | 7B  |  1.35 |1.15 |0.97 | 0.58 |0.64 |1.38 | 1.30 |1.27|1.03 | 1.14|
|VILA1.5 | 13B  | 1.51 |1.45 |1.26 | 0.24 |0.80 | 1.52 | 1.30 |1.40 |1.28 | 1.36|
|InternVL2 | 8B  | 1.41 |1.37 |1.15 | 0.19 |0.90 | 1.34 | 1.38 |1.14 |1.00 | 1.26|
|Qwen2-VL | 7B  | 1.63 |1.51 |1.19 | 0.55 | 1.16 |1.56 | 1.49 | 1.37 |1.21 | 1.44|
|IXC2.5-OL | 7B  | 1.53 |1.61 |1.20 | 0.15 | 0.93 |1.44 | 1.57 | 1.30 |1.08 | 1.42|



## 引用

如果你觉得我们模型/代码/技术报告对你有帮助，请给我 ⭐ 和 引用 📝，谢谢 :)

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

## 许可证 & 联系我们

本仓库的代码依照 Apache-2.0 协议开源。模型权重对学术研究完全开放，也可申请免费的商业使用授权（[申请表](https://wj.qq.com/s2/12725412/f7c1/)）。其他问题与合作请联系 <internlm@pjlab.org.cn>。
