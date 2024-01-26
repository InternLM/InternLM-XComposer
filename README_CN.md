<p align="center">
    <img src="./assets/logo_cn.png" width="400"/>
</p>
<p align="center">
    <b><font size="6">浦语·灵笔2</font></b>
</p>

<!-- <div align="center">
        InternLM-XComposer <a href="">🤖 <a> <a href="">🤗</a>&nbsp ｜ InternLM-VL <a href="">🤖 <a> <a href="">🤗</a>&nbsp | Technical Report <a href=""> <a> 📄  -->

<div align="center">
        InternLM-XComposer2 <a href="https://huggingface.co/internlm/internlm-xcomposer2-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b"><img src="./assets/modelscope_logo.png" width="20px"></a> &nbsp｜ InternLM-XComposer2-VL <a href="https://huggingface.co/internlm/internlm-xcomposer2-vl-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b"><img src="./assets/modelscope_logo.png" width="20px"></a> &nbsp | 技术报告 <a href="">  📄 </a>

[English](./README.md) | [简体中文](./README_CN.md)


<p align="center">
    感谢社区提供的 InternLM-XComposer2 <a href="https://huggingface.co/spaces/Willow123/InternLM-XComposer">在线试用</a>
</p>

</div>
<p align="center">
    👋 加入我们的 <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> 和 <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">微信社区</a>
</p>

<br>

## 本仓库包括的多模态项目

> [**InternLM-XComposer2**](https://github.com/InternLM/InternLM-XComposer): **Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Models**

> [**InternLM-XComposer**](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-1.0): **A Vision-Language Large Model for Advanced Text-image Comprehension and Composition**

> <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo_tight.png" style="vertical-align: -20px;" :height="25px" width="25px">[**ShareGPT4V**](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V): **Improving Large Multi-modal Models with Better Captions**

</br>



**浦语·灵笔2**是基于[书生·浦语2](https://github.com/InternLM/InternLM/tree/main)大语言模型研发的突破性的图文多模态大模型，具有非凡的图文写作和图像理解能力，在多种应用场景表现出色：

- **自由指令输入的图文写作：** 浦语·灵笔2可以理解**自由形式的图文指令输入，包括大纲、文章细节要求、参考图片等**，为用户打造图文并貌的专属文章。生成的文章文采斐然，图文相得益彰，提供沉浸式的阅读体验。

- **准确的图文问题解答：** 浦语·灵笔2具有海量图文知识，可以准确的回复各种图文问答难题，在识别、感知、细节描述、视觉推理等能力上表现惊人。

- **杰出性能：** 浦语·灵笔2基于书生·浦语2-7B模型，我们在13项多模态评测中大幅领先同量级多模态模型，在其中6项评测中超过 GPT-4V 和 Gemini Pro。

<p align="center">
    <img src="assets/benchmark.png" width="1000"/>
</p>

我们开源的 浦语·灵笔2 包括两个版本:

- **InternLM-XComposer2-VL-7B** <a href="https://huggingface.co/internlm/internlm-xcomposer2-vl-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b"><img src="./assets/modelscope_logo.png" width="20px"> </a>（浦语·灵笔2-视觉问答-7B）: 基于书生·浦语2-7B大语言模型训练，面向多模态评测和视觉问答。浦语·灵笔2-视觉问答-7B是目前最强的基于7B量级语言模型基座的图文多模态大模型，领跑多达13个多模态大模型榜单。

- **InternLM-XComposer2-7B** <a href="https://huggingface.co/internlm/internlm-xcomposer2-vl-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer-vl2-7b"><img src="./assets/modelscope_logo.png" width="20px"> </a>: 进一步微调，支持自由指令输入图文写作的图文多模态大模型。
 
更多方法细节请参考[技术报告]()．
  <br>

<!-- 
<p align="center">
    <figcaption align = "center"><b> InternLM-XComposer </b></figcaption>
<p> -->


<!-- ## Demo



https://github.com/InternLM/InternLM-XComposer/assets/22662425/0a2b475b-3f74-4f41-a5df-796680fa56cd
 -->

## Demo Video

<video src='https://openxlab.oss-cn-shanghai.aliyuncs.com/xcomposer-writer/InternLM_XComposer_demo_CN.mp4' controls='' height=540 width=960>
</video>



## 更新消息
* ```2023.01.26``` 🎉🎉🎉 **InternLM-XComposer-VL-7B**的[评测代码](./evaluation/)已开源。
* ```2023.01.26``` 🎉🎉🎉 [InternLM-XComposer2-7B](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b) and [InternLM-XComposer-VL2-7B](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b)已在**ModelScope**开源。
* ```2023.01.26``` 🎉🎉🎉 [InternLM-XComposer2-7B](https://huggingface.co/internlm/internlm-xcomposer2-7b) and [InternLM-XComposer-VL2-7B](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b)已在**Hugging Face**开源。
* ```2023.01.26``` 🎉🎉🎉 我们公开了InternLM-XComposer2更多技术细节，请参考[技术报告]()。
* ```2023.11.22``` 🎉🎉🎉 我们开源了[ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V), 一个高质量的大规模图文描述数据集，以及性能优秀的多模态大模型ShareGPT4V-7B。
* ```2023.10.30``` 🎉🎉🎉 灵笔在[Q-Bench](https://github.com/Q-Future/Q-Bench/tree/master/leaderboards#overall-leaderboards) 和 [Tiny LVLM](https://github.com/OpenGVLab/Multi-Modality-Arena/tree/main/tiny_lvlm_evaluation) 取得了第一名。
* ```2023.10.19``` 🎉🎉🎉 支持多卡测试，多卡Demo. 两张4090显卡可部署全量Demo。
* ```2023.10.12``` 🎉🎉🎉 支持4比特量化Demo， 模型文件可从[Hugging Face](https://huggingface.co/internlm/internlm-xcomposer-7b-4bit) and [ModelScope](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer-7b-4bit) 获取。
* ```2023.10.8``` 🎉🎉🎉 [InternLM-XComposer-7B](https://huggingface.co/internlm/internlm-xcomposer-7b) 和 [InternLM-XComposer-VL-7B](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer-vl-7b) 已在Modelscope开源。
* ```2023.9.27``` 🎉🎉🎉 **InternLM-XComposer-VL-7B**的[评测代码](./evaluation/)已开源。
* ```2023.9.27``` 🎉🎉🎉 [InternLM-XComposer-7B](https://huggingface.co/internlm/internlm-xcomposer-7b) 和 [InternLM-XComposer-VL-7B](https://huggingface.co/internlm/internlm-xcomposer-vl-7b) 已在Hugging Face开源。
* ```2023.9.27``` 🎉🎉🎉 更多技术细节请参考[技术报告](https://arxiv.org/pdf/2309.15112.pdf)。
</br>


## 评测

我们在13个多模态评测对InternLM-XComposer2-VL上进行测试，包括：[MathVista](https://mathvista.github.io/), [MMMU](https://mmmu-benchmark.github.io/), [AI2D](https://prior.allenai.org/projects/diagram-understanding), [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation), [MMBench](https://opencompass.org.cn/leaderboard-multimodal), [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal), [SEED-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard), [QBench](https://github.com/Q-Future/Q-Bench/tree/master/leaderboards#overall-leaderboards), [HallusionBench](https://github.com/tianyi-lab/HallusionBench), [ChartQA](https://github.com/vis-nlp/ChartQA), [MM-Vet](https://github.com/yuweihao/MM-Vet), [LLaVA-in-the-wild](https://github.com/haotian-liu/LLaVA), [POPE](https://github.com/AoiDragon/POPE).


复现评测结果，请参考[评测细节](./evaluation/README.md)。
 


### 对比闭源多模态API以及开源SOTA模型。
|               | MathVista | AI2D   | MMMU  | MME    | MMB    | MMBCN  | SEEDI | LLaVAW | QBenchT | MM-Vet | HallB  | ChartVQA  |
|---------------|-----------|--------|-------|--------|--------|--------|-------|--------|---------|--------|--------|-----------|
|  Open-source Previous  SOTA | SPH-MOE   | Monkey | Yi-VL | WeMM   | L-Int2 | L-Int2 | SPH-2 | CogVLM | Int-XC  | CogVLM | Monkey | CogAgent  |
|    | 8x7B      | 10B    | 34B   | 6B     | 20B    | 20B    | 17B   | 17B    | 8B      | 30B    | 10B    | 18B       |
|  | 42.3      | 72.6   | 45.9  | 2066.6 | 75.1   | 73.7   | 74.8  | 73.9   | 64.4    | 56.8   | 58.4   | 68.4      |
|               |           |        |       |        |        |        |       |        |         |        |        |           |
| GPT-4V        | 49.9      | 78.2   | 56.8  | 1926.5 | 77     | 74.4   | 69.1  | 93.1   | 74.1    | 67.7   | 65.8   | 78.5      |
| Gemini-Pro    | 45.2      | 73.9   | 47.9  | 1933.3 | 73.6   | 74.3   | 70.7  | 79.9   | 70.6    | 64.3   | 63.9   | 74.1      |
| QwenVL-Plus   | 43.3      | 75.9   | 46.5  | 2183.3 | 67     | 70.7   | 72.7  | 73.7   | 68.9    | 55.7   | 56.4   | 78.1      |
| Ours          | 57.6      | 78.7   | 42    | 2242.7 | 79.6   | 77.6   | 75.9  | 81.8   | 72.5    | 51.2   | 60.3   | 72.6      |


### 对比开源模型。
| Method       | LLM          | MathVista | MMMU | MMEP     | MMEC  | MMB  | MMBCN | SEEDI | LLaVAW | QBenchT | MM-Vet | HallB  | POPE  |
|--------------|--------------|-----------|------|----------|-------|------|-------|-------|--------|---------|--------|--------|--------|
| BLIP-2       | FLAN-T5      | -         | 35.7 | 1,293.8  | 290.0 | -    | -     | 46.4  | 38.1   | -       | 22.4   | -      | -      |
| InstructBLIP | Vicuna-7B    | 25.3      | 30.6 | -        | -     | 36.0 | 23.7  | 53.4  | 60.9   | 55.9    | 26.2   | 53.6   | 78.9   |
| IDEFICS-80B  | LLaMA-65B    | 26.2      | 24.0 | -        | -     | 54.5 | 38.1  | 52.0  | 56.9   | -       | 39.7   | 46.1   | -      |
| Qwen-VL-Chat | Qwen-7B      | 33.8      | 35.9 | 1,487.5  | 360.7 | 60.6 | 56.7  | 58.2  | 67.7   | 61.7    | 47.3   | 56.4   | -      |
| LLaVA        | Vicuna-7B    | 23.7      | 32.3 | 807.0    | 247.9 | 34.1 | 14.1  | 25.5  | 63.0   | 54.7    | 26.7   | 44.1   | 80.2      |
| LLaVA-1.5    | Vicuna-13B   | 26.1      | 36.4 | 1,531.3  | 295.4 | 67.7 | 63.6  | 68.2  | 70.7   | 61.4    | 35.4   | 46.7   | 85.9      |
| ShareGPT4V   | Vicuna-7B    | 25.8      | 36.6 | 1,567.4  | 376.4 | 68.8 | 62.2  | 69.7  | 72.6   | -       | 37.6   | 49.8   | -      |
| CogVLM-17B   | Vicuna-7B    | 34.7      | 37.3 | -        | -     | 65.8 | 55.9  | 68.8  | 73.9   | -       | 54.5   | 55.1   | -      |
| LLaVA-XTuner | InernLM2-20B | 24.6      | 39.4 | -        | -     | 75.1 | 73.7  | 70.2  | 63.7   | -       | 37.2   | 47.7   | -      |
| Monkey-10B   | Qwen-7B      | 34.8      | 40.7 | 1,522.4 | 401.4 | 72.4 | 67.5  | 68.9  | 33.5   | -       | 33.0     | 58.4   | -      |
| InternLM-XC  | InernLM-7B   | 29.5      | 35.6 | 1,528.4  | 391.1 | 74.4 | 72.4  | 66.1  | 53.8   | 64.4    | 35.2   | 57.0   | -      |
| Ours         | InernLM2-7B  | 57.6      | 43.0 | 1,712.0  | 530.7 | 79.6 | 77.6  | 75.9  | 81.8   | 72.5    | 51.2   | 59.1   | 87.7    |

 

## 环境要求

* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users)
  <br>

## 安装教程

在运行代码之前，请先按照要求配置环境。请确认你的设备符合以上环境需求，然后安装环境。
请参考[安装教程](docs/install_CN.md)

## 快速开始

我们提供了一个简单实用的 🤗 Transformers 版本 InternLM-XComposer 的使用案例。

<details>
  <summary>
    <b>🤗 Transformers</b>
  </summary>


```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)  
 
text = '<ImageHere>仔细描述这张图'
image = 'examples/image1.webp'
with torch.cuda.amp.autocast(): 
  response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False) 
print(response)
#这张图片是一个引用的奥斯卡·王尔德的名言，它被放在一个美丽的日落背景上。
#引用的内容是“Live life with no excuses, travel with no regrets”，意思是“生活不要找借口，旅行不要后悔”。
# 在日落时分，两个身影站在山丘上，他们似乎正在享受这个美景。整个场景传达出一种积极向上、勇敢追求梦想的情感。
```
</details>


<details>
  <summary>
    <b>🤖 ModelScope</b>
  </summary>


```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b')
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model.tokenizer = tokenizer

text = '<ImageHere>仔细描述这张图'
image = 'examples/image1.webp'
with torch.cuda.amp.autocast(): 
  response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False) 
print(response)
#这张图片是一个引用的奥斯卡·王尔德的名言，它被放在一个美丽的日落背景上。
#引用的内容是“Live life with no excuses, travel with no regrets”，意思是“生活不要找借口，旅行不要后悔”。
# 在日落时分，两个身影站在山丘上，他们似乎正在享受这个美景。整个场景传达出一种积极向上、勇敢追求梦想的情感。
```
</details>


## Web UI

我们提供了一个轻松搭建 Web UI demo 的代码.
 
```
# 自由形式的图文创作demo
python examples/gradio_demo_composition.py

# 多模态对话demo
python examples/gradio_demo_chat.py
```
更多信息请参考 Web UI [用户指南](demo_asset/demo.md)。 如果您想要更改模型存放的文件夹，请使用 --folder=new_folder 选项。
 
<br>

## 引用

如果你觉得我们的代码和模型对你有帮助，请给我一个 star :star: 和 引用 :pencil: :)

```BibTeX
@misc{zhang2023internlmxcomposer,
      title={InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition}, 
      author={Pan Zhang and Xiaoyi Dong and Bin Wang and Yuhang Cao and Chao Xu and Linke Ouyang and Zhiyuan Zhao and Shuangrui Ding and Songyang Zhang and Haodong Duan and Hang Yan and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      year={2023},
      eprint={2309.15112},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<br>

## 许可证 & 联系我们

本仓库的代码依照 Apache-2.0 协议开源。模型权重对学术研究完全开放，也可申请免费的商业使用授权（[申请表](https://wj.qq.com/s2/12725412/f7c1/)）。其他问题与合作请联系 <internlm@pjlab.org.cn>。
