<p align="center">
    <img src="logo.png" width="400"/>
</p>
<p align="center">
    <b><font size="6">浦语·灵笔</font></b>
</p>

<!-- <div align="center">
        InternLM-XComposer <a href="">🤖 <a> <a href="">🤗</a>&nbsp ｜ InternLM-VL <a href="">🤖 <a> <a href="">🤗</a>&nbsp | Technical Report <a href=""> <a> 📄  -->

<div align="center">
        InternLM-XComposer <a href="https://huggingface.co/internlm/internlm-xcomposer-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer-7b">🤖 </a> &nbsp ｜ InternLM-XComposer-VL <a href="https://huggingface.co/internlm/internlm-xcomposer-vl-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer-vl-7b">🤖 </a> &nbsp | Technical Report <a href="https://arxiv.org/pdf/2309.15112.pdf">  📄 </a>

[English](./README.md) | [简体中文](./README_CN.md)

</div>

<br>



**浦语·灵笔**是基于[书生·浦语](https://github.com/InternLM/InternLM/tree/main)大语言模型研发的视觉-语言大模型，提供出色的图文理解和创作能力，具有多项优势：

- **图文交错创作**: 浦语·灵笔可以为用户打造图文并貌的专属文章。生成的文章文采斐然，图文相得益彰，提供沉浸式的阅读体验。这一能力由以下步骤实现：
    1. **理解用户指令，创作符合要求的长文章**。
    2. **智能分析文章，自动规划插图的理想位置，确定图像内容需求。**
    3. **多层次智能筛选，从图库中锁定最完美的图片。**

- **基于丰富多模态知识的图文理解**: 浦语·灵笔设计了高效的训练策略，为模型注入海量的多模态概念和知识数据，赋予其强大的图文理解和对话能力。
- **杰出性能**: 浦语·灵笔在多项视觉语言大模型的主流评测上均取得了最佳性能，包括[MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) (英文评测), [MMBench](https://opencompass.org.cn/leaderboard-multimodal) (英文评测), [Seed-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) (英文评测), [CCBench](https://opencompass.org.cn/leaderboard-multimodal)(中文评测), [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal) (中文评测).

我们开源的浦语·灵笔包括两个版本:

- **InternLM-XComposer-VL-7B** <a href="https://huggingface.co/internlm/internlm-xcomposer-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer-7b">🤖 </a>: 基于书生·浦语大语言模型的多模态预训练和多任务训练模型，在多种评测上表现出杰出性能, 例如：MME Benchmark, MMBench Seed-Bench, CCBench, MMBench-CN.
- **InternLM-XComposer-7B** <a href="https://huggingface.co/internlm/internlm-xcomposer-vl-7b">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer-vl-7b">🤖 </a>: 面向 *图文交错文章创作* 和 *智能对话* 的微调模型。
 
更多方法细节请参考[技术报告](https://arxiv.org/pdf/2309.15112.pdf)．
  <br>

<!-- 
<p align="center">
    <figcaption align = "center"><b> InternLM-XComposer </b></figcaption>
<p> -->

## Demo



https://github.com/InternLM/InternLM-XComposer/assets/22662425/0a2b475b-3f74-4f41-a5df-796680fa56cd






## 更新消息
* ```2023.10.8``` 🎉🎉🎉 [InternLM-XComposer-7B](https://huggingface.co/internlm/internlm-xcomposer-7b) 和 [InternLM-XComposer-VL-7B](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer-vl-7b) 已在Modelscope开源. 
* ```2023.9.27``` 🎉🎉🎉 **InternLM-XComposer-VL-7B**的[评测代码](./evaluation/)已开源.
* ```2023.9.27``` 🎉🎉🎉 [InternLM-XComposer-7B](https://huggingface.co/internlm/internlm-xcomposer-7b) 和 [InternLM-XComposer-VL-7B](https://huggingface.co/internlm/internlm-xcomposer-vl-7b) 已在Hugging Face开源. 
* ```2023.9.27``` 🎉🎉🎉 更多技术细节请参考[技术报告](https://arxiv.org/pdf/2309.15112.pdf).
<br>

## 评测

我们在5个多模态评测上测试 InternLM-XComposer-VL 的性能，包括英文评测 [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation), [MMBench](https://opencompass.org.cn/leaderboard-multimodal), [Seed-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) 和中文评测 [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal), [CCBench](https://opencompass.org.cn/leaderboard-multimodal).

   - [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation): 包括14个子任务的多模态模型全面评测。
   - [MMBench](https://opencompass.org.cn/leaderboard-multimodal): 提供精心收集的多模态评测题目和使用ChatGPT的循环评估策略的多模态评测。
   - [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal): 简体中文版本问题和答案的 [MMBench](https://opencompass.org.cn/leaderboard-multimodal) 评测。
   - [Seed-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard): 包括人工标注的1.9万道多模态多选题目的多模态评测。
   - [CCBench](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation): 针对中国文化理解的中文多模态评测。

InternLM-XComposer-VL 在**全部5个评测**上均超过其他多模态大模型，表现出强大的多模态理解能力。


### MME Benchmark

[MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) 是一个针对多模态大模型设计的多模态评测，关注模型的感知和认知能力，包括14个子任务。

InternLM-XComposer-VL 在感知和认知能力的综合性能上超过其他多模态大模型。点击查看[更多信息](evaluation/mme/MME_Bench.md)。


<p align="center">
综合性能
</p>


<div align="center">

| 排名 |      模型      |          版本         |  分数  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | [InternLM-XComposer-VL](https://github.com/InternLM/InternLM-XComposer) | [InternLM-7B](https://github.com/InternLM/InternLM-XComposer) | 1919.5 |
|   2  | Qwen-VL-Chat    |        Qwen-7B            | 1848.3 |
|   3  |      MMICL      |         FlanT5xxl        | 1810.7 |
|   4  |    Skywork-MM   |      Skywork-MM-13B      | 1775.5 |
|   5  |       BLIVA     |    FlanT5xxl             | 1669.2 |

</div>



<p align="center">
    <img src="evaluation/mme/perception.PNG" width="600"/>
</p>
<p align="center">
    <img src="evaluation/mme/cognition.PNG" width="600"/>
</p>


### MMBench & MMBench-CN

[MMBench](https://opencompass.org.cn/leaderboard-multimodal) 提供精心收集的多模态评测题目和使用ChatGPT的循环评估策略，包括了20个能力项。MMBench 还提供了中文版的 MMBench-CN 用于测试模型的中文能力。

InternLM-XComposer-VL 在 MMBench 和 MMBench-CN 测试集上都取得了最佳性能。点击查看[更多信息](evaluation/mmbench/MMBench.md).


<p align="center">
MMBench 测试集性能
</p>

<div align='center'>

| 排名 |      模型      |          版本         |  分数  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 74.4 |
|   2  |    Pink  |        Vicuna-7B            | 74.1 |
|   3  |      JiuTian      |        FLANT5-XXL        | 71.8 |
|   4  |  WeMM   |      InternLM-7B      | 69.0 |
|   5  |     mPLUG-Owl     |    LLaMA2 7B            |  68.5 |

</div>

<p align="center">
    <img src="evaluation/mmbench/mmbench.PNG" width="1000"/>
</p>

<p align="center">
MMBench-CN 测试集性能
</p>

<div align='center'>

| 排名 |      模型      |          版本         |  分数  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 72.4 |
|   2  |    QWen-VL-Chat | Qwen-7B | 56.3 |
|   3  |    LLaVA       | LLaMA 7B  |36.6 |
|   4  |    VosualGLM   | ChatGLM 6B | 25.6 |
|   5  |    mPLUG-Owl | LLaMA2 7B  | 24.9 |

</div>

<p align="center">
    <img src="evaluation/mmbench/mmbench_cn.PNG" width="1000"/>
</p>

### SEED-Bench

[SEED-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) 提供包括人工标注的1.9万道多模态多选题目的多模态评测, 覆盖12个评测为度。SEED-Bench同时提供 *图像* 和 *视频* 理解能力评测。点击查看[更多信息](evaluation/seed_bench/SEED.md).

InternLM-XComposer-VL 在图像理解评测取得最佳性能。


<p align="center">
SeedBench 图像理解评测
</p>

<div align="center">

| 排名 |      模型      |          版本         |  分数  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 66.9 |
|   2  |    QWen-VL-Chat | Qwen-7B | 65.4 |
|   3  |    QWen-VL | Qwen-7B | 62.3 |
|   4  |    InstructBLIP-Vicuna   |        Vicuna 7B  | 58.8 |
|   5  |    InstructBLIP   |     Flan-T5-XL  | 57.8 |

</div>

<p align="center">
    <img src="evaluation/seed_bench/seed_bench.PNG" width="1000"/>
</p>

### CCBench

[CCBench](https://opencompass.org.cn/leaderboard-multimodal) 针对中国文化理解设计的多模态评测. 点击查看[更多信息](evaluation/seed_bench/MMBench.md).

<p align="center">
CCBench 评测
</p>

<div align="center">

| 排名 |      模型      |          版本         |  分数  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 47.6 |
|   2  |    QWen-VL-Chat | Qwen-7B | 39.3 |
|   3  |    mPLUG-Owl | LLaMA2 7B  | 12.9 |
|   3  |    InstructBLIP       |        Vicuna 7B  | 12.1 |
|   4  |    VosualGLM   | ChatGLM 6B | 9.2  |

</div>

<p align="center">
    <img src="evaluation/mmbench/ccbench.PNG" width="1000"/>
</p>

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

#### 🤗 Transformers

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer-7b', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer-7b', trust_remote_code=True)
model.tokenizer = tokenizer

# example image
image = 'examples/images/aiyinsitan.jpg'

# Single-Turn Pure-Text Dialogue
text = '请介绍下爱因斯坦的生平'
response = model.generate(text)
print(response)
# '阿尔伯特·爱因斯坦（Albert Einstein，1879年3月14日－1955年4月18日），德国裔瑞士籍物理学家。他创立了现代物理学的两大支柱理论：
# 相对论和量子力学， 而质能等价公式E=mc2便是他的相对论思想的明证，因而被公认为是继伽利略、牛顿之后最伟大的物理学家。
# 1999年，爱因斯坦被美国《时代周刊》评选为20世纪的“世纪人物”，他在物理学上的贡献，使他在世界各地受到人们的尊敬。'

# Single-Turn Text-Image Dialogue
text = '请问这张图片里面的人是谁？并介绍下他。'
image = 'examples/images/aiyinsitan.jpg'
response = model.generate(text, image)
print(response)
# 图片中的男子是阿尔伯特·爱因斯坦（Albert Einstein），一位著名的物理学家和理论物理学家。他于1879年3月14日出生于德国巴登-符腾堡州的乌尔姆市，
# 1955 年4月18日逝世于美国新泽西州普林斯顿市。爱因斯坦在20世纪初提出了狭义相对论和广义相对论，对现代物理学的发展产生了深远影响。

# Multi-Turn Text-Image Dialogue
# 1st turn
text = '图片里面的是谁？'
response, history = model.chat(text=text, image=image, history=None)
print(response)
# 阿尔伯特·爱因斯坦。

# 2nd turn
text = '他有哪些成就?'
response, history = model.chat(text=text, image=None, history=history)
print(response)
# 阿尔伯特·爱因斯坦是20世纪最伟大的物理学家之一，他提出了狭义相对论和广义相对论，对现代物理学的发展产生了深远影响。
# 此外，他还提出了著名的质能方程E=mc²，为核能的开发提供了理论基础。

# 3rd turn
text = '他是最伟大的物理学家吗?'
response, history = model.chat(text=text, image=None, history=history)
print(response)
# 是的，阿尔伯特·爱因斯坦是20世纪最伟大的物理学家之一。
```

#### 🤖 ModelScope

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-xcomposer-7b')
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model.tokenizer = tokenizer

# example image
image = 'examples/images/aiyinsitan.jpg'

# Single-Turn Pure-Text Dialogue
text = '请介绍下爱因斯坦的生平'
response = model.generate(text)
print(response)
# '阿尔伯特·爱因斯坦（Albert Einstein，1879年3月14日－1955年4月18日），德国裔瑞士籍物理学家。他创立了现代物理学的两大支柱理论：
# 相对论和量子力学， 而质能等价公式E=mc2便是他的相对论思想的明证，因而被公认为是继伽利略、牛顿之后最伟大的物理学家。
# 1999年，爱因斯坦被美国《时代周刊》评选为20世纪的“世纪人物”，他在物理学上的贡献，使他在世界各地受到人们的尊敬。'
```

## Demo

### Web UI

我们提供了一个轻松搭建 Web UI demo 的代码.

<p align="center">
    <img src="demo_asset/assets/UI_en.png" width="800"/>
</p>


请运行以下代码

```
python examples/web_demo.py
```
更多信息请参考 Web UI [用户指南](demo_asset/demo.md).

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
