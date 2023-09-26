<p align="center">
    <img src="logo.png" width="400"/>
<p>
<p align="center">
    <b><font size="6">InternLM-XComposer</font></b>
<p>

<!-- <div align="center">
        InternLM-XComposer <a href="">🤖 <a> <a href="">🤗</a>&nbsp ｜ InternLM-VL <a href="">🤖 <a> <a href="">🤗</a>&nbsp | Technical Report <a href=""> <a> 📄  -->

<div align="center">
        InternLM-XComposer <a href="">🤗</a>&nbsp ｜ InternLM-XComposer-VL <a href="">🤗</a>&nbsp | Technical Report <a href="">  📄 <a>

<!-- [English](./README.md) | [简体中文](./README_zh.md) -->

</div>

<br>




**InternLM-XComposer** is a vision-language large model (VLLM) based on [InternLM](https://github.com/InternLM/InternLM/tree/main) for advanced text-image comprehension and composition. InternLM-XComposer has serveal appealing properties:

- **Interleaved Text-Image Composition**: InternLM-XComposer can effortlessly generate coherent and contextual articles that seamlessly integrate images, providing a more engaging and immersive reading experience. The interleaved text-image composition is implemented in following steps:

    1. **Text Generation**: It crafts long-form text based on human-provided instructions.
    2. **Image Spoting and Captioning**: It pinpoints optimal locations for image placement and furnishes image descriptions.
    3. **Image Retrieval and Selection**: It select image candidates and identify the image that optimally complements the content.

- **Comprehension with Rich Multilingual Knowledge**: The text-image comprehension is empowered by training on extensive multi-modal multilingual concepts with carefully crafted strategies, resulting in a deep understanding of visual content.
- **Strong performance**: It consistently achieves state-of-the-art results across various benchmarks for vision-language large models, including [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) (English), [MMBench](https://opencompass.org.cn/leaderboard-multimodal) (English), [Seed-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) (English), [CCBench](https://opencompass.org.cn/leaderboard-multimodal)(Chinese), and [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal) (Chineese).

We release InternLM-XComposer series in two versions:

- [InternLM-XComposer-VL-7B](https://huggingface.co/internlm/internlm-xcomposer-vl-7b): The pretrained VLLM model with InternLM as the initialization of the LLM, achieving strong performance on various multimodal benchmarks, e.g., MME Benchmark, MMBench Seed-Bench, CCBench, and MMBench-CN.
- [InternLM-XComposer-7B](https://huggingface.co/internlm/internlm-xcomposer-7b): The finetuned VLLM for *Interleaved Text-Image Composition* and *LLM-based AI assistant*.
  <br>

<!-- 
<p align="center">
    <figcaption align = "center"><b> InternLM-XComposer </b></figcaption>
<p> -->



## News and Updates
* ```2023.9.27``` 🎉🎉🎉 The [evaluation code](./evaluation/) of **InternLM-XComposer-VL-7B** are publicly available.
* ```2023.9.27``` 🎉🎉🎉 **InternLM-XComposer-7B** and **InternLM-XComposer-VL-7B** are publicly available on ModelScope and Hugging Face. 
* ```2023.9.27``` 🎉🎉🎉 We release a [technical report]() for more details of our model series.
<br>

## Evaluation

We evaluate InternLM-XComposer-VL on five multimodal benchmarks: [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation), [MMBench](https://opencompass.org.cn/leaderboard-multimodal), [Seed-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) in the English language, [CCBench](https://opencompass.org.cn/leaderboard-multimodal), [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal) in the simplified chinese language.

   - [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation): A comprehensive evaluation benchmark for multimodal large language models with 14 subtasks.
   - [MMBench](https://opencompass.org.cn/leaderboard-multimodal): A comprehensive evaluation pipeline comprised of meticulously curated multimodal dataset and a novel circulareval strategy using ChatGPT.
   - [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal): A simplified chinese language version of [MMBench](https://opencompass.org.cn/leaderboard-multimodal).
   - [Seed-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard): A multimodal benchmark of 19K multiple-choice questions with accurate human annotations for evaluating Multimodal LLMs.
   - [CCBench](): A multimodal benchmark for chinese cultural comprehension.

InternLM-XComposer-VL outperforms existing vision-language large models on **all the five benchmarks**, demonstrating stronger multilingual comprehension ability.


### MME Benchmark

[MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) is a comprehensive evaluation benchmark for multimodal large language models. It measures both perception and cognition abilities on a total of 14 subtasks, including existence, count, position, color, poster, celebrity, scene, landmark, artwork, OCR, commonsense reasoning, numerical calculation, text translation, and code reasoning.

InternLM-XComposer-VL achieves SOTAs on overall performance evaluation. See more details on [HERE](evaluation/mme/MME_Bench.md).


<p align="center">
Overall Performance
<p>


<center>

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | [InternLM-XComposer-VL](https://github.com/InternLM/InternLM-XComposer) | [InternLM-7B](https://github.com/InternLM/InternLM-XComposer) | 1919.5 |
|   2  | Qwen-VL-Chat    |        Qwen-7B            | 1848.3 |
|   3  |      MMICL      |         FlanT5xxl        | 1810.7 |
|   4  |    Skywork-MM   |      Skywork-MM-13B      | 1775.5 |
|   5  |       BLIVA     |    FlanT5xxl             | 1669.2 |

</center>



<p align="center">
    <img src="evaluation/mme/perception.PNG" width="600"/>
<p>
<p align="center">
    <img src="evaluation/mme/cognition.PNG" width="600"/>
<p>


### MMBench & MMBench-CN

[MMBench](https://opencompass.org.cn/leaderboard-multimodal) is a comprehensive evaluation pipeline comprised of meticulously curated multimodal dataset and a novel circulareval strategy using ChatGPT. It is comprised of 20 ability dimensions defined by MMBench. 

InternLM-XComposer-VL achieves SOTAs on both test and dev split. See more details on [HERE](evaluation/mmbench/MMBench.md).


<p align="center">
MMBench Test Split
<p>

<center>

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 74.4 |
|   2  |    Pink  |        Vicuna-7B            | 74.1 |
|   3  |      JiuTian      |        FLANT5-XXL        | 71.8 |
|   4  |  WeMM   |      InternLM-7B      | 69.0 |
|   5  |     mPLUG-Owl     |    LLaMA2 7B            |  68.5 |

</center>

<p align="center">
    <img src="evaluation/mmbench/mmbench.PNG" width="1000"/>
<p>

<p align="center">
MMBench-CN Test Split
<p>

<center>

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 72.4 |
|   2  |    QWen-VL-Chat | Qwen-7B | 56.3 |
|   3  |    LLaVA       | LLaMA 7B  |36.6 |
|   4  |    VosualGLM   | ChatGLM 6B | 25.6 |
|   5  |    mPLUG-Owl | LLaMA2 7B  | 24.9 |

</center>

<p align="center">
    <img src="evaluation/mmbench/mmbench_cn.PNG" width="1000"/>
<p>

### SEED-Bench

[SEED-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) is a multimodal benchmark of 19K multiple-choice questions with accurate human annotations for evaluating Multimodal LLMs, covering 12 evaluation dimensions including both **image** and **video** understanding. See more details on [HERE](evaluation/seed_bench/SEED.md).

InternLM-VL achieves SOTAs on this benchmark for images.


<p align="center">
SeedBench Image Evaluation
<p>

<center>

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 66.9 |
|   2  |    QWen-VL-Chat | Qwen-7B | 65.4 |
|   3  |    QWen-VL | Qwen-7B | 62.3 |
|   4  |    InstructBLIP-Vicuna   |        Vicuna 7B  | 58.8 |
|   5  |    InstructBLIP   |     Flan-T5-XL  | 57.8 |

</center>

<p align="center">
    <img src="evaluation/seed_bench/seed_bench.PNG" width="1000"/>
<p>

### CCBench

[CCBench](https://opencompass.org.cn/leaderboard-multimodal) is a multimodal benchmark for chinese cultural comprehension. See more details on [HERE](evaluation/seed_bench/MMBench.md).

<p align="center">
CCBench Performance
<p>

<center>

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 47.6 |
|   2  |    QWen-VL-Chat | Qwen-7B | 39.3 |
|   3  |    mPLUG-Owl | LLaMA2 7B  | 12.9 |
|   3  |    InstructBLIP       |        Vicuna 7B  | 12.1 |
|   4  |    VosualGLM   | ChatGLM 6B | 9.2  |

</center>

<p align="center">
    <img src="evaluation/mmbench/ccbench.PNG" width="1000"/>
<p>

## Requirements

* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users)
  <br>

## Installation

Before running the code, make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.
Please refer to the [installation instructions](docs/install.md)

## Quickstart

We provide a simple example to show how to use InternLM-XComposer with 🤗 Transformers.

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
# '阿尔伯特·爱因斯坦（Albert Einstein，1879年3月14日－1955年4月18日），德国裔瑞士籍物理学家。他创立了现代物理学的两大支柱理论：
# 相对论和量子力学， 而质能等价公式E=mc2便是他的相对论思想的明证，因而被公认为是继伽利略、牛顿之后最伟大的物理学家。
# 1999年，爱因斯坦被美国《时代周刊》评选为20世纪的“世纪人物”，他在物理学上的贡献，使他在世界各地受到人们的尊敬。'

# Single-Turn Text-Image Dialogue
text = '请问这张图片里面的人是谁？并介绍下他。'
image = 'examples/images/aiyinsitan.jpg'
response = model.generate(text, image)
# 图片中的男子是阿尔伯特·爱因斯坦（Albert Einstein），一位著名的物理学家和理论物理学家。他于1879年3月14日出生于德国巴登-符腾堡州的乌尔姆市，
# 1955 年4月18日逝世于美国新泽西州普林斯顿市。爱因斯坦在20世纪初提出了狭义相对论和广义相对论，对现代物理学的发展产生了深远影响。

# Multi-Turn Text-Image Dialogue
# 1st turn
text = '图片里面的是谁？'
response, history = model.chat(text=text, image=image, history=None)
# 阿尔伯特·爱因斯坦。

# 2nd turn
text = '他有哪些成就?'
response, history = model.chat(text=text, image=None, history=history)
# 阿尔伯特·爱因斯坦是20世纪最伟大的物理学家之一，他提出了狭义相对论和广义相对论，对现代物理学的发展产生了深远影响。
# 此外，他还提出了著名的质能方程E=mc²，为核能的开发提供了理论基础。

# 3rd turn
text = '他是最伟大的物理学家吗?'
response, history = model.chat(text=text, image=None, history=history)
# 是的，阿尔伯特·爱因斯坦是20世纪最伟大的物理学家之一。
```

## Demo

### Web UI

We provide code for users to build a web UI demo.

Please run the command below:

```
python examples/web_demo.py
```

<br>

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@article{InternLM-XComposer,
  title={InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition},
  author={Zhang and Pan, Dong and Xiaoyi, Wang and Bin, Cao and Yuhang, Xu and Chao, Ouyang and Linke, Zhao and Zhiyuan, Ding and Shuangrui, Zhang and Songyang, Duan and Haodong, Yan and Hang, Zhang and Xinyue, Li and Wei, Li and Jingwen, Chen and Kai, He and Conghui, Zhang and Xingcheng, Qiao and Yu, Lin and Dahua, Wang and Jiaqi},
  journal={arXiv preprint},
  year={2023}
}
```

<br>

## License & Contact Us

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow free commercial usage. To apply for a commercial license, please fill in the application form (English)/申请表（中文）. For other questions or collaborations, please contact internlm@pjlab.org.cn.
