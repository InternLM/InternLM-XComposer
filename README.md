<p align="center">
    <img src="logo.png" width="400"/>
</p>
<p align="center">
    <b><font size="6">InternLM-XComposer</font></b>
</p>

<!-- <div align="center">
        InternLM-XComposer <a href="">🤖 <a> <a href="">🤗</a>&nbsp ｜ InternLM-VL <a href="">🤖 <a> <a href="">🤗</a>&nbsp | Technical Report <a href=""> <a> 📄  -->

<div align="center">
        InternLM-XComposer <a href="https://huggingface.co/internlm/internlm-xcomposer-7b">🤗</a>&nbsp ｜ InternLM-XComposer-VL <a href="https://huggingface.co/internlm/internlm-xcomposer-vl-7b">🤗</a>&nbsp | Technical Report <a href="https://arxiv.org/pdf/2309.15112.pdf">  📄 <a>

[English](./README.md) | [简体中文](./README_CN.md)

</div>

<br>




**InternLM-XComposer** is a vision-language large model (VLLM) based on [InternLM](https://github.com/InternLM/InternLM/tree/main) for advanced text-image comprehension and composition. InternLM-XComposer has serveal appealing properties:

- **Interleaved Text-Image Composition**: InternLM-XComposer can effortlessly generate coherent and contextual articles that seamlessly integrate images, providing a more engaging and immersive reading experience. The interleaved text-image composition is implemented in following steps:

    1. **Text Generation**: It crafts long-form text based on human-provided instructions.
    2. **Image Spoting and Captioning**: It pinpoints optimal locations for image placement and furnishes image descriptions.
    3. **Image Retrieval and Selection**: It select image candidates and identify the image that optimally complements the content.

- **Comprehension with Rich Multilingual Knowledge**: The text-image comprehension is empowered by training on extensive multi-modal multilingual concepts with carefully crafted strategies, resulting in a deep understanding of visual content.
- **Strong performance**: It consistently achieves state-of-the-art results across various benchmarks for vision-language large models, including [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) (English), [MMBench](https://opencompass.org.cn/leaderboard-multimodal) (English), [Seed-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) (English), [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal)(Chinese), and [CCBench](https://opencompass.org.cn/leaderboard-multimodal)(Chinese).

We release InternLM-XComposer series in two versions:

- [InternLM-XComposer-VL-7B](https://huggingface.co/internlm/internlm-xcomposer-vl-7b): The pretrained VLLM model with InternLM as the initialization of the LLM, achieving strong performance on various multimodal benchmarks, e.g., MME Benchmark, MMBench Seed-Bench, CCBench, and MMBench-CN.
- [InternLM-XComposer-7B](https://huggingface.co/internlm/internlm-xcomposer-7b): The finetuned VLLM for *Interleaved Text-Image Composition* and *LLM-based AI assistant*.
  <br>

<!-- 
<p align="center">
    <figcaption align = "center"><b> InternLM-XComposer </b></figcaption>
<p> -->


## Demo






https://github.com/InternLM/InternLM-XComposer/assets/22662425/fdb89a38-c650-45f2-b5b7-51182e89a5cc




Please refer to [Chinese Demo](https://github.com/InternLM/InternLM-XComposer/blob/main/README_CN.md#demo) for the demo of the Chinese version.




## News and Updates
* ```2023.9.27``` 🎉🎉🎉 The [evaluation code](./evaluation/) of **InternLM-XComposer-VL-7B** are publicly available.
* ```2023.9.27``` 🎉🎉🎉 **InternLM-XComposer-7B** and **InternLM-XComposer-VL-7B** are publicly available on ModelScope and Hugging Face. 
* ```2023.9.27``` 🎉🎉🎉 We release a [technical report](https://arxiv.org/pdf/2309.15112.pdf) for more details of our model series.
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
</p>


<div align="center">

| Rank |      Model      |          Version         |  Score  |
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

[MMBench](https://opencompass.org.cn/leaderboard-multimodal) is a comprehensive evaluation pipeline comprised of meticulously curated multimodal dataset and a novel circulareval strategy using ChatGPT. It is comprised of 20 ability dimensions defined by MMBench. [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal) is the Chinese language version of MMBench.

InternLM-XComposer-VL achieves SOTAs on the test split of both MMBench and MMBench-CN. See more details on [HERE](evaluation/mmbench/MMBench.md).


<p align="center">
MMBench Test Split
</p>

<div align='center'>

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 74.4 |
|   2  |    Pink  |        Vicuna-7B            | 74.1 |
|   3  |      JiuTian      |        FLANT5-XXL        | 71.8 |
|   4  |  WeMM   |      InternLM-7B      | 69.0 |
|   5  |     mPLUG-Owl     |    LLaMA2 7B            |  68.5 |

</div>

<p align="center">
    <img src="evaluation/mmbench/mmbench_en.PNG" width="1000"/>
</p>

<p align="center">
MMBench-CN Test Split
</p>

<div align='center'>

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 72.4 |
|   2  |    QWen-VL-Chat | Qwen-7B | 56.3 |
|   3  |    LLaVA       | LLaMA 7B  |36.6 |
|   4  |    VosualGLM   | ChatGLM 6B | 25.6 |
|   5  |    mPLUG-Owl | LLaMA2 7B  | 24.9 |

</div>

<p align="center">
    <img src="evaluation/mmbench/mmbench_cn_en.PNG" width="1000"/>
</p>

### SEED-Bench

[SEED-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) is a multimodal benchmark of 19K multiple-choice questions with accurate human annotations for evaluating Multimodal LLMs, covering 12 evaluation dimensions including both **image** and **video** understanding. See more details on [HERE](evaluation/seed_bench/SEED.md).

InternLM-XComposer-VL achieves SOTAs on this benchmark for images.


<p align="center">
SeedBench Image Evaluation
</p>

<div align="center">

| Rank |      Model      |          Version         |  Score  |
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

[CCBench](https://opencompass.org.cn/leaderboard-multimodal) is a multimodal benchmark for chinese cultural comprehension. See more details on [HERE](evaluation/seed_bench/MMBench.md).

<p align="center">
CCBench Performance
</p>

<div align="center">

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 47.6 |
|   2  |    QWen-VL-Chat | Qwen-7B | 39.3 |
|   3  |    mPLUG-Owl | LLaMA2 7B  | 12.9 |
|   3  |    InstructBLIP       |        Vicuna 7B  | 12.1 |
|   4  |    VosualGLM   | ChatGLM 6B | 9.2  |

</div>

<p align="center">
    <img src="evaluation/mmbench/ccbench_en.PNG" width="1000"/>
</p>

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
text = 'Please introduce Einstein.'
response = model.generate(text)
print(response)
# 'Albert Einstein was a German-born theoretical physicist. He developed the general theory of relativity, 
# one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for its influence 
# on the philosophy of science. In 1921, Einstein was awarded the Nobel Prize in Physics "for his services to 
# theoretical physics, and especially for his discovery of the law of the photoelectric effect.'


# Single-Turn Text-Image Dialogue
text = 'Please introduce the person in this picture in detail.'
image = 'examples/images/aiyinsitan.jpg'
response = model.generate(text, image)
print(response)
# 'The person in the picture is Albert Einstein, a renowned theoretical physicist and one of the most influential 
# scientists of the 20th century. He was born on March 14, 1879, in Ulm, Germany, and died on April 18, 1955, 
# in Princeton, New Jersey.'


# Multi-Turn Text-Image Dialogue
# 1st turn
text = 'Who is in the picture?'
response, history = model.chat(text=text, image=image, history=None)
print(response)
# 'Albert Einstein is in the picture.'

# 2nd turn
text = 'What are his achievements?'
response, history = model.chat(text=text, image=None, history=history)
print(response)
# 'Albert Einstein was a German-born theoretical physicist who developed the general theory of relativity, 
# one of the two pillars of modern physics (alongside quantum mechanics). He is best known for his mass–energy 
# equivalence formula E = mc2 (which has been dubbed "the world's most famous equation"), and his explanation of 
# the photoelectric effect, both of which are examples of his special and general theories of relativity.'

# 3rd turn
text = 'Is he the greatest physicist?'
response, history = model.chat(text=text, image=None, history=history)
print(response)
# 'Yes, Albert Einstein is widely regarded as one of the greatest physicists of all time'
```

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
@misc{zhang2023internlmxcomposer,
      title={InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition}, 
      author={Pan Zhang and Xiaoyi Dong Bin Wang and Yuhang Cao and Chao Xu and Linke Ouyang and Zhiyuan Zhao and Shuangrui Ding and Songyang Zhang and Haodong Duan and Hang Yan and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      year={2023},
      eprint={2309.15112},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<br>

## License & Contact Us

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow free commercial usage. To apply for a commercial license, please fill in the application form (English)/申请表（中文）. For other questions or collaborations, please contact internlm@pjlab.org.cn.
