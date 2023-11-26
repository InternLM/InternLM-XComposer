# <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo_tight.png" style="vertical-align: -10px;" :height="50px" width="50px"> ShareGPT4V: Improving Large Multi-modal Models with Better Captions

⭐️ [**Star to follow our team's projects !**](https://github.com/InternLM/InternLM-XComposer)

---

🚀🚀🚀 Official implementation of **ShareGPT4V: Improving Large Multi-modal Models with Better Captions**.
<p align="center">
  <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/teaser.png">
</p>

- **Authors**: [Lin Chen*](https://lin-chen.site), [Jinsong Li*](https://li-jinsong.github.io/), [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en), [Pan Zhang](https://panzhang0212.github.io/), [Conghui He](https://conghui.github.io/), [Jiaqi Wang](https://myownskyw7.github.io/), [Feng Zhao📧](https://scholar.google.com/citations?hl=en&user=r6CvuOUAAAAJ), [Dahua Lin📧](http://dahua.site/)

- **Institutes**: University of Science and Technology of China; Shanghai AI Laboratory
- **Resources**: [[Paper](https://arxiv.org/pdf/2311.12793.pdf)] [[Project Page](https://ShareGPT4V.github.io/)] [[<img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo_tight.png" style="vertical-align: -10px;" :height="20px" width="20px">ShareGPT4V Dataset](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)] [[Model Zoo](https://huggingface.co/Lin-Chen/ShareGPT4V-7B)] 
- **ShareGPT4V-7B Demo** [[OpenXLab](https://openxlab.org.cn/apps/detail/xiaoachenyo/ShareGPT4V-7B)] [[🤗HuggingFace](https://huggingface.co/spaces/Lin-Chen/ShareGPT4V-7B)] [[Colab](https://github.com/camenduru/ShareGPT4V-colab)]
- **Share-Captioner Demo** [[OpenXlab](https://openxlab.org.cn/apps/detail/xiaoachenyo/Share-Captioner)] [[🤗HuggingFace](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)]

## 💡 Highlights
- 🔥 A **large-scale** **highly descriptive** image-text dataset
- 🔥 **100K** GPT4-Vision-generated captions, **1.2M** high-quality captions
- 🔥 A **general image captioner**, approaching GPT4-Vision's caption capability.
- 🔥 A superior large multi-modal model, **ShareGPT4V-7B**

## 📜 News
[2023/11/23] We release the [web demo](https://huggingface.co/spaces/Lin-Chen/Share-Captioner) of general Share-Captioner!💥

[2023/11/23] We release code to build your local demo of ShareGPT4V-7B!💥

[2023/11/22] [Web demo](https://huggingface.co/spaces/Lin-Chen/ShareGPT4V-7B) and [checkpoint](https://huggingface.co/Lin-Chen/ShareGPT4V-7B) are available now!💥

[2023/11/21] [ShareGPT4V Dataset](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) is available now!💥

[2023/11/20] The [paper]([ShareGPT4V.pdf](https://arxiv.org/pdf/2311.12793.pdf)) and [project page](https://ShareGPT4V.github.io/) are released!

## 👨‍💻 Todo
- [ ] Training and evaluation code for ShareGPT4V-7B
- [x] Web demo and local demo of ShareGPT4V-7B
- [x] Checkpoints of ShareGPT4V-7B

## 🤖 Model Zoo

| Name | LLM | Checkpoint | LLaVA-Bench-Wild | MME-perception | MME-cognition | MMBench | MMBench-CN | SEED-image | MM-Vet | QBench | SQA-image | VQA-v2 | VizWiz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ShareGPT4V-7B | Vicuna-7B | [ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B) | 72.6 | 1567.4 | 376.4 | 68.8 | 62.2 | 69.7 | 37.6 | 63.4 | 68.4 | 80.6 | 57.2 |

## 🛠️Usage

### Build Local Demo
First, prepare the environment.

```
# Create env
conda create -n sharegpt4v python=3.10 -y
conda activate sharegpt4v

cd projects/ShareGPT4V/

# Clone llava 
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA & pip install -e .

# You may get warning due to the gradio version. Do not worry about it.
pip install gradio==4.5.0 
```

Then, you should update only one line in the builder script of the vision encoder to enable loading fine-tuned vision tower:
```python
# replace line 8 in llava/model/multimodal_encoder/builder.py with following line:
  if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
```

Finally, you can build your local demo by:
```
# move to ShareGPT4V/ from LLaVA/
cd ..

# run script
python app.py
```

### Environment Set Up
Follow [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) to set up the code and environment. Remember to update only one line in the builder script of the vision encoder to enable loading fine-tuned vision tower.

### Data Preparation

Our captions data are available at [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) in the JSON format.

In addition to preparing the datasets specified in [LLaVA-1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md), it is necessary to procure the [SAM](https://ai.meta.com/datasets/segment-anything-downloads/) dataset (Only the first 50 parquets have been used so far.) and various [web data](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Currently, we provide direct download access to the web data. However, to avoid potential disputes, we plan to release URLs for these datasets rather than the raw data in the near future.

```none
Your Project Path
├── ...
├── data
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
│   ├── sam
│   │   ├── images
│   ├── share_textvqa
│   │   ├── images
│   ├── web-celebrity
│   │   ├── images
│   ├── web-landmark
│   │   ├── images
│   ├── wikiart
│   │   ├── images
├── ...
```

## ❤️ Acknowledgments
- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@misc{chen2023sharegpt4v,
      title={ShareGPT4V: Improving Large Multi-Modal Models with Better Captions}, 
      author={Lin Chen and Jisong Li and Xiaoyi Dong and Pan Zhang and Conghui He and Jiaqi Wang and Feng Zhao and Dahua Lin},
      year={2023},
      eprint={2311.12793},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
