<!-- # <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo.png" style="vertical-align: -10px;" :height="52px" width="52px"> ShareGPT4V: Improving Large Multi-modal Models with Better Captions -->
<h1>
  <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo.png" alt="Logo" height="50" style="vertical-align: top;">
  ShareGPT4V: Improving Large Multi-modal Models with Better Captions
</h1>


---

🚀🚀🚀 Official implementation of **ShareGPT4V: Improving Large Multi-modal Models with Better Captions**.
<p align="center">
  <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/teaser.png">
</p>

- **Authors**: [Lin Chen*](https://lin-chen.site), [Jinsong Li*](https://li-jinsong.github.io/), [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en), [Pan Zhang](https://panzhang0212.github.io/), [Conghui He](https://conghui.github.io/), [Jiaqi Wang](https://myownskyw7.github.io/), [Feng Zhao📧](https://scholar.google.com/citations?hl=en&user=r6CvuOUAAAAJ), [Dahua Lin📧](http://dahua.site/)

- **Institutes**: University of Science and Technology of China; Shanghai AI Laboratory
- **Resources**: [[Arxiv]]() [[Project Page](https://ShareGPT4V.github.io/)] [[Demo]()] [[Data](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)] [[Model Zoo]()] [[🤗Space]()]

## 💡 Highlights
- 🔥 A **large-scale** **highly descriptive** image-text dataset
- 🔥 **100K** GPT4-Vision-generated captions, **1.2M** high-quality captions
- 🔥 A **general image captioner**, approaching GPT4-Vision's caption capability.
- 🔥 A superior large multi-modal model, **ShareGPT4V-7B**

## 📜 News
[2023/11/20] [ShareGPT4V Dataset](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) is available now!💥
[2023/11/20] The [paper]() and [project page](https://ShareGPT4V.github.io/) are released!

## 👨‍💻 Todo
- [ ] Training and evaluation code for ShareGPT4V-7B
- [ ] Web demo of ShareGPT4V-7B
- [ ] Checkpoints of ShareGPT4V-7B

## 🛠️Usage

### Environment Set Up
Follow [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) to set up the code and environment.

### Data Preparation

Our captions data are available at [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) in the JSON format.

In addition to preparing the datasets specified in LLaVA-1.5, it is necessary to procure the [SAM](https://ai.meta.com/datasets/segment-anything-downloads/) dataset and various [web data](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Currently, we provide direct download access to the web data. However, to avoid potential disputes, we plan to release URLs for these datasets rather than the raw data in the near future.

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
@article{chen2023sharegpt4v,
      title={ShareGPT4V: Improving Large Multi-modal Models with Better Captions}, 
      author={Lin Chen and Jinsong Li and Xiaoyi Dong and Pan Zhang and Conghui He and Jiaqi Wang and Feng Zhao and Dahua Lin},
      year={2023},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
