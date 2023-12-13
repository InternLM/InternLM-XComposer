# <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo_tight.png" style="vertical-align: -10px;" :height="50px" width="50px"> ShareGPT4V: Improving Large Multi-modal Models with Better Captions

⭐️ [**Star to follow our team's projects !**](https://github.com/InternLM/InternLM-XComposer)

---

🚀🚀🚀 Official implementation of **ShareGPT4V: Improving Large Multi-modal Models with Better Captions**.
<p align="center">
  <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/teaser.png">
</p>

- **Authors**: [Lin Chen*](https://lin-chen.site), [Jinsong Li*](https://li-jinsong.github.io/), [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en), [Pan Zhang](https://panzhang0212.github.io/), [Conghui He](https://conghui.github.io/), [Jiaqi Wang](https://myownskyw7.github.io/), [Feng Zhao📧](https://scholar.google.com/citations?hl=en&user=r6CvuOUAAAAJ), [Dahua Lin📧](http://dahua.site/)

- **Institutes**: University of Science and Technology of China; Shanghai AI Laboratory
- **Resources**: [[Paper](https://arxiv.org/pdf/2311.12793.pdf)] [[Project Page](https://ShareGPT4V.github.io/)] [[<img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo_tight.png" style="vertical-align: -10px;" :height="20px" width="20px">ShareGPT4V Dataset](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)] 
- **Models**: [[ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B)] [[ShareCaptioner](https://huggingface.co/Lin-Chen/ShareCaptioner)]
- **ShareGPT4V-7B Demo** [[OpenXLab](https://openxlab.org.cn/apps/detail/xiaoachenyo/ShareGPT4V-7B)] [[🤗HuggingFace](https://huggingface.co/spaces/Lin-Chen/ShareGPT4V-7B)] [[Colab](https://github.com/camenduru/ShareGPT4V-colab)]
- **Share-Captioner Demo** [[OpenXlab](https://openxlab.org.cn/apps/detail/xiaoachenyo/Share-Captioner)] [[🤗HuggingFace](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)]

## 💡 Highlights
- 🔥 A **large-scale** **highly descriptive** image-text dataset
- 🔥 **100K** GPT4-Vision-generated captions, **1.2M** high-quality captions
- 🔥 A **general image captioner**, approaching GPT4-Vision's caption capability.
- 🔥 A superior large multi-modal model, **ShareGPT4V-7B**

## 📜 News
[2023/12/13] Training and evaluation code is available.

[2023/12/13] **Local ShareCaptioner** is available now! You can utilize it to generate high-quality captions for your dataset with batch inference by directly run `tools/share-cap_batch_infer.py`.

[2023/11/23] We release the [web demo](https://huggingface.co/spaces/Lin-Chen/Share-Captioner) of general Share-Captioner!💥

[2023/11/23] We release code to build your local demo of ShareGPT4V-7B!💥

[2023/11/22] [Web demo](https://huggingface.co/spaces/Lin-Chen/ShareGPT4V-7B) and [checkpoint](https://huggingface.co/Lin-Chen/ShareGPT4V-7B) are available now!💥

[2023/11/21] [ShareGPT4V Dataset](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) is available now!💥

[2023/11/20] The [paper]([ShareGPT4V.pdf](https://arxiv.org/pdf/2311.12793.pdf)) and [project page](https://ShareGPT4V.github.io/) are released!

## 👨‍💻 Todo
- [x] Training and evaluation code for ShareGPT4V-7B
- [x] Local ShareCaptioner
- [x] Web demo and local demo of ShareGPT4V-7B
- [x] Checkpoints of ShareGPT4V-7B

## 🤖 Model Zoo

| Name | LLM | Checkpoint | LLaVA-Bench-Wild | MME-perception | MME-cognition | MMBench | MMBench-CN | SEED-image | MM-Vet | QBench | SQA-image | VQA-v2 | VizWiz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ShareGPT4V-7B | Vicuna-7B | [ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B) | 72.6 | 1567.4 | 376.4 | 68.8 | 62.2 | 69.7 | 37.6 | 63.4 | 68.4 | 80.6 | 57.2 |

## Install

```bash
git clone https://github.com/InternLM/InternLM-XComposer --depth=1
cd projects/ShareGPT4V
conda create -n share4v python=3.10 -y
conda activate share4v

pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Demo
You can build your local demo by:
```
# run script
python tools/app.py
```

## Data Preparation

You should follow this instruction [Data.md](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V/projects/ShareGPT4V/docs/Data.md) to manage the datasets. Currently, we provide direct download access to the web data. However, to avoid potential disputes, we plan to release URLs for these datasets rather than the raw data in the near future.

## Train

ShareGPT4V model training consists of two stages: (1) feature alignment stage: use our ShareGPT4V-PT dataset with 1.2M ShareCaptioner-generated high-quality image-text pairs to finetune the vision encoder, projector, and the LLM to align the textual and visual modalities; (2) visual instruction tuning stage: finetune the projector and LLM to teach the model to follow multimodal instructions.

To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size x gradient_accumulation_steps x num_gpus`.

### Hyperparameters
We use a similar set of hyperparameters as ShareGPT4V-7B in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShareGPT4V-7B | 256 | 2e-5 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShareGPT4V-7B | 128 | 2e-5 | 1 | 2048 | 0 |

### Pretrain

More details **TBD**.

You can run `projects/ShareGPT4V/scripts/sharegpt4v/slurm_pretrain_7b.sh` to pretrain the model.

### Finetune

More details **TBD**.

You can run `projects/ShareGPT4V/scripts/sharegpt4v/slurm_finetune_7b.sh` to finetune the model.

## Evaluation

To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](). **TBD**


## ❤️ Acknowledgments
- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@misc{chen2023sharegpt4v,
      title={ShareGPT4V: Improving Large Multi-Modal Models with Better Captions}, 
      author={Lin Chen and Jinsong Li and Xiaoyi Dong and Pan Zhang and Conghui He and Jiaqi Wang and Feng Zhao and Dahua Lin},
      year={2023},
      eprint={2311.12793},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
