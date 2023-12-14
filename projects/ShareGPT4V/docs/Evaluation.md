# Evaluation

In ShareGPT4V, we evaluate models on a diverse set of 12 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

Currently, we mostly utilize the official toolkit or server for the evaluation.

## Evaluate on Custom Datasets

You can evaluate ShareGPT4V mode on your custom datasets by converting your dataset to ShareGPT4V's jsonl format, and evaluate using [`model_vqa.py`](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/share4v/eval/model_vqa.py).

Below we provide a general guideline for evaluating datasets with some common formats.

1. Short-answer (e.g. VQAv2, MME).

```
<question>
Answer the question using a single word or phrase.
```

2. Option-only for multiple-choice (e.g. MMBench, SEED-Bench).

```
<question>
A. <option_1>
B. <option_2>
C. <option_3>
D. <option_4>
Answer with the option's letter from the given choices directly.
```

3. Natural QA (e.g. LLaVA-Bench, MM-Vet).

No postprocessing is needed.

## Scripts

Before preparing task-specific data, **you MUST first download ShareGPT4V's [playgroud.zip](https://drive.google.com/file/d/1tdbFb1Q4WXvBzab2o0B6sVvDJY0lQGUp/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with ShareGPT4V-7B. Extract to `./projects/ShareGPT4V/playground`. This also provides a general structure for all datasets.

### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.
2. Multi-GPU inference.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/vqav2.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/vqav2.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `./playground/data/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Multi-GPU inference.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/gqa.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/gqa.sh
```


### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `./playground/data/eval/vizwiz`.
2. Single-GPU inference.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/vizwiz.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/vizwiz.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission): `./playground/data/eval/vizwiz/answers_upload`.

### ScienceQA

1. Under `./playground/data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/sqa.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `./playground/data/eval/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/textvqa.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/textvqa.sh
```

### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `./playground/data/eval/MME`.
4. Single-GPU inference and evaluate.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/mme.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/mme.sh
```

### MMBench-EN

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.
2. Multi-GPU inference.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/mmbench_en.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/mmbench_en.sh
```
3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `./playground/data/eval/mmbench`.
2. Multi-GPU inference.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/mmbench_cn.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/mmbench_cn.sh
```
3. Submit the results to the evaluation server: `./playground/data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.

### SEED-Bench-Image

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images. Put images under `./playground/data/eval/seed_bench/SEED-Bench-image`.
2. Multiple-GPU inference and evaluate.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/seed.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/seed.sh
```
1. Optionally, submit the results to the leaderboard: `./playground/data/eval/seed_bench/answers_upload` using the official jupyter notebook.

### LLaVA-Bench-in-the-Wild

1. Extract contents of [`llava-bench-in-the-wild`](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) to `./playground/data/eval/llava-bench-in-the-wild`.
2. Single-GPU inference and evaluate.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/llavabench.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/llavabench.sh
```

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `./playground/data/eval/mmvet`.
2. Single-GPU inference.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/mmvet.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/mmvet.sh
```
3. Evaluate the predictions in `./playground/data/eval/mmvet/results` using the official jupyter notebook.

## Q-Bench

1. Download [`llvisionqa_dev.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/llvisionqa_dev.json) (for `dev`-subset) and [`llvisionqa_test.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/llvisionqa_test.json) (for `test`-subset). Put them under `./playground/data/eval/qbench`. 
2. Download and extract [images](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/images_llvisionqa.tar) and put all the images directly under `./playground/data/eval/qbench/images_llviqionqa`.
3. Single-GPU inference.
```Shell
# for single node inference
CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/qbench.sh
```
```Shell
# for slurm inference
srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/qbench.sh
```
We only support dev evaluation in your local machine for now.
