# Evaluation

## InternLM-XComposer2-4KHD Evaluation
We support the evaluation of InternLM-XComposer2-4KHD in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)


## InternLM-XComposer2-VL Evaluation
In InternLM-XComposer2, we evaluate models on a diverse set of 13 benchmarks with the following scripts. The evaluation is also supported in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) (The results will have slight difference).
 
### MathVista

1. Run the notebook `MathVista.ipynb`.  

<details>
  <summary>
    <b>MathVista results</b>
  </summary>

| test | testmini |
|---------|--------|
| 57.93   | 57.6  |

</details>

### MMMU

1. Run the notebook `MMMU/MMMU_Validation.ipynb`.  

<details>
  <summary>
    <b>MMMU results</b>
  </summary>

| test | val |
|---------|--------|
| 38.2   | 42.0  |

</details>


### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `./data/`.
4. Single-GPU inference.
```Shell
cd MME
CUDA_VISIBLE_DEVICES=0 python -u eval.py
```

<details>
  <summary>
    <b>MME results</b>
  </summary>

```
=========== Perception ===========
total score: 1711.9952981192478

         existence  score: 195.0
         count  score: 160.0
         position  score: 163.33333333333334
         color  score: 195.0
         posters  score: 171.08843537414964
         celebrity  score: 153.8235294117647
         scene  score: 164.75
         landmark  score: 176.0
         artwork  score: 185.5
         OCR  score: 147.5


=========== Cognition ===========
total score: 530.7142857142858

         commonsense_reasoning  score: 145.71428571428572
         numerical_calculation  score: 137.5
         text_translation  score: 147.5
         code_reasoning  score: 100.0

```
</details>

### MMBench

1. Download [`mmbench_test_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_20230712.tsv) and put under `./data/`.
2. Single-GPU inference.
```Shell
cd MMBench
CUDA_VISIBLE_DEVICES=0 python -u eval.py
```
3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `Output/submit_test.xlsx`.
<details>
  <summary>
    <b>MMBench Testset results</b>
  </summary>

| Overall | AR    | CP    | FP-C | FP-S  | LR    | RR    |
|---------|-------|-------|------|-------|-------|-------|
| 79.64   | 82.35 | 83.82 | 72   | 85.75 | 66.47 | 75.11 |

</details>



### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `./data/`.
2. Single-GPU inference.
```Shell
cd MMBench
CUDA_VISIBLE_DEVICES=0 python -u eval_cn.py
```
3. Submit the results to the evaluation server: `Output/submit_dev_cn.xlsx`.
<details>
  <summary>
    <b>MMBench-CN Testset results</b>
  </summary>

| Overall | AR    | CP    | FP-C  | FP-S  | LR    | RR    |
|---------|-------|-------|-------|-------|-------|-------|
| 77.57   | 84.37 | 83.29 | 69.23 | 83.16 | 60.69 | 68.72 |

</details>



### SEED-Bench 

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images and the json file. Put images under `./data/SEED-Bench-image` and `./data/SEED-Bench.json`. 
2. Single-GPU inference.
```Shell
cd SEED
CUDA_VISIBLE_DEVICES=0 python -u eval.py
```
<details>
  <summary>
    <b>Seed-Bench Image Set results</b>
  </summary>

| Overall | Instance Attributes | Instance Identity | Instance Interaction | Instance Location | Instances Counting | Scene Understand | Spatial Relation | Text Understand | Visual Reasoning  |
|---------|---------------------|-------------------|----------------------|-------------------|--------------------|---------------------|------------------|--------------------|-------------------|
| 75.87   | 77.84               | 78.37             | 79.38                | 72.69             | 69.96              | 79.32               | 63.47            | 67.85              | 80.06             |

</details>


### AI2D

1. Download the processed images [here](https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing), unzip the images to `./data/ai2d/`. 
2. Run the notebook `AI2D.ipynb`.
 
<details>
  <summary>
    <b>AI2D results</b>
  </summary>

| Overall |  
|---------|  
| 78.73   |  

</details>


### ChartQA

1. Download the processed images from the [official webset](https://huggingface.co/datasets/ahmed-masry/ChartQA/tree/main), unzip the images to `./data/chartqa/`. 
2. Run the notebook `ChartQA.ipynb`.
 
<details>
  <summary>
    <b>AI2D results</b>
  </summary>

| Overall | Human | Augmented |
|---------|-------|-----------|
| 72.68   | 63.52 | 81.84     |

</details>


### LLaVA-Bench-in-the-Wild

1. Extract contents of [`llava-bench-in-the-wild`](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) to `./data/llava-bench-in-the-wild`.
2. Run the notebook `LLaVA_Wild_Eval.ipynb`.
<details>
  <summary>
    <b>LLaVA Wild results</b>
  </summary>

|                     | Answer/GPT4 | GPT4 score | Answer score  |
|---------------------|-------------|------------|---------------|
| llava_bench_complex | 92.3        | 83.9       | 77.5          |
| llava_bench_conv    | 67.6        | 87.1       | 58.8          |
| llava_bench_detail  | 78.8        | 83.3       | 65.7          |
| all                 | 81.8        | 84.7       | 69.2          |

</details>


### MM-Vet

1. Download the [mm-vet.zip](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip), unzip and move them to `./data/mm-vet/`. 
2. Run the notebook `MMVet_Eval.ipynb`.
2. Run the notebook `MMVet_evaluator.ipynb`.
<details>
  <summary>
    <b>MM-Vet results</b>
  </summary>

| rec  | ocr  | know | gen  | spat | math | total |
|------|------|------|------|------|------|-------|
| 50.8 | 50.5 | 35.0 | 38.5 | 52.8 | 41.9 | 51.2  |

</details>


### Q-Bench

1. Download [`llvisionqa_dev.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/llvisionqa_dev.json) (for `dev`-subset) and [`llvisionqa_test.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/llvisionqa_test.json) (for `test`-subset). Put them under `./data/qbench`. 
2. Download and extract [images](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/images_llvisionqa.tar) and put all the images directly under `./data/qbench/llv_dev`.
3. Run the notebook `QBench.ipynb`. 
4. For the testset results, set the split to `test` and submit the results by instruction [here](https://github.com/VQAssessment/Q-Bench#option-1-submit-results): `Output/QBench_test_en_InternLM_XComposer_VL.json.pth`.

<details>
  <summary>
    <b>Q-Bench results</b>
  </summary>

| test-en | dev-en |
|---------|--------|
| 72.52   | 70.70  |

</details>


### Chinese-Q-Bench

1. Download [`质衡-问答-验证集.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/%E8%B4%A8%E8%A1%A1-%E9%97%AE%E7%AD%94-%E9%AA%8C%E8%AF%81%E9%9B%86.json) (for `dev`-subset) and [`质衡-问答-测试集.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/%E8%B4%A8%E8%A1%A1-%E9%97%AE%E7%AD%94-%E6%B5%8B%E8%AF%95%E9%9B%86.json) (for `test`-subset). Put them under `./data/qbench`. 
2. Download and extract [images](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/images_llvisionqa.tar) and put all the images directly under `./data/qbench/llv_dev`.
3. Run the notebook `QBench.ipynb`. 
4. For the testset results, set the split to `test` and submit the results by instruction [here](https://github.com/VQAssessment/Q-Bench#option-1-submit-results): `Output/QBench_test_cn_InternLM_XComposer_VL.json.pth`.

<details>
  <summary>
    <b>Chinese-Q-Bench results</b>
  </summary>

| test-cn | dev-cn |
|---------|--------|
| 70.32   | 72.11  |

</details>


### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put three json files under `data/json_files/`.
2. Run the notebook `POPE.ipynb`.
<details>
  <summary>
    <b>POPE results</b>
  </summary>

```
Average F1-Score: 0.8773077717611343

Adversarial 
TP	FP	TN	FN	
1217	91	1409	283
Accuracy: 0.8753333333333333
Precision: 0.9304281345565749
Recall: 0.8113333333333334
F1 score: 0.8668091168091169
Yes ratio: 0.436

Popular
TP	FP	TN	FN	
1217	58	1442	283
Accuracy: 0.8863333333333333
Precision: 0.9545098039215686
Recall: 0.8113333333333334
F1 score: 0.877117117117117
Yes ratio: 0.425

Random
TP	FP	TN	FN	
1217	24	1386	283
Accuracy: 0.8945017182130585
Precision: 0.9806607574536664
Recall: 0.8113333333333334
F1 score: 0.8879970813571689
Yes ratio: 0.42646048109965634
```
</details>


### HallusionBench

1. Download the [question](https://github.com/tianyi-lab/HallusionBench/blob/main/HallusionBench.json) and [images](https://drive.google.com/file/d/1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0/view?usp=sharing), move them to `./data/hallu/`. 
2. Run the notebook `AI2D.ipynb`.
 
<details>
  <summary>
    <b>HallusionBench Image Part results</b>
  </summary>

|  aAcc |  fAcc | qAcc | 
|---------|  --------|  --------|  
| 60.3   | 30.01 |  32.97  | 

</details>
