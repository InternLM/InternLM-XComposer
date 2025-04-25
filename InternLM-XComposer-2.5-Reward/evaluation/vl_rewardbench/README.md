# VL-RewardBench Evaluation Code

## Introduction

This repository provides evaluation code for the [VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench) benchmark.

## Setup

1.  **Download Benchmark Data:**
    * The benchmark data file, [`combined_data_tagged.jsonl`](https://huggingface.co/datasets/MMInstruction/VL-RewardBench/blob/main/inference/data/combined_data_tagged.jsonl), is provided by the authors of VL-RewardBench via their Hugging Face dataset repository.
    * Download this file and place it in the root directory of this project.

2.  **Download Benchmark Images:**
    * The images required for the VL-RewardBench evaluation need to be downloaded separately.
    * We have processed the images and made them available as `images.zip`. Download it from [this GoogleDrive link](https://drive.google.com/file/d/1SXAwYUihHIzoKXJF_kxNeGCMrUNv0URx/view?usp=sharing):
    * After downloading `images.zip`, unzip the file.
    * Place the resulting `images` directory into the root of this project directory.

3.  **Verify Directory Structure:**
    Ensure your project directory looks like this:

    ```
    .
    ├── images/
    │   ├── povid/
    │   │   ├── xxx.jpg
    │   │   └── ...
    │   └── ...
    │   └── wildvision-battle/
    │       ├── xxx.jpg
    │       └── ...
    ├── combined_data_tagged.jsonl
    ├── inference.py
    └── README.md
    ```

## Usage

To run the evaluation script and generate inference results, execute the following command in your terminal:

```bash
python inference.py
```

We provide the official inference results saved in `results.json`.