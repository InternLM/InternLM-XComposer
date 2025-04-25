# VL-RewardBench Evaluation Code

## Introduction

This repository provides evaluation code for the [RM-Bench](https://github.com/THU-KEG/RM-Bench) benchmark.

## Setup

1.  **Download Benchmark Data:**
    * The benchmark data file, [`total_dataset.json`](https://huggingface.co/datasets/THU-KEG/RM-Bench/blob/main/total_dataset.json), is provided by the authors of RM-Bench via their Hugging Face dataset repository.
    * Download this file and place it in the root directory of this project.
2.  **Verify Directory Structure:**
    ```
    .
    ├── total_dataset.json
    ├── inference.py
    └── README.md
    ```

## Usage

To run the evaluation script and generate inference results, execute the following command in your terminal:

```bash
python inference.py
```

We provide the official inference results saved in `results.json`.