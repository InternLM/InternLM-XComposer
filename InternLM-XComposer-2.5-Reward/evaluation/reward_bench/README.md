# VL-RewardBench Evaluation Code

## Introduction

This repository provides evaluation code for the [Reward Bench](https://huggingface.co/datasets/allenai/reward-bench) benchmark.

## Setup

1.  **Download Benchmark Data:**
    * The benchmark data file, [`filtered-00000-of-00001.parquet`](https://huggingface.co/datasets/allenai/reward-bench/blob/main/data/filtered-00000-of-00001.parquet), is provided by the authors of Reward Bench via their Hugging Face dataset repository.
    * Download this file and place it in the root directory of this project.
2.  **Verify Directory Structure:**
    ```
    .
    ├── filtered-00000-of-00001.parquet
    ├── inference.py
    └── README.md
    ```

## Usage

To run the evaluation script and generate inference results, execute the following command in your terminal:

```bash
python inference.py
```

We provide the official inference results saved in `results.json`.