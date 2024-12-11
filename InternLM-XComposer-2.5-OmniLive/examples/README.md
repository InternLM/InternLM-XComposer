# InternLM-XComposer-2.5-OL Quick Start Guidelines

This directory contains some examples for working with **InternLM-XComposer-2.5-OL**.

## Getting Started

Follow the steps below to set up the project and start using the provided scripts.

### Step 1: Clone the Repository

Clone the repository to your local machine:

```shell
git clone https://github.com/InternLM/InternLM-XComposer.git
cd InternLM-XComposer/InternLM-XComposer-2.5-OmniLive
```

### Step 2: Download the Model

```shell
huggingface-cli download internlm/internlm-xcomposer2d5-ol-7b \
  --local-dir internlm-xcomposer2d5-ol-7b \
  --local-dir-use-symlinks False \
  --resume-download
```

### Step 3: Run Inference

Three inference scripts are provided for different use cases:

#### Infer the Audio LLM

Run the following command to perform inference with the audio-based model:

```shell
python examples/infer_audio.py
```

#### Infer the Base LLM

```shell
python examples/infer_llm_base.py
```

#### Infer the LLM with Memory

```shell
python examples/merge_lora.py
python examples/infer_llm_with_memory.py
```