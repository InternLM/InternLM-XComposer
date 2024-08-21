## Example conda environment setup

**Step 1.** Create a conda environment and activate it.

```bash
conda create -n intern_clean python=3.9 -y
conda activate intern_clean
```

**Step 2.** Install PyTorch (We use PyTorch 2.0.1 / CUDA 11.7)

```bash
pip3 install torch torchvision torchaudio

# Please use the following command to install PyTorch so you can replicate our results:
# pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

**Step 3.** Install require packages

```bash
pip install transformers==4.33.2 timm==0.4.12 sentencepiece==0.1.99 gradio==4.13.0 markdown2==2.4.10 xlsxwriter==3.1.2 einops

```

**Step 4.** Install flash-attention2 to save GPU memory

We strongly recommend installing flash-attention2 to save GPU memory, although you can run IXC models without it.

How to install flash-attention2: [https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)

### Optional: 4-bit inference

Please install the pypi package with `pip install lmdeploy`. By default, it depends on CUDA 12.x. 
For a CUDA 11.x environment, please refer to the [installation guide](https://lmdeploy.readthedocs.io/en/latest/get_started.html#installation).

### Optional: Fine-tuning

Fine-turning requires deepspeed, peft (optional for LoRA fine-tuning)

```bash
# install deepspeed
pip install deepspeed

# install peft
pip install peft
```
