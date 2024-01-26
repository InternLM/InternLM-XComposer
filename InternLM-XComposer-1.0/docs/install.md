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
pip install transformers==4.33.1 timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops

```

### Optional: 4-bit inference

```bash
pip install auto_gptq
```

### Optional: Fine-tuning
Fine-turning requires deepspeed, flash-attention and rotary_emb
```bash
# install deepspeed
pip install deepspeed

# install flash attention
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install

# install rotaty operator
cd csrc/rotary
pip install -e .
```
