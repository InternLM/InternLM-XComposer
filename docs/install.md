## Example conda environment setup


**Step 1.** Create a conda environment and activate it.
```bash
conda create -n intern_clean python=3.9 -y
conda activate intern_clean
```

**Step 2.** Install PyTorch (We use PyTorch 2.0.1 / CUDA 11.7)
```bash
pip3 install torch torchvision torchaudio
```

**Step 3.** Install require packages
```bash
pip install transformers==4.30.2 timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2

# install flash attention
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install

# install rotaty operator
cd csrc/rotary
pip install -e .
```
