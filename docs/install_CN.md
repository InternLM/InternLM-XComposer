## 配置 conda 环境案例


**Step 1.** 创建一个 conda 环境并激活。
```bash
conda create -n intern_clean python=3.9 -y
conda activate intern_clean
```

**Step 2.** 安装 PyTorch (我们使用 PyTorch 2.0.1 / CUDA 11.7 测试通过)
```bash
pip3 install torch torchvision torchaudio
```

**Step 3.** 安装需要的包
```bash
pip install transformers==4.30.2 timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 ninja

# install flash attention
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install

# install rotaty operator
cd csrc/rotary
pip install -e .
```
