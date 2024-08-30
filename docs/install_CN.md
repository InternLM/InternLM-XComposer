## 配置 conda 环境案例

**Step 1.** 创建一个 conda 环境并激活。

```bash
conda create -n intern_clean python=3.9 -y
conda activate intern_clean
```

**Step 2.** 安装 PyTorch (我们使用 PyTorch 2.0.1 / CUDA 11.7 测试通过)

```bash
pip3 install torch torchvision torchaudio

# 推荐使用以下命令安装Pytorch，以准确复现结果:
# pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

**Step 3.** 安装需要的包

```bash
pip install transformers==4.33.2 timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops
```

### 可选: 4-bit测试额外需要安装的包

请使用“pip install lmdeploy”安装 pypi 包。默认情况下，它依赖于 CUDA 12.x。
对于 CUDA 11.x 环境，请参阅[安装指南](https://lmdeploy.readthedocs.io/en/latest/get_started.html#installation)。

### 可选: 微调 (Fine-tuning)

微调需要安装deepspeed，peft (用于 LoRA 微调)

```bash
# install deepspeed
pip install deepspeed==0.12.3

# install peft
pip install peft==0.8.2
```
