## Run InternLM-XComposer-2d5-4bit with LMDeploy

Thanks to the LMDeploy team for providing AWQ quantization support, https://github.com/InternLM/lmdeploy/pull/1932. The following document shows how to set up LMDeploy to run InternLM-XComposer-2d5-4bit:

1. Download the LMDeploy branch that supports InternLM-XComposer-2d5-4bit: 
```bash
git clone --depth=1 -b xcomposer2d5 git@github.com:irexyc/lmdeploy.git 
```

2. Build LMDeploy from source   

Following this guide https://lmdeploy.readthedocs.io/en/latest/build.html to build lmdeploy.

3. Install nvidia-smi python api to compute used memory
```bash
pip install nvidia-ml-py3
```

4. Run InternLM-XComposer-2d5-4bit

```python
python lmdeploy_chat_4bit.py
```


## GPU Memory comparison

It is noted for lmdeploy we set the value of `cache_max_entry_count=0.01` to save the cost gpu memory. The program is tested on pytorch 2.2.2+cu118.

|               | GPU Memory (GB) |
|---------------|-----------------|
| pytorch       | 58.65           |
| lmdeploy      | 31.81           |
| lmdeploy-4bit | 23.21           |
