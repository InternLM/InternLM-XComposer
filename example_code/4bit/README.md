## Run InternLM-XComposer-2d5-4bit with LMDeploy

Thanks to the LMDeploy team for providing AWQ quantization support (https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/xcomposer2d5.md#quantization). We compare the memory usage between the FP16 model and the 4-bit model, setting cache_max_entry_count=0.01 to reduce GPU memory usage and better observe memory savings. The program was tested on PyTorch 2.2.2+cu118.

|               | GPU Memory (GB) |
|---------------|-----------------|
| lmdeploy      | 31.81           |
| lmdeploy-4bit | 23.21           |
