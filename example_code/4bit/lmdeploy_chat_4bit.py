import nvidia_smi
from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
nvidia_smi.nvmlShutdown()

engine_config = TurbomindEngineConfig(model_format='awq', cache_max_entry_count=0.1)
pipe = pipeline('internlm/internlm-xcomposer2d5-7b-4bit', backend_config=engine_config)
image = load_image('examples/dubai.png')
response = pipe(('describe this image', image))
print(response.text)

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total / 1024 / 1024 / 1024)
print("Free memory:", info.free / 1024 / 1024 / 1024)
print("Used memory:", info.used / 1024 / 1024 / 1024)
