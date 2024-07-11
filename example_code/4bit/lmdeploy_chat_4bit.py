import nvidia_smi
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl.constants import IMAGE_TOKEN

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

nvidia_smi.nvmlShutdown()

engine_config = TurbomindEngineConfig(
    model_format='awq',
    cache_max_entry_count=0.1,  # you can change this parameter to tune the kv cache memory
)
pipe = pipeline('internlm/internlm-xcomposer2d5-7b-4bit',
                log_level='INFO',
                backend_config=engine_config)
query = f'{IMAGE_TOKEN} Analyze the given image in a detail manner.'
res = pipe((
    query,
    'https://raw.githubusercontent.com/InternLM/InternLM-XComposer/main/examples/dubai.png'
))
print(res.text)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total / 1024 / 1024 / 1024)
print("Free memory:", info.free / 1024 / 1024 / 1024)
print("Used memory:", info.used / 1024 / 1024 / 1024)
