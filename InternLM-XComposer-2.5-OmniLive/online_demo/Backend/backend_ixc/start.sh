export ROOT_DIR=/home/${USER}/InternLM-XComposer/InternLM-XComposer-2.5-OmniLive/internlm-xcomposer2d5-ol-7b

uvicorn main:app --reload --port 7862 --host 0.0.0.0 --loop asyncio