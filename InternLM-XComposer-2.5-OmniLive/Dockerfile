# Use the base image
FROM openmmlab/lmdeploy:v0.6.0a0-cu11

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install 'ms-swift[llm]' -U && \
    pip install transformers -U && \
    pip install torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install librosa decord opencv-python websocket funasr ffmpeg-python opencc openai-whisper nest_asyncio cn2an pypinyin g2p_en txtsplit cached_path num2words anyascii && \
    pip install flash-attn --no-build-isolation

# Set default command (optional)
CMD ["bash"]