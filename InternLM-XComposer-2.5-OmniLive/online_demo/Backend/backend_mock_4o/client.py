import os
import re
import shutil
import sys
import json
import ssl
import time
import cv2
import base64
import threading
import websocket
from queue import Queue
from collections import deque
from asr import WebsocketClient
from funasr import AutoModel
import numpy as np
import wave
import asyncio
from audio_stream_consume import AudioStreamer
from video_stream_consume import scheduled_screenshot
from tts import TTSClient


def set_proxy(url):
    os.environ['http_proxy'] = url
    os.environ['https_proxy'] = url


height_threshold = 1024
def encode_image(image):
    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 判断长边是否超过1024像素
    if max(width, height) > height_threshold:
        # 计算缩放比例
        scale = height_threshold / max(width, height)

        # 根据缩放比例计算新的宽度和高度
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 调整图像尺寸
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        # 如果不需要调整尺寸，则直接使用原图
        resized_image = image

    # 将图像转换为字节流
    _, buffer = cv2.imencode('.jpg', resized_image)
    byte_image = buffer.tobytes()

    # Base64编码
    encoded_image = base64.b64encode(byte_image).decode('utf-8')

    return encoded_image


class Client():
    def __init__(self, session_id):
        self.proxy = "http://closeai-proxy.pjlab.org.cn:23128"
        self.other_proxy = "http://caoyuhang:68JQtChWMEr9jUcgHYIratL9SAK4z8VThcQxsj2vmw3vuJKsdykZGCQ9I71K@10.1.20.58:23128"
        self.gpt4_key = "sk-proj-3XRkdoBvywdui0M5ps8ST3BlbkFJ3fuo9olwPdArIxLkD8sP"

        self.ws = WebsocketClient(punctuation=1)

        self.stream_url = f'rtmp://srs-xcomposer-dev.intern-ai.org.cn:1935/live/doctest{session_id}'
        self.frame_list = deque(maxlen=10)
        self.stop_event = threading.Event()
        self.video_thread = threading.Thread(target=scheduled_screenshot,
                                             kwargs={'video_stream_url': self.stream_url,
                                                     'interval': 1, 'frame_list': self.frame_list,
                                                     'stop_event': self.stop_event})

        self.vad_model = AutoModel(model="fsmn-vad")

        self.audio_thread = threading.Thread(target=self.asr)

        self.task_list = {}
        self.asr_queue = Queue()

        self.history = []
        self.system_prompt = "You are a video conversation assistant who will receive up to three images corresponding to the most recent three seconds of video. Please engage in conversation based on this."
        self.chat_image_nums = 1
        self.interrupt = False

        set_proxy(self.other_proxy)
        from openai import OpenAI
        self.openaiclient = OpenAI(api_key=self.gpt4_key, base_url='https://api.openai.com/v1')
        set_proxy('')

        self.tts_thread = threading.Thread(target=self.tts)
        if os.path.exists('tmp'):
            shutil.rmtree('tmp')
        os.makedirs('tmp', exist_ok=True)
        self.tts_client = TTSClient()
        self.llm_out_queue = Queue()
        self.websocket = None

        new_loop = asyncio.new_event_loop()
        self.tts_pcm_thread = threading.Thread(target=self.process_audio_in_thread, args=(new_loop,))

    def tts(self):
        print(f'TTS Thread start')

        while not self.stop_event.is_set():
            time.sleep(0.3)
            if not self.llm_out_queue.empty():
                output = self.llm_out_queue.get()
                print(f'send tts text: {output}')
                self.tts_client.set_text(output)
                self.tts_client.run()

    def process_audio_in_thread(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.send_tts_pcm_bytes())

    async def send_tts_pcm_bytes(self):
        def wav_to_pcm(wav_file):
            # 打开 WAV 文件
            with wave.open(wav_file, 'rb') as wav:
                # 提取参数
                params = wav.getparams()
                n_channels, sampwidth, framerate, n_frames = params[:4]
                pcm_data = wav.readframes(n_frames)
            return pcm_data

        while not self.stop_event.is_set():
            if self.interrupt:
                await self.websocket.send_text('@@interrupt')
                print('send interrupt!!!!!')
                self.interrupt = False
                self.llm_out_queue.queue.clear()
                self.tts_client.tts_queue.queue.clear()

            tts_queue = self.tts_client.tts_queue
            try:
                # 尝试从队列中获取音频数据
                wav_file = tts_queue.get_nowait()
            except:
                # 如果队列为空，等待一小段时间再继续循环
                await asyncio.sleep(0.1)
                continue

            await self.websocket.send_text('@@voice_start')
            pcm_bytes = wav_to_pcm(wav_file)
            await self.websocket.send_bytes(pcm_bytes)
            await self.websocket.send_text('@@voice_end')


    def asr(self):
        audio_streamer = AudioStreamer(self.stream_url, stop_event=self.stop_event)
        audio_stream = audio_streamer.audio_stream(block_size=4096, ac=1, webcam=False)

        cache = {}
        start = False
        audios = []
        state = 'silence'
        silence_chuck = 0
        st = time.time()
        while not self.stop_event.is_set():
            try:
                audio = next(audio_stream)
            except StopIteration:
                break

            audio_array = np.frombuffer(audio, dtype=np.int16)
            #print(audio_array.max(), audio_array.min())
            audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max
            chunk_size = len(audio_array) / 16
            res = self.vad_model.generate(input=audio_array, cache=cache, is_final=False,
                                          chunk_size=chunk_size, disable_pbar=True)

            #print(res)
            if len(res[0]["value"]):
                if res[0]["value"][0][0] != -1:
                    state = 'voice'
                    start = False
                    self.interrupt = True

                if res[0]["value"][0][1] != -1:
                    state = 'silence'
                    start = True
                    st = time.time()

            if start and state == 'silence':
                silence_chuck += 1
                if silence_chuck > 3:  # 3 * 0.128s
                    if audios:
                        frame_truck = [self.frame_list[-self.chat_image_nums] for i in
                                       range(min(self.chat_image_nums, len(self.frame_list) - 1), 0, -1)]
                        slice_audio = b''.join(audios)
                        print('send slice audio')
                        self.ws.set_audio(slice_audio)
                        self.ws.run()
                        text = self.ws.txt
                        print(text)
                        if text:
                            self.asr_queue.put((text, frame_truck, st))
                    silence_chuck = 0
                    audios = []
                    start = False
                else:
                    audios.append(audio)

            if state == 'voice':
                audios.append(audio)
                silence_chuck = 0

    async def run(self, websocket):
        self.video_thread.start()
        self.audio_thread.start()
        self.tts_thread.start()
        self.tts_pcm_thread.start()
        self.tts_client.stream_ws = websocket
        self.websocket = websocket
        await asyncio.sleep(1)
        try:
            while not self.stop_event.is_set():
                await websocket.send_text('1')
                await asyncio.sleep(0.1)
                if not self.asr_queue.empty():
                    text, frame_truck, st_audio = self.asr_queue.get()
                    st = time.time()
                    base64_images = [encode_image(img) for img in frame_truck]
                    history = self.history
                    all_messages = [{"role": "system", "content": self.system_prompt}] + history + \
                                   [{"role": "user", "content": [{"type": "text", "text": text},
                                                                 *[{"type": "image_url", "image_url": {
                                                                     "url": f"data:image/jpeg;base64,{base64_image}"}}
                                                                   for base64_image in base64_images],
                                                                 ]},
                                    ]
                    print("send to gpt4o")
                    set_proxy(self.other_proxy)
                    response = self.openaiclient.chat.completions.create(
                        model="gpt-4o-2024-08-06",
                        messages=all_messages,
                        max_tokens=4096,
                        stream=True
                    )
                    set_proxy('')

                    latency = time.time() - st
                    all_latency = time.time() - st_audio

                    print(f"print response:{response}")
                    if response:
                        print("gpt4o finish")
                        output = ''
                        start_idx = 0

                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                if self.interrupt:
                                    break

                                output += chunk.choices[0].delta.content

                                match = re.search(r'[.\n。？！?!]', output[start_idx:])
                                if match:
                                    match_idx = start_idx + match.start()
                                    temp_text = output[start_idx:match_idx + 1]
                                    start_idx = match_idx + 1
                                    self.llm_out_queue.put(temp_text)
                                    print(f'put to: {temp_text}')

                                print(output)

                    print(output)

                    self.history.append({'role': 'user', 'content': text})
                    self.history.append({'role': 'assistant', 'content': output})

        except Exception as e:
            print(f'error is {type(e)}')

        self.stop_event.set()
