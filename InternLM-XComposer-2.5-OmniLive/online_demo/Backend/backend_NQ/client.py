import os
import time
import threading
from funasr import AutoModel
import numpy as np
import wave
from queue import Queue
import asyncio
from audio_stream_consume import AudioStreamer


class Client():
    def __init__(self):
        self.finished_closed = True

    def initiate(self, session_id):
        self.vad_model = AutoModel(model="fsmn-vad")
        self.stream_url = f'rtmp://srs-xcomposer-dev.intern-ai.org.cn:1935/live/doctest{session_id}'
        self.stop_event = threading.Event()
        self.audio_thread = threading.Thread(target=self.asr)
        self.asr_queue = Queue()

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
            audio_array = (audio_array * 0.2).astype(np.int16)
            audio = audio_array.tobytes()
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

                if res[0]["value"][0][1] != -1:
                    state = 'silence'
                    start = True
                    st = time.time()

            if start and state == 'silence':
                silence_chuck += 1
                if silence_chuck > 3:  # 3 * 0.128s
                    if audios:
                        slice_audio = b''.join(audios)
                        print('send slice audio')
                        self.asr_queue.put(slice_audio)
                    silence_chuck = 0
                    audios = []
                    start = False
                else:
                    audios.append(audio)

            if state == 'voice':
                audios.append(audio)
                silence_chuck = 0

    async def run(self, websocket, session_id):
        self.audio_thread.start()
        folder = f'NQ_audio/{session_id}'
        os.makedirs(folder, exist_ok=True)
        await websocket.send_text("@@socket_ready")

        cnt = 0
        i = 0
        while not self.stop_event.is_set():
            if i % 100 == 0:
                await websocket.send_text('ping......')
            i += 1
            await asyncio.sleep(0.1)
            if not self.asr_queue.empty():
                audio = self.asr_queue.get()
                pass
