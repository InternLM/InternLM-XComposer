import base64
import socket
import time
import io
import argparse
from PIL import Image
from queue import Queue
import gradio as gr
import requests
import wave

hostname = socket.gethostname()

import numpy as np
from funasr import AutoModel

import cv2
import pyaudio


parser = argparse.ArgumentParser()
parser.add_argument("--audio_source", type=str, default='gradio', choices=['local', 'gradio'])
parser.add_argument("--video_source", type=str, default='local', choices=['local', 'gradio'])
parser.add_argument("--backend_ip", type=str, default='10.140.1.126')
args = parser.parse_args()


class Client():
    def __init__(self, backend_ip):
        self.backend_ip = backend_ip
        self.vad_model = AutoModel(model="fsmn-vad")

        self.raw_audios = Queue()
        self.audio_queue = Queue()
        self.audio_time = time.time()
        self.video_time = time.time()

    def streaming_audio(self, audio_file):
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        self.raw_audios.put(audio_data)

    def append_wav_info(self, bytes_audio, p, rate):
        wf = wave.open('tmp.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(bytes_audio)
        wf.close()
        with open('tmp.wav', 'rb') as f:
            new_bytes_audio = f.read()

        return new_bytes_audio

    def streaming_local_audio(self):
        p = pyaudio.PyAudio()
        rate = 16000
        chunk_millseconds = 200
        chunk = chunk_millseconds * rate // 1000
        # 打开音频流
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=True,
                        input_device_index=1,
                        frames_per_buffer=chunk)

        print("开始录音...")

        cache = {}
        start = False  # start to save salience chunck
        audios = []
        state = 'silence'
        silence_chuck = 0
        pre_chunk = None
        cls_chunk = []
        cls_chunk_size = 30
        start_cls = True  # start to save chunk for classification
        count = 0
        while True:
            bytes_audio = stream.read(chunk)
            audio_array = np.frombuffer(bytes_audio, dtype=np.int16)

            # print(count, audio_array.max(), audio_array.min())
            # count += 1
            # print(f'audio time is : {time.time() - self.audio_time}')
            # self.audio_time = time.time()

            audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max
            res = self.vad_model.generate(input=audio_array, cache=cache, is_final=False,
                                        chunk_size=chunk_millseconds, disable_pbar=True)

            if len(res[0]["value"]):
                if res[0]["value"][0][0] != -1:
                    state = 'voice'
                    start = False
                    start_cls = False
                    if pre_chunk:
                        audios = [pre_chunk]

                if res[0]["value"][0][1] != -1:
                    state = 'silence'
                    start = True

            if start and state == 'silence':
                silence_chuck += 1
                if silence_chuck > 1:  # 1 * 0.128s
                    if audios:
                        slice_audio = b''.join(audios)
                        cls_audio = b''.join(cls_chunk[:-1])
                        print('send slice audio')
                        slice_audio = self.append_wav_info(slice_audio, p, rate)
                        cls_audio = self.append_wav_info(cls_audio, p, rate)
                        self.audio_queue.put((slice_audio, cls_audio))
                    silence_chuck = 0
                    audios = []
                    cls_chunk = []
                    start = False
                    start_cls = True
                else:
                    audios.append(bytes_audio)

            if state == 'voice':
                audios.append(bytes_audio)
                silence_chuck = 0

            pre_chunk = bytes_audio
            if start_cls:
                cls_chunk.append(bytes_audio)
                if len(cls_chunk) > cls_chunk_size + 1:
                    cls_chunk = cls_chunk[1:]

    def vad_consume(self):
        cache = {}
        start = False  # start to save salience chunck
        audios = []
        state = 'silence'
        silence_chuck = 0
        pre_chunk = None
        cls_chunk = []
        cls_chunk_size = 30
        start_cls = True  # start to save chunk for classification
        count = 0
        while True:
            if self.raw_audios.empty():
                time.sleep(0.15)
            else:
                bytes_audio = self.raw_audios.get()
                audio_array = np.frombuffer(bytes_audio, dtype=np.int16)
                #print(count, audio_array.max(), audio_array.min())
                count += 1
                #print(f'audio time is : {time.time() - self.audio_time}')
                self.audio_time = time.time()

                audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max
                #print(audio_array.shape)
                chunk_size = len(audio_array) / 16
                res = self.vad_model.generate(input=audio_array, cache=cache, is_final=False,
                                              chunk_size=chunk_size, disable_pbar=True)

                #print(res)
                if len(res[0]["value"]):
                    if res[0]["value"][0][0] != -1:
                        state = 'voice'
                        start = False
                        start_cls = False
                        if pre_chunk:
                            audios = [pre_chunk]

                    if res[0]["value"][0][1] != -1:
                        state = 'silence'
                        start = True

                if start and state == 'silence':
                    silence_chuck += 1
                    if silence_chuck > 3:  # 1 * 0.128s
                        if audios:
                            slice_audio = b''.join(audios)
                            cls_audio = b''.join(cls_chunk[:-1])
                            print('send slice audio')
                            self.audio_queue.put((slice_audio, cls_audio))
                        silence_chuck = 0
                        audios = []
                        cls_chunk = []
                        start = False
                        start_cls = True
                    else:
                        audios.append(bytes_audio)

                if state == 'voice':
                    audios.append(bytes_audio)
                    silence_chuck = 0

                pre_chunk = bytes_audio
                if start_cls:
                    cls_chunk.append(bytes_audio)
                    if len(cls_chunk) > cls_chunk_size + 1:
                        cls_chunk = cls_chunk[1:]

    def send_audio(self):
        while True:
            if self.audio_queue.empty():
                time.sleep(0.15)
            else:
                slice_audio, cls_audio = self.audio_queue.get()

                base64_audio = str(base64.b64encode(slice_audio), encoding="utf-8")
                base64_cls = str(base64.b64encode(cls_audio), encoding="utf-8")
                files = {
                    'file_audio': ('audio1.wav', base64_audio, 'audio/wav'),
                    'file_cls': ('audio2.wav', base64_cls, 'audio/wav'),
                }
                response = requests.post(f"http://{self.backend_ip}:8000/transcribe", files=files)
                print(response)

    def recv_audio(self):
        while True:
            response = requests.get(f"http://{self.backend_ip}:8000/get_audio", timeout=600)
            audio = eval(response.content.decode('utf-8'))
            if response.status_code == 200 and audio:
                audio_data = base64.b64decode(audio)
                print('recv audio.')
                yield audio_data
            else:
                with open('silence.wav', 'rb') as f:
                    audio_data = f.read()
                yield audio_data


    def send_video(self, img_file):
        print(f'video time is : {time.time() - self.video_time}')
        self.video_time = time.time()

        with open(img_file, 'rb') as f:
            img_data = f.read()

        base64_encoded = str(base64.b64encode(img_data), encoding="utf-8")

        t = time.time()
        files = {
            'file': ('img.jpg', base64_encoded, 'img/jpg'),
        }
        requests.post(f"http://{self.backend_ip}:8002/send_vs", files=files)


    def send_local_video(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Can not open camera")
            exit()

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS of the camera: {fps}")

        current_frame = 0
        while True:
            if current_frame % fps != 0:
                ret = cap.grab()

                if not ret:
                    print(f"scheduled_screenshot read frame failed, current_frame: {current_frame}, fps: {fps}")
                    break
                current_frame += 1
                continue

            # print(f'{current_frame} video time is : {time.time() - self.video_time}')
            # self.video_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print(f"scheduled_screenshot read frame failed, current_frame: {current_frame}, fps: {fps}")
                break

            current_frame += 1

            is_success, im_buf_arr = cv2.imencode(".jpg", frame)
            byte_im = im_buf_arr.tobytes()
            base64_encoded = str(base64.b64encode(byte_im), encoding="utf-8")

            files = {
                'file': ('img.jpg', base64_encoded, 'img/jpg'),
            }
            requests.post(f"http://{self.backend_ip}:8002/send_vs", files=files)

            frame = frame[:, :, ::-1]
            yield frame


my_client = Client(args.backend_ip)


def launch_demo():
    with gr.Blocks(title="IXC-2.5-Live") as demo:
        gr.Markdown("""<center><font size=8> IXC-2.5-Live </center>""")

        if args.video_source == 'gradio':
            input_img = gr.Image(label="Input", sources="webcam", type='filepath')
        else:
            input_img = gr.Image(label="Input", streaming=True, interactive=False)

        with gr.Column():
            audio_out = gr.Audio(streaming=True, autoplay=True, interactive=False)

        with gr.Row():
            pushvideo_btn = gr.Button('Push Video')
            if args.audio_source == 'gradio':
                audio_in = gr.Audio(sources=["microphone"], type='filepath', streaming=True, label="🎤 Record Audio (录音)")

        if args.video_source == 'gradio':
            input_img.stream(my_client.send_video, [input_img], [], stream_every=1, time_limit=2)
        else:
            pushvideo_btn.click(my_client.send_local_video, [], [input_img])

        if args.audio_source == 'gradio':
            audio_in.stream(my_client.streaming_audio, [audio_in], [], stream_every=0.4, time_limit=2)
            audio_in.start_recording(my_client.send_audio, [], [])
            audio_in.start_recording(my_client.recv_audio, [], [audio_out])
            audio_in.start_recording(my_client.vad_consume, [], [])
        else:
            pushvideo_btn.click(my_client.streaming_local_audio, [], [])
            pushvideo_btn.click(my_client.send_audio, [], [])
            pushvideo_btn.click(my_client.recv_audio, [], [audio_out])

        demo.launch()


if __name__ == '__main__':

    launch_demo()


