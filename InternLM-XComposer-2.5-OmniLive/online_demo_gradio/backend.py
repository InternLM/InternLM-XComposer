import base64
import io
import time
from PIL import Image
import os
import socket
import requests
import numpy as np
import shutil
import datetime
import librosa
import soundfile
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from queue import Queue
import threading

from swift.llm import get_model_tokenizer, get_template, ModelType, get_default_template_type, inference
from swift.utils import seed_everything


hostname = socket.gethostname()
ip_addr = socket.gethostbyname(hostname)
os.environ["no_proxy"] = f"localhost,127.0.0.1,::1,{ip_addr}"


class Client():
    def __init__(self):
        model_type = ModelType.qwen2_audio_7b_instruct
        model_id_or_path = '/mnt/hwfile/mllm/caoyuhang/models/sft_continue_silence'
        template_type = get_default_template_type(model_type)
        print(f'template_type: {template_type}')

        self.audio_model, audio_tokenizer = get_model_tokenizer(model_type, torch.float16,
                                                                model_id_or_path=model_id_or_path,
                                                                model_kwargs={'device_map': 'cuda:0'})
        self.audio_model.generation_config.max_new_tokens = 256
        self.audio_template = get_template(template_type, audio_tokenizer)
        seed_everything(42)

        tts_model = 'f5-tts'
        if tts_model == 'f5-tts':
            from cached_path import cached_path
            from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
            from f5_tts.model import DiT
            self.f5_vocoder = load_vocoder(vocoder_name='vocos', is_local=False,
                                           local_path="../checkpoints/vocos-mel-24khz")
            ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            self.f5_model = load_model(DiT, model_cfg, ckpt_file, mel_spec_type='vocos', vocab_file="")
            self.f5_infer_process = infer_process

            self.f5_ref_audio, self.f5_ref_text = preprocess_ref_audio_text("girl_01_ref.wav", "因为我的性格好像也是这种大大咧咧的,所以我对这种舞蹈,哎呀,我是特别热衷。")

        if os.path.exists('tmp'):
            shutil.rmtree('tmp')
        os.makedirs('tmp', exist_ok=True)

        self.llm_out_queue = Queue()

        self.audio_id = 0
        self.tts_finish_queue = Queue()

        self.tts = threading.Thread(target=self.tts_thread)
        self.tts.start()
        print('main init finish!!!')

    def tts_thread(self):
        while True:
            if self.llm_out_queue.empty():
                time.sleep(0.15)
            else:
                output = self.llm_out_queue.get()
                st_time = time.time()
                print(f'send tts text: {output}')
                cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                wav_file = os.path.join('tmp', f'{self.audio_id}_end_{cur_time}.wav')
                audio, final_sample_rate, spectragram = self.f5_infer_process(self.f5_ref_audio, self.f5_ref_text,
                                                                              output, self.f5_model, self.f5_vocoder,
                                                                              mel_spec_type='vocos', speed=1.0)
                audio_resampled = librosa.resample(audio, orig_sr=final_sample_rate, target_sr=16000)
                soundfile.write(wav_file, audio_resampled, 16000)
                self.audio_id += 1
                self.tts_finish_queue.put(wav_file)
                print("         tts:{:.2f}s".format(time.time() - st_time))

    async def transcribe_audio(self, audio, cls_audio):
        start_time = time.time()
        try:
            with open('tmp.wav', 'wb') as f:
                f.write(cls_audio)

            query = "<audio>Classify the audio."
            cls_label, _ = inference(self.audio_model, self.audio_template, query, audios='tmp.wav')
        except:
            print('issue....')
            cls_label = 'issue'

        with open('tmp.wav', 'wb') as f:
            f.write(audio)
        query = '<audio>Detect the language and recognize the speech.'
        text, _ = inference(self.audio_model, self.audio_template, query, audios='tmp.wav')
        #print(cls_label)
        if cls_label != 'Speech' and cls_label != 'speech' and cls_label != 'silence' and '声音' in text:
            chinese_count = sum('\u4e00' <= char <= '\u9fff' for char in text)
            if chinese_count >= len(text) // 2:
                if cls_label in self.en2ch:
                    text = f"'声音是{self.en2ch[cls_label]}。<##>{text}"
            else:
                text = f"'Sound classification：{cls_label}. <##>{text}"

        time_dict = {}
        time_dict['ast_t'] = time.time() - start_time
        print(text)
        if text:
            query = {'query': text, 'time_dict': time_dict}
            requests.post(f"http://{ip_addr}:8001/send_ixc", json=query)


my_client = Client()

# Initialize FastAPI app
app = FastAPI()


@app.post("/transcribe")
async def transcribe_audio(file_audio: UploadFile = File(...), file_cls: UploadFile = File(...)):
    audio_data = await file_audio.read()
    audio_data_buf = audio_data.decode("utf-8")
    audio_data = base64.b64decode(audio_data_buf)

    cls_data = await file_cls.read()
    cls_data_buf = cls_data.decode("utf-8")
    cls_data = base64.b64decode(cls_data_buf)
    await my_client.transcribe_audio(audio_data, cls_data)


@app.post("/recv_llm")
async def recv_llm(text: str):
    my_client.llm_out_queue.put(text)


@app.get("/get_audio")
async def get_audio():
    if not my_client.tts_finish_queue.empty():
        wav_file = my_client.tts_finish_queue.get()
        print(f'wav_file is {wav_file}')

        with open(wav_file, 'rb') as f:
            audio_data = f.read()
        base64_encoded = str(base64.b64encode(audio_data), encoding="utf-8")
        return base64_encoded
    else:
        return ''



if __name__ == "__main__":
    uvicorn.run(app, host=ip_addr, port=8000)

#uvicorn backend:app --reload --port 8000 --host 10.140.0.242 --loop asyncio
