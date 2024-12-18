import io
import mmengine
import os
import re
import shutil
import sys
sys.path.insert(0, 'third_party')
import time
import cv2
import base64
import threading
import datetime
from PIL import Image, ImageDraw, ImageFont
from queue import Queue
import multiprocessing
from multiprocessing import Value, Manager
from multiprocessing import Queue as MPQueue
from funasr import AutoModel as FunAutoModel
import numpy as np
import wave
import asyncio
from audio_stream_consume import AudioStreamer
from video_stream_consume import scheduled_screenshot
import opencc
import whisper
import soundfile
import librosa
from scipy.io import wavfile
import torchvision.transforms as transforms

import torch
import nest_asyncio
nest_asyncio.apply()
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

from swift.llm import get_model_tokenizer, get_template, ModelType, get_default_template_type, inference
from swift.utils import seed_everything

from vs_deploy.video_streaming import Model as VSModel
from third_party.melo.api import TTS


def set_proxy(url):
    os.environ['http_proxy'] = url
    os.environ['https_proxy'] = url


def img_process(imgs):
    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w/h
        if w > h:
            new_w = 560 * 2
            new_h = int(560 * 2 / scale)
        else:
            new_w = int(560 * 2 * scale)
            new_h = 560 * 2
        img = transforms.functional.resize(img, [new_h, new_w],)
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w,h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        font = ImageFont.truetype(f"{os.getenv('ROOT_DIR')}/SimHei.ttf", pad)
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w,h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h ), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(0, pad +curr_h + h +5), (new_w, pad +curr_h + h +5)], fill = 'black', width=2)
            curr_h += h + 10 + pad
        #print (new_w, new_h)
    else:
        for im in imgs:
            w,h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        font = ImageFont.truetype(f"{os.getenv('ROOT_DIR')}/SimHei.ttf", pad)
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w,h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill = 'black', width=2)
            curr_w += w + 10
    return new_img


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


def model_consume(finish_model, interrupt, mm_querys, llm_out_queue, tp):
    debug = os.getenv('DEBUG', False)
    if not debug:
        hf_model = f'{os.getenv("ROOT_DIR")}/merge_lora'
        if tp > 1:
            backend_config = TurbomindEngineConfig(tp=tp)
            pipe = pipeline(hf_model, backend_config=backend_config)
        else:
            pipe = pipeline(hf_model)
            pipe.chat_template.meta_instruction = """你是一个多模态人工智能助手，名字叫浦语·灵笔。
- 浦语·灵笔是由上海人工智能实验室开发的一个多模态对话模型，是一个有用，真实且无害的模型。
- 浦语·灵笔可以根据看到和听到的内容，流利的同用户进行交流，并使用用户使用的语言（中文或英文）进行回复。
"""
        gen_config = GenerationConfig(top_k=50, top_p=0.8, temperature=0.1)
    else:
        pipe = None

    finish_model.value = 1

    while True:
        time.sleep(0.1)
        if not mm_querys.empty():
            query, imgs, time_dict = mm_querys.get()
            time_dict['mm_querys_get'] = time.time()

            inst_yes = False
            query_nopunc = query[:-1] if query[-1] in ['?', '？'] else query
            mid_prompt = f'''Is the following string an instruction? "{query_nopunc}"'''
            if debug:
                inst_resp = False
                llm_out_queue.put(query)
            else:
                inst_resp = pipe.stream_infer((mid_prompt, []), gen_config=gen_config)
            if inst_resp:
                for chunk in inst_resp:
                    if chunk.text:
                        if "Yes" in chunk.text:
                            inst_yes = True
                            break

            time_dict['instruct_finish'] = time.time()
            print(f"model instruct output:  ========={inst_yes}=========")

            if inst_yes:
                if len(imgs) > 0:
                    #img = frame2img(imgs, get_font())
                    img = img_process(imgs)
                else:
                    img = Image.new('RGB', (100, 100), (0, 0, 0))

                img = [img]

                #print(f'send to model:{query}')
                response = pipe.stream_infer((query, img), gen_config=gen_config)
                if response:
                    output = ''
                    start_idx = 0

                    for chunk in response:
                        print(chunk)
                        if chunk.text:
                            if interrupt.value:
                                break

                            output += chunk.text

                            match = re.search(r'[.\n。？！?!]', output[start_idx:])
                            if match:
                                match_idx = start_idx + match.start()
                                temp_text = output[start_idx:match_idx + 1]
                                start_idx = match_idx + 1
                                if 'llm_first_chunck' not in time_dict:
                                    time_dict['llm_first_chunck'] = time.time()
                                llm_out_queue.put((temp_text, time_dict))
                    if start_idx < len(output):
                        llm_out_queue.put((output[start_idx:], time_dict))
                    print(f"model output: {output}")

                    if 'llm_first_chunck' not in time_dict:
                        time_dict['llm_first_chunck'] = 0

                    print(f"==============time start==================")
                    print('''
                    voice_end2put:{:.2f}s
                    voice_queue:{:.2f}s
                    asr:{:.2f}s
                    text_queue:{:.2f}s
                    mem:{:.2f}s
                    mem_query:{:.2f}s
                    instruct:{:.2f}s
                    llm_1tr:{:.2f}s
                    all:{:.2f}s'''.format(time_dict['voice_put_queue'] - time_dict['voice_end'],
                                                 time_dict['voice_get_queue'] - time_dict['voice_put_queue'],
                                                 time_dict['text_put_queue'] - time_dict['voice_get_queue'],
                                                 time_dict['text_get_queue'] - time_dict['text_put_queue'],
                                                 time_dict['mem_finish'] - time_dict['text_get_queue'],
                                                 time_dict['mm_querys_get'] - time_dict['mem_finish'],
                                                 time_dict['instruct_finish'] - time_dict['mm_querys_get'],
                                                 time_dict['llm_first_chunck'] - time_dict['instruct_finish'],
                                                 time_dict['llm_first_chunck'] - time_dict['voice_put_queue']))
                    print(f"==============time end==================")


def vsmodel_consume(frame_list, vs_dict):
    vsmodel = VSModel(f'{os.getenv("ROOT_DIR")}/memory')
    while True:
        print('init vsmodel sess!!!!!!!!!!!')
        vsmodel.init_sess(frame_list)
        vs_dict['stream_open'] = False
        vs_dict['query'] = ''
        vs_dict['query_finish'] = False
        vs_dict['break'] = False
        vs_dict['backup'] = False

        vsmodel.process(vs_dict)


class Client():
    def __init__(self, asr_model, tts_model, tp):
        self.en2ch = mmengine.load(f'sound_en2cn.json')
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.tp = tp

        if asr_model == 'whisper_large-v2':
            self.text_converter = opencc.OpenCC('t2s')
            self.whisper_model = whisper.load_model('large-v2')
        elif asr_model == 'streaming_audio':
            model_type = ModelType.qwen2_audio_7b_instruct
            model_id_or_path = f'{os.getenv("ROOT_DIR")}/audio'
            template_type = get_default_template_type(model_type)
            print(f'template_type: {template_type}')

            self.audio_model, audio_tokenizer = get_model_tokenizer(model_type, torch.float16,
                                                            model_id_or_path=model_id_or_path, model_kwargs={'device_map': 'cuda:0'})
            self.audio_model.generation_config.max_new_tokens = 256
            self.audio_template = get_template(template_type, audio_tokenizer)
            seed_everything(42)

        if tts_model == 'meloTTS':
            self.t2s_model = TTS(language="ZH", device="auto")
            self.speaker_ids = self.t2s_model.hps.data.spk2id
            # warm up
            self.t2s_model.tts_to_file('123', self.speaker_ids['ZH'], None, speed=1.0, quiet=True)
        elif tts_model == 'f5-tts':
            from cached_path import cached_path
            from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text, remove_silence_for_generated_wav
            from f5_tts.model import DiT
            self.f5_vocoder = load_vocoder(vocoder_name='vocos', is_local=False, local_path="../checkpoints/vocos-mel-24khz")
            ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            self.f5_model = load_model(DiT, model_cfg, ckpt_file, mel_spec_type='vocos', vocab_file="")
            self.f5_infer_process = infer_process

            #self.f5_ref_audio, self.f5_ref_text = preprocess_ref_audio_text("./third_party/basic_ref_en.wav", "Some call me nature, others call me Mother Nature.")
            self.f5_ref_audio, self.f5_ref_text = preprocess_ref_audio_text("./third_party/girl_01_ref.wav", "因为我的性格好像也是这种大大咧咧的,所以我对这种舞蹈,哎呀,我是特别热衷。")


        self.vad_model = FunAutoModel(model="fsmn-vad")

        finish_model = Value('i', 0)
        self.interrupt = Value('i', 0)
        self.mm_querys = MPQueue()
        self.llm_out_queue = MPQueue()
        model_process = multiprocessing.Process(target=model_consume, args=(finish_model, self.interrupt, self.mm_querys, self.llm_out_queue, self.tp))
        model_process.start()

        self.frame_list = MPQueue()
        self.vs_dict = Manager().dict()
        vs_process = multiprocessing.Process(target=vsmodel_consume, args=(self.frame_list, self.vs_dict))
        vs_process.start()

        self.finished_closed = True
        while not finish_model.value:
            time.sleep(0.1)

    def initiate(self, session_id):
        # self.stream_url = f'rtmp://srs-xcomposer-dev.intern-ai.org.cn:1935/live/doctest{session_id}'
        self.stream_url = f'rtmp://10.1.101.102:1935/live/livestream{session_id}'
        self.stop_event = threading.Event()
        self.video_thread = threading.Thread(target=scheduled_screenshot,
                                             kwargs={'video_stream_url': self.stream_url,
                                                     'interval': 1, 'frame_list': self.frame_list,
                                                     'stop_event': self.stop_event})

        self.task_list = {}
        self.audio_queue = Queue()
        self.asr_queue = Queue()

        self.history = []

        while not self.llm_out_queue.empty():
            self.llm_out_queue.get()
        while not self.mm_querys.empty():
            self.mm_querys.get()
        self.websocket = None

        self.audio_thread = threading.Thread(target=self.audio_vad)
        self.asr_thread = threading.Thread(target=self.asr)

        if os.path.exists('tmp'):
            shutil.rmtree('tmp')
        os.makedirs('tmp', exist_ok=True)
        new_loop = asyncio.new_event_loop()
        self.tts_pcm_thread = threading.Thread(target=self.process_audio_in_thread, args=(new_loop,))
        if self.tts_model == 'meloTTS' or self.tts_model == 'f5-tts':
            self.audio_id = 0


    def tts(self):
        print(f'TTS Thread start')

        while not self.stop_event.is_set():
            time.sleep(0.02)
            if not self.llm_out_queue.empty():
                output, time_dict = self.llm_out_queue.get()
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
            if self.interrupt.value:
                await self.websocket.send_text('@@interrupt')
                print('send interrupt!!!!!')
                self.interrupt.value = 0
                while not self.llm_out_queue.empty():
                    self.llm_out_queue.get()

            if self.tts_model == 'meloTTS':
                if not self.llm_out_queue.empty():
                    output, time_dict = self.llm_out_queue.get()
                    st_time = time.time()
                    print(f'send tts text: {output}')
                    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    wav_file = os.path.join('tmp', f'{self.audio_id}_end_{cur_time}.wav')
                    audio = self.t2s_model.tts_to_file(output, self.speaker_ids['ZH'], None, speed=1.0, quiet=True)
                    audio = audio * 10
                    audio_resampled = librosa.resample(audio, orig_sr=self.t2s_model.hps.data.sampling_rate, target_sr=16000)
                    soundfile.write(wav_file, audio_resampled, 16000)
                    self.audio_id += 1
                    print("         tts:{:.2f}s".format(time.time() - st_time))
                else:
                    await asyncio.sleep(0.02)
                    continue
            elif self.tts_model == 'f5-tts':
                if not self.llm_out_queue.empty():
                    output, time_dict = self.llm_out_queue.get()
                    st_time = time.time()
                    print(f'send tts text: {output}')
                    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    wav_file = os.path.join('tmp', f'{self.audio_id}_end_{cur_time}.wav')
                    audio, final_sample_rate, spectragram = self.f5_infer_process(self.f5_ref_audio, self.f5_ref_text,
                                                                  output, self.f5_model, self.f5_vocoder, mel_spec_type='vocos', speed=1.0)
                    audio_resampled = librosa.resample(audio, orig_sr=final_sample_rate, target_sr=16000)
                    soundfile.write(wav_file, audio_resampled, 16000)
                    self.audio_id += 1
                    print("         tts:{:.2f}s".format(time.time() - st_time))
                else:
                    await asyncio.sleep(0.02)
                    continue

            await self.websocket.send_text('@@voice_start')
            pcm_bytes = wav_to_pcm(wav_file)
            await self.websocket.send_bytes(pcm_bytes)
            await self.websocket.send_text('@@voice_end')

    def audio_vad(self):
        audio_streamer = AudioStreamer(self.stream_url, stop_event=self.stop_event)
        audio_stream = audio_streamer.audio_stream(block_size=4096, ac=1, webcam=False)

        cache = {}
        start = False  # start to save salience chunck
        audios = []
        state = 'silence'
        silence_chuck = 0
        pre_chunk = None
        st = time.time()
        count = 0

        cls_chunk = []
        cls_chunk_size = 30
        start_cls = True  # start to save chunk for classification
        while not self.stop_event.is_set():
            try:
                audio = next(audio_stream)
            except StopIteration:
                break
            audio_array = np.frombuffer(audio, dtype=np.int16)
            audio_array = (audio_array * 0.2).astype(np.int16)
            #print(count, audio_array.max(), audio_array.min())
            count += 1
            audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max
            chunk_size = len(audio_array) / 16
            res = self.vad_model.generate(input=audio_array, cache=cache, is_final=False,
                                          chunk_size=chunk_size, disable_pbar=True)

            # print(res)
            if len(res[0]["value"]):
                if res[0]["value"][0][0] != -1:
                    state = 'voice'
                    start = False
                    start_cls = False
                    self.interrupt.value = 1
                    if pre_chunk:
                        audios = [pre_chunk]

                if res[0]["value"][0][1] != -1:
                    state = 'silence'
                    start = True
                    time_dict = {'voice_end':time.time()}

            if start and state == 'silence':
                silence_chuck += 1
                if silence_chuck > 1:  # 1 * 0.128s
                    if audios:
                        self.vs_dict['backup'] = True  # backup for search mem
                        slice_audio = b''.join(audios)
                        cls_audio = b''.join(cls_chunk[:-1])
                        print('send slice audio')
                        time_dict['voice_put_queue'] = time.time()
                        self.audio_queue.put((slice_audio, cls_audio, time_dict))
                    silence_chuck = 0
                    audios = []
                    cls_chunk = []
                    start = False
                    start_cls = True
                else:
                    audios.append(audio)

            if state == 'voice':
                audios.append(audio)
                silence_chuck = 0

            pre_chunk = audio
            if start_cls:
                cls_chunk.append(audio)
                if len(cls_chunk) > cls_chunk_size + 1:
                    cls_chunk = cls_chunk[1:]


    def asr(self):
        while not self.stop_event.is_set():
            if not self.audio_queue.empty():
                audio, cls_audio, time_dict = self.audio_queue.get()
                time_dict['voice_get_queue'] = time.time()
                if self.asr_model == 'whisper_large-v2':
                    audio_array = np.frombuffer(audio, dtype=np.int16).flatten().astype(np.float32) / 32768.0
                    text = self.text_converter.convert(self.whisper_model.transcribe(audio_array)['text'])
                elif self.asr_model == 'streaming_audio':
                    query = "<audio>Classify the audio."
                    audio_array = np.frombuffer(cls_audio, dtype=np.int16).flatten().astype(np.float32) / 32768.0
                    wav_io = io.BytesIO()
                    wavfile.write(wav_io, 16000, audio_array)
                    try:
                        cls_label, _ = inference(self.audio_model, self.audio_template, query, audios=wav_io)
                    except:
                        print('issue....')
                        print(audio_array.shape, len(cls_audio))
                    wav_io.close()
                    wavfile.write('cls_tmp.wav', 16000, audio_array)

                    query = '<audio>Detect the language and recognize the speech.'
                    audio_array = np.frombuffer(audio, dtype=np.int16).flatten().astype(np.float32) / 32768.0
                    audio_array = audio_array[:-4096]
                    wav_io = io.BytesIO()
                    wavfile.write(wav_io, 16000, audio_array)
                    text, _ = inference(self.audio_model, self.audio_template, query, audios=wav_io)
                    wav_io.close()
                    print(cls_label)
                    wavfile.write('speech_tmp.wav', 16000, audio_array)
                    if cls_label != 'Speech' and cls_label != 'speech' and cls_label != 'silence':
                        chinese_count = sum('\u4e00' <= char <= '\u9fff' for char in text)
                        if chinese_count >= len(text) // 2:
                            if cls_label in self.en2ch:
                                text = f"'声音是{self.en2ch[cls_label]}。{text}"
                        else:
                            text = f"'Sound classification：{cls_label}. {text}"

                time_dict['text_put_queue'] = time.time()
                print(text)
                if text:
                    self.asr_queue.put((text, time_dict))
            else:
                time.sleep(0.02)

    async def run(self, websocket):
        self.video_thread.start()
        self.vs_dict['stream_open'] = False
        while not self.vs_dict['stream_open']:
            await asyncio.sleep(1)
            await websocket.send_text('waiting for stream open.')
            print('waiting for stream open.')

        self.audio_thread.start()
        self.asr_thread.start()

        self.tts_pcm_thread.start()

        self.websocket = websocket

        mem_folder = f"tmp/mem"
        if os.path.exists(mem_folder):
            shutil.rmtree(mem_folder)
        os.makedirs(mem_folder)

        await websocket.send_text("@@socket_ready")
        t = 0
        while not self.stop_event.is_set():
            if t % 50 == 0:
                await websocket.send_text('ping...')
                print('ping...')
            await asyncio.sleep(0.1)
            t += 1
            if not self.asr_queue.empty():
                text, time_dict = self.asr_queue.get()
                time_dict['text_get_queue'] = time.time()

                self.vs_dict['query_finish'] = False
                self.vs_dict['query'] = text
                while not self.vs_dict['query_finish']:
                    await asyncio.sleep(0.02)

                time_dict['mem_finish'] = time.time()

                imgs = torch.load(os.path.join(mem_folder, 'temp_lol_img.pth'))
                #imgs = imgs[::2]
                imgs = imgs[-3:][::2]

                self.mm_querys.put((text, imgs, time_dict))


