import datetime
import _thread
import ssl
import time
import os
import json
import sys
import wave
from queue import Queue

import websocket


class TTSClient:

    def __init__(self):
        self.audio = bytearray()
        self.audio_id = 0
        self.text = ''
        self.tts_queue = Queue()

        self.stream_ws = None

    def wav_to_pcm(self, wav_file):
        # 打开 WAV 文件
        with wave.open(wav_file, 'rb') as wav:
            # 提取参数
            params = wav.getparams()
            n_channels, sampwidth, framerate, n_frames = params[:4]

            # 读取帧数据
            pcm_data = wav.readframes(n_frames)
        return pcm_data

    def on_data(self, ws, message, msg_type, continue_flag):
        print(f'TTS on data begins')
        try:
            partial_id = 0
            num_bytes = 0
            if isinstance(message, str):
                msg = json.loads(message)
                #print(msg)
                if msg['status'] == 'ok':
                    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    if msg['type'] == 'partial':
                        # 写入同步队列，供其他函数获得
                        task_id = os.path.join('tmp', f'{self.audio_id}_partial_{cur_time}.wav')
                        partial_id += 1
                        self.audio_write(self.audio, task_id)
                    if msg['type'] == 'end':
                        # 写入同步队列，供其他函数获得
                        task_id = os.path.join('tmp', f'{self.audio_id}_end_{cur_time}.wav')
                        self.audio_id += 1
                        self.audio_write(self.audio, task_id)
                        self.tts_queue.put(task_id)

                        # pcm_bytes = self.wav_to_pcm(task_id)
                        # self.stream_ws.send_text('@@voice_start')
                        # self.stream_ws.send_bytes(pcm_bytes)
                        # self.stream_ws.send_text('@@voice_end')

                        self.ws.close()
                        self.audio = bytearray()
                        print(f'TTS recive end, num bytes: {num_bytes}')
                    if msg['type'] == 'phone':
                        pass
                else:
                    print('message status fail')
                    self.ws.close()
                    raise Exception("message status fail")
            else:
                num_bytes += 1
                self.audio.extend(message)
        except Exception as ex:
            print(f"ws exception {ex}")
            self.ws.close()
            raise Exception("ws exception", ex)

    def set_text(self, text):
        self.text = text

    def on_error(self, ws, error):
        print(error, file=sys.stderr, flush=True)
        self.ws.close()

    def on_open(self, ws):
        def run():
            message = json.dumps({'query': self.text, 'need_phone': False, 'ssml': False,
                                  'continuous_synthesis': False, 'language': 'zh-CN',
                                  'sample_rate': 16000, 'volume': 5,
                                  'speed_ratio': 1.0,
                                  'pitch': 0,
                                  'tradition': False, 'voice': 'xiaoyue', 'style': 'default',
                                  'audio_type': 'wav', 'need_subtitle': False,
                                  'need_polyphone': False, })
            self.ws.send(message.encode('utf-8'))
        _thread.start_new_thread(run, ())

    def on_close(self, ws, close_status_code, close_msg):
        print("### TTS close ###")
        self.ws.close()

    def audio_write(self, message, task_id):
        with open(task_id, 'ab') as fo:
            fo.write(message)

    def run(self):
        self.ws = websocket.WebSocketApp('wss://speech.sensetime.com/ttsV2',
                                         on_open=self.on_open,
                                         on_data=self.on_data,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
