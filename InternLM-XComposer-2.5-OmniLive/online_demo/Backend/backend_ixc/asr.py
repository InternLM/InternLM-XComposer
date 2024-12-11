import threading
import _thread
import argparse
import websocket
import datetime
import json
import wave
import time
import sys
import ssl
import os


class WebsocketClient():

    def __init__(self, host="127.0.0.1", port='10086', wav_path='', log_path='',
                 url: str='wss://speech.sensetime.com/asrv2/non-streaming',
                 millseconds: int=40, speech_pause_time: int=300, sample_rate: int=16000, audio_type: str='wav',
                 continuous_decoding: bool=True, server_vad: bool=True, punctuation: int=0, streaming: bool=True,
                 tradition: bool=False, pid: str='test', appid: str='test', devid: str='test', store: bool=True):
        self.host = host
        self.port = port
        self.wav_path = wav_path
        self.log_path = log_path
        self.file_name, _ = os.path.splitext(os.path.basename(self.wav_path))
        self.start = 0

        self.url = url

        self.millseconds = millseconds
        self.speech_pause_time = speech_pause_time
        self.sample_rate = sample_rate
        self.audio_type = audio_type
        self.continuous_decoding = continuous_decoding
        self.server_vad = server_vad
        self.punctuation = punctuation
        self.streaming = streaming
        self.tradition = tradition
        self.pid = pid
        self.appid = appid
        self.devid = devid
        self.store = store

        self.txt = ''

    def set_audio(self, audio_bytes):
        self.audio_bytes = audio_bytes
        self.txt = ''

    def file_write(self, now_time, msg):
        if self.log_path != '':
            with open(self.log_path, 'a') as file:
                # file.write(self.file_name)
                # file.write('\t')
                # file.write(now_time)
                # file.write('\t')
                file.write(msg)
                file.write('\n')

    def on_message(self, ws, message):
        msg = json.loads(message)
        # print(msg)
        if msg['status'] == 'ok':
            if msg['type'] == 'partial_result':
                result = msg['result']
                self.txt += result
                # print(result, file=sys.stderr, flush=True)
            if msg['type'] == 'final_result':
                # print(message)
                result = msg['result']
                # if self.start != 0:
                #     end = datetime.datetime.now()
                #     with open("time.log", 'a') as f:
                #         f.write(str((end - self.start).microseconds))
                #         f.write('\n')
                #     print((end - self.start).microseconds)
                #     self.start = 0
                #     print(end.strftime('%Y-%m-%d %H:%M:%S.%f'))
                # self.file_write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), result)
                # # print(result, file=sys.stderr, flush=True)
                self.txt += result
            if msg['type'] == 'speech_end':
                print(f'receive speech end: {self.txt}', file=sys.stderr, flush=True)
                self.ws.close()
        else:
            print(msg['message'], file=sys.stderr, flush=True)
            self.ws.close()

    def on_error(self, ws, error):
        print(error, file=sys.stderr, flush=True)
        self.ws.close()

    def on_close(self, ws, close_status_code, close_msg):
        print("### closed ###")
        self.ws.close()

    def on_open(self, ws):
        def run(*args):
            try:
                chunk = 2 * self.millseconds * self.sample_rate // 1000
                # print("chunk:", chunk)
                self.send_start_signal()
                self.send_data(self.audio_bytes)
                self.send_end_signal()
            except:
                print("Unexpected error:", sys.exc_info()[0])

        _thread.start_new_thread(run, ())

    def run(self):
        if self.url:
            url = self.url
        else:
            url = 'ws://' + self.host + ':' + self.port + '/'
        self.ws = websocket.WebSocketApp(url,
                                         {'Authorization': ''},
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def send_start_signal(self):
        message = json.dumps({'signal': 'start', 'continuous_decoding': self.continuous_decoding,
                              'speech_pause_time': self.speech_pause_time, 'server_vad': self.server_vad,
                              'punctuation': self.punctuation, 'tradition': self.tradition,
                              'product_id': self.pid, 'app_id': self.appid, 'device_id': self.devid,
                              'mic_volume': 0.8, 'store': self.store})
        self.ws.send(message)

    def send_end_signal(self):
        message = json.dumps({'signal': 'end'})
        self.ws.send(message)
        print('send end message')

    def send_data(self, binary):
        self.ws.send(binary, websocket.ABNF.OPCODE_BINARY)
