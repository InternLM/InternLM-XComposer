import json
import time
import ffmpeg
import re
import wave
import os
import sys
import threading
import asyncio
import cv2

from typing import Callable, Any, List
from concurrent.futures import ThreadPoolExecutor


import logging

def log(name):
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)
    if not _logger.hasHandlers():
        formatter = logging.Formatter(
                "%(levelname)s: %(asctime)s %(module)s-%(funcName)s:%(lineno)d - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        _logger.addHandler(ch)
        _logger.propagate = False # 避免打印多次
    return _logger

logger = log("util")



class AudioStreamer(object):
    bitrate_pattern = r'Stream #0:\d+: Audio: .+, (\d+) kb/s'
    cancelled_flag = False

    def __init__(self, url: str, stop_event: threading.Event, *,
                 ac: int = 1, ar: int = 16000, aw: int = 16):
        self.url = url
        self.stop_event = stop_event
        self.stop_flag = False
        self.consumed_bytes_len = 0

        self.ac = ac
        self.ar = ar
        self.aw = aw
        self.bitrate = self.ac * self.ar * self.aw

        self.lock = threading.Lock()

    def cancell(self):
        self.cancelled_flag = True

    def stop(self):
        self.stop_flag = True

    def cancelled(self):
        while True:
            yield self.cancelled_flag

    def match_bitrate(self, string: str) -> int:
        """从字符串中提取比特率。

        返回：
        - int: 提取到的比特率，单位：kb/s
        """

        print(string)
        match = re.search(self.bitrate_pattern, string)

        if match:
            bitrate = int(match.group(1))  # 提取比特率并转换为整数
            return bitrate

        return None

    def calculate_duration(self, bytes_len: int = 0) -> float:
        """
        计算音频的时长。

        参数：
        - total_bytes: 音频数据的总字节数。
        - bitrate: 音频的比特率，单位：kb/s。
        """

        bytes_len = bytes_len if bytes_len else self.consumed_bytes_len

        return bytes_len / self.bitrate / 8

    def audio_stream(self, stream_path: str = None, webcam: bool = False, **kwargs):
        stream_path = stream_path if stream_path else self.url

        if webcam:
            return self.webcam_stream(stream_path)

        if "://" in stream_path:
            return self.rtmp_stream(stream_path, **kwargs)

        return self.file_stream(stream_path, **kwargs)

    def webcam_stream(self, device: str):
        '''
            使用如下命令列出可用设备：`ffmpeg -list_devices true -f dshow -i dummy`
        '''
        return self.rtmp_stream(device)

    def file_stream(self, file_path: str = None, *,
                    block_size: int = 4096, **kwargs):
        file_path = file_path if file_path else self.url
        block_size = block_size if block_size else 4096

        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            self.bitrate = sample_rate * sample_width * channels * 8

            while True:
                audio_data = wf.readframes(block_size)
                if not audio_data:
                    break

                # 记录读取字节数
                self.consumed_bytes_len += min(len(audio_data), block_size)

                yield audio_data

    def rtmp_stream(self, url: str = None, *,
                    format: str = 's16le', ac: int = 2, ar: str = '16000', acodec: str = 'pcm_s16le',
                    block_size: int = 4096, **kwargs):
        """
        从给定的URL获取音频流。

        使用FFmpeg工具从指定的URL读取音频数据，并将其以指定的格式输出到管道中。
        这个函数启动一个异步的FFmpeg进程，用于持续从URL获取音频流。

        参数:
        - url: 音频资源的URL。
        - format: 音频数据的格式，默认为's16le'，表示16位线性PCM。
        - ac: 音频通道数，默认为2，表示双声道。
        - ar: 音频采样率，默认为'16000'，表示16000Hz的采样率。
        - acodec: 音频编解码器，默认为'pcm_s16le'，表示16位线性PCM编解码器。
        - block_size: 读取音频数据的块大小，默认为4096字节。
        """
        url = url if url else self.url
        block_size = block_size if block_size else 4096

        ac = ac if ac else self.ac
        ar = ar if ar else str(self.ar)

        process = (ffmpeg
                   .input(url)
                   .output('pipe:', format=format, acodec=acodec, ac=ac, ar=ar)
                   .run_async(pipe_stdout=True, pipe_stderr=True))

        # 获取音频bitrate
        self.bitrate = 256000  # TODO
        count = 2
        # while True:
        #     output = process.stderr.readline().decode('utf-8')

        #     if count <= 0:
        #             break

        #     bitrate = self.match_bitrate(output)
        #     if bitrate:
        #         # for kbs
        #         self.bitrate = bitrate * 1000
        #         count = count - 1

        print(f"bitrate: {self.bitrate}")

        try:
            while True:
                with self.lock:
                    if self.stop_event.is_set() or self.stop_flag:
                        logger.info(
                            f'停止 audio stream: stop_event:{self.stop_event.is_set()}, stop_flag:{self.stop_flag}')
                        process.terminate()
                        break

                    audio_data = process.stdout.read(block_size)
                    last_print_time = time.time()
                    if time.time() - last_print_time > 5:
                        logger.info("audio stream读取到了数据")
                        last_print_time = time.time()

                    if not audio_data:
                        logger.info('audio stream 暂未读取到数据')
                        break

                    # 记录读取字节数
                    self.consumed_bytes_len += min(len(audio_data), block_size)

                    yield audio_data
        except Exception as e:
            logger.info(f'audio stream 异常:{str(e)}')
        finally:
            process.stdout.close()
            process.stderr.close()
            process.terminate()
            process.wait()
            logger.info('audio stream 资源清理完毕')

# audio_streamer = AudioStreamer("rtmp://10.1.100.37:1935/live/livestream")
# audio_stream = audio_streamer.audio_stream(block_size=4096, ac=1, webcam=False)
#
# import pdb
# pdb.set_trace()
# while True:
#     try:
#         data = next(audio_stream)
#     except StopIteration:
#         break
