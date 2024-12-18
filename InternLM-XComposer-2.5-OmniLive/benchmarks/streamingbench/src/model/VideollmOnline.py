import os
import transformers

import sys
sys.path.append('/mnt/petrelfs/dingshuangrui/LLaVA/StreamingBench/videollm-online')
from demo.inference import LiveInfer
from data.utils import ffmpeg_once
import pdb
# LiveOnePlusTrainingArguments
logger = transformers.logging.get_logger('liveinfer')

from model.modelclass import Model
class VideollmOnline(Model):
    def __init__(self):
        """
        Initialize the model
        """
        super().__init__()
        self.liveinfer = LiveInfer()

    def Run(self, file, inp, timestamp):
        """
        Given the video file and input prompt, run the model and return the response
        file: Video file path
        inp: Input prompt
        timestampe: The time when question is asked
        """
        # timestamp = float(file.split('/')[-1].rsplit('.', 1)[0].split('_')[-1])
        
        return self.videollmOnline_Run(file, inp, timestamp)

    @staticmethod
    def name():
        """
        Return the name of the model
        """
        return "VideollmOnline"

    def videollmOnline_Run(self, file, inp, timestamp):
        self.liveinfer.reset()
        name, ext = os.path.splitext(file)
        name = name.split('/')[-1]
        ffmpeg_video_path = os.path.join('./cache', name + f'_{self.liveinfer.frame_fps}fps_{self.liveinfer.frame_resolution}' + ext)
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(file, ffmpeg_video_path, fps=self.liveinfer.frame_fps, resolution=self.liveinfer.frame_resolution)
        logger.warning(f'{file} -> {ffmpeg_video_path}, {self.liveinfer.frame_fps} FPS, {self.liveinfer.frame_resolution} Resolution')

        self.liveinfer.load_video(ffmpeg_video_path)
        self.liveinfer.input_query_stream(inp, video_time=timestamp)

        for i in range(self.liveinfer.num_video_frames):
            self.liveinfer.input_video_stream(i / self.liveinfer.frame_fps)
            query, response = self.liveinfer()

            if response:
                print(response)
                return response
