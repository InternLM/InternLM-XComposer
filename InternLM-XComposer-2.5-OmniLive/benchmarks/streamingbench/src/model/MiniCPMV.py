import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model

from model.modelclass import Model
class MiniCPMV(Model):
    def __init__(self, model_path='openbmb/MiniCPM-V-2_6'):
        """
        Initialize the model by loading the pretrained MiniCPM-V model and tokenizer.
        """

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.MAX_NUM_FRAMES = 64  # Maximum number of frames to process

    def encode_video(self, video_path):
        """
        Encode the video frames from the video path.
        """
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > self.MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, self.MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        # print('Number of frames:', len(frames))
        return frames

    def Run(self, file, inp):
        """
        Given the file (video file path) and input prompt (inp), run the model and return the response.
        """
        frames = self.encode_video(file)
        msgs = [
            {'role': 'user', 'content': frames + [inp]},
        ]

        # Set decode parameters for video
        params = {
            "use_image_id": False,
            "max_slice_nums": 1  # Adjust if CUDA OOM and video resolution > 448x448
        }

        # Generate the response using the model
        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        print(answer)
        return answer

    @staticmethod
    def name():
        """
        Return the name of the model
        """
        return "MiniCPM-V"
