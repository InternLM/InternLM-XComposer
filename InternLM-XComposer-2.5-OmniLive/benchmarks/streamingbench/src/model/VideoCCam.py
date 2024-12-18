from eval import load_decord
from model_ccam import create_videoccam

mllm  = None

from model.modelclass import Model
class VideoCCam(Model):
    def __init__(self):
        VideoCCam_Init()

    def Run(self, file, inp):
        return VideoCCam_Run(file, inp)
    
    def name(self):
        return "VideoCCam"

def VideoCCam_Init():
    global mllm
    mllm = create_videoccam(
        model_name='Video-CCAM-14B',
        model_path='/yeesuanAI05/thumt/wzh/huggingface_cache/video-ccam-14b',
        llm_name_or_path='/yeesuanAI05/thumt/wzh/huggingface_cache/phi3',                   # automatically download by default
        visual_encoder_name_or_path='/yeesuanAI05/thumt/wzh/huggingface_cache/siglip',     # automatically download by default
        torch_dtype='bfloat16'
    )

def VideoCCam_Run(file, inp):
    video_path = file
    question = '<video>\n' + inp

    sample_config = dict(
        sample_type='uniform',
        num_frames=32
    )

    frames = load_decord(video_path, **sample_config)
    response = mllm.generate(texts=[question], videos=[frames])[0]

    print(response)

    return response