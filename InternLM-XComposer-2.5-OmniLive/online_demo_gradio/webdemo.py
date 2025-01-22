import sys
import os
import socket

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
os.environ["no_proxy"] = f"localhost,127.0.0.1,::1,172.30.56.42"
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), 'tmp')

sys.path.insert(0, 'online_demo/Backend/backend_ixc/third_party')
from melo.api import TTS

import os

import re

import cv2
import gradio as gr
import librosa
import numpy as np
import requests
from PIL import Image
from lmdeploy import pipeline, GenerationConfig, VisionConfig
from swift.llm import InferRequest, PtEngine, RequestConfig

from ixc_util import img_process

from decord import VideoReader, cpu

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PUNCTUATION = "!?.;！？。；"


def remove_special_characters(input_str):
    # Remove special tokens
    special_tokens = ['☞', '☟', '☜', '<unk>', '<|im_end|>']
    for token in special_tokens:
        input_str = input_str.replace(token, '')
    return input_str


def sample_frames_from_video_decord(video_path, max_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)  # 视频的总帧数
    fps = vr.get_avg_fps()  # 平均帧率
    duration = total_frames / fps  # 视频时长（秒）

    # 计算帧索引
    if duration <= max_frames:
        frame_indices = np.arange(0, total_frames, fps).astype(int)  # 每秒采样一帧
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames,
                                    dtype=int)  # 平均采样 max_frames 帧

    sampled_frames = []

    # 提取并保存帧
    for i, frame_idx in enumerate(frame_indices):
        frame = vr[frame_idx].asnumpy()  # 获取帧并转为 NumPy 数组
        sampled_frames.append(Image.fromarray(frame))

    return sampled_frames


def is_video(file_path):
    video_extensions = {
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'
    }
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions


def is_image(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions


def is_wav(file_path):
    wav_extensions = {'.wav'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in wav_extensions


def split_into_sentences(text):
    sentence_endings = re.compile(r'[!?.;！？。；]')
    sentences = sentence_endings.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def convert_webm_to_mp4(input_file, output_file):
    try:
        cap = cv2.VideoCapture(input_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, 20.0,
                              (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
    except Exception as e:
        print(f"Error: {e}")
        raise


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0

    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0 and count % 2 == 1:
                line = line.replace("`", r"\`")
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                line = line.replace(" ", "&nbsp;")
                line = line.replace("*", "&ast;")
                line = line.replace("_", "&lowbar;")
                line = line.replace("-", "&#45;")
                line = line.replace(".", "&#46;")
                line = line.replace("!", "&#33;")
                line = line.replace("(", "&#40;")
                line = line.replace(")", "&#41;")
                line = line.replace("$", "&#36;")
            lines[i] = "<br>" + line

    return "".join(lines)


def add_text(history, task_history, text):
    task_text = text
    if len(text
           ) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
        task_text = text[:-1]
    history = history + [(_parse_text(text), None)]
    task_history = task_history + [(task_text, None)]
    return history, task_history


def add_file(history, task_history, file):
    history = history + [((file.name, ), None)]
    task_history = task_history + [((file.name, ), None)]
    return history, task_history


def add_audio(history, task_history, file):
    if file is None:
        return history, task_history

    infer_request = InferRequest(messages=[{
        "role":
        "user",
        "content":
        '<audio>Detect the language and recognize the speech.'
    }], audios=[file])
    request_config = RequestConfig(max_tokens=256, temperature=0)

    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content

    history = history + [((file, ), None), (response, None)]
    task_history = task_history + [((file, ), None), (response, None)]

    return history, task_history


def add_audio_ana(history, task_history, file):
    if file is None:
        return history, task_history

    infer_request = InferRequest(messages=[{
        "role":
        "user",
        "content":
        '<audio>Classify the audio.'
    }], audios=[file])
    request_config = RequestConfig(max_tokens=256, temperature=0)

    resp_list = engine.infer([infer_request], request_config)
    audio_cls = resp_list[0].choices[0].message.content
    response = f'Audio cls: {audio_cls}.'

    keep_word = ('speech', 'human', 'voice')
    is_speech = False
    for word in keep_word:
        if word in audio_cls.lower():
            is_speech = True
    if is_speech:
        infer_request = InferRequest(messages=[{
            "role":
                "user",
            "content":
                '<audio>Detect the language and recognize the speech.'
        }], audios=[file])
        request_config = RequestConfig(max_tokens=256, temperature=0)

        resp_list = engine.infer([infer_request], request_config)
        audio_text = resp_list[0].choices[0].message.content
        response += f' Audio text: {audio_text}'

    history = history + [((file, ), response)]
    task_history = task_history + [((file, ), response)]

    return history, task_history


def add_video(history, task_history, file):
    print(f'Video record: {file}')
    if file is None:
        return history, task_history
    new_file_name = file.replace(".webm", ".mp4")
    print(f'Convert webm to mp4')
    if file.endswith(".webm"):
        convert_webm_to_mp4(file, new_file_name)
    task_history = task_history + [((new_file_name, ), None)]
    return history, task_history


def reset_user_input():
    return gr.update(value="")


def reset_state(task_history):
    task_history.clear()
    return []


def stream_audio_output(history, task_history):
    text = task_history[-1][-1]
    for idx, chunk_text in enumerate(split_into_sentences(text)):
        audio = t2s_model.tts_to_file(chunk_text,
                                      speaker_ids['ZH'],
                                      None,
                                      speed=1.0,
                                      quiet=True)
        audio = audio * 10
        audio_resampled = librosa.resample(
            audio, orig_sr=t2s_model.hps.data.sampling_rate, target_sr=24000)
        yield 24000, audio_resampled


def predict(_chatbot, task_history):
    text_query = []
    image_query = []
    video_query = []
    chat_query = task_history[-1][0]

    print('##### Begin of predict #####')
    print(task_history)

    for i, (q, a) in enumerate(task_history):
        if isinstance(q, (tuple, list)):
            if is_image(q[0]):
                image_query.append(q[0])
            if is_video(q[0]):
                video_query.append(q[0])

    print(f'len image query: {len(image_query)}, len video query: {len(video_query)}')
    if len(image_query) > 0 and len(video_query) > 0:
        raise gr.Error("You can't upload video and image in the same session. Please refresh and upload again.")

    if len(image_query) > 1:
        gr.Warning("You have upload multiple images, please describe clearly in your prompt.")
        image_query = img_process(image_query)
        image_query.save('tmp.png')
        image_query = 'tmp.png'
    elif len(image_query) == 1:
        image_query = image_query[0]
    else:
        image_query = []

    if len(video_query) > 1:
        video_query = video_query[-1:]
    if len(video_query) == 1:
        video_query = video_query[0]
        video_frames = sample_frames_from_video_decord(video_query)
        image_query = img_process(video_frames)
        image_query.save('tmp.png')
        image_query = 'tmp.png'

    print(f'Text query: {chat_query}')
    print(f'Image query: {image_query}')

    response = pipe.stream_infer((chat_query, image_query),
                                 gen_config=gen_config)
    output = ''
    for chunk in response:
        if chunk.text:
            output += chunk.text
            task_history[-1] = (chat_query, output)
            remove_special_characters_output = remove_special_characters(
                output)
            _chatbot[-1] = (chat_query,
                            _parse_text(remove_special_characters_output))
            yield _chatbot

    print("query", chat_query)
    print("answer:  ", output)
    print(task_history)
    print('##### End of predict #####')


def predict_audio(_chatbot, task_history):
    print('##### Begin of predict #####')
    audio_query = task_history[-1][0][0]
    files = {'audio': open(audio_query, 'rb')}
    print(f'Audio query: {audio_query}')
    output = ''
    with requests.post(audio_url, files=files, stream=True,
                       verify=False) as response:
        # 逐块读取并处理响应
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                output += chunk
                if audio_query:
                    chat_query = (audio_query, )
                task_history[-1] = (chat_query, output)
                remove_special_characters_output = remove_special_characters(
                    output)
                _chatbot[-1] = (chat_query,
                                _parse_text(remove_special_characters_output))
                yield _chatbot
        print("query", chat_query)
        print("answer:  ", output)
    print(task_history)
    print('##### End of predict #####')


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(title="VideoMLLM", js=js_func, theme=gr.themes.Ocean()) as demo:
    gr.Image('assets/logo_en.png', width=600)
    chatbot = gr.Chatbot(label='IXC-OL',
                         elem_classes="control-height",
                         height=500)
    query = gr.Textbox(lines=2, label='Text Input')
    task_history = gr.State([])
    with gr.Row():
        add_text_button = gr.Button("Submit Text for Chat (提交文本进行对话)")
        add_audio_button = gr.Button("Submit Audio for Chat (提交音频进行对话)")
        add_audio_ana_button = gr.Button(
            "Submit Audio for Analysis (提交音频进行语音分析)")
    with gr.Row():
        with gr.Column(scale=2):
            addfile_btn = gr.UploadButton("📁 Upload (上传文件[视频,图片])",
                                          file_types=["video", "image"])
            video_input = gr.Video(sources=["webcam"],
                                   height=400,
                                   width=700,
                                   container=True,
                                   interactive=True,
                                   show_download_button=True,
                                   label="📹 Video Recording (视频录制)")

        with gr.Column(scale=1):
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            record_btn = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Record or Upload Audio (录音或上传音频)",
                show_download_button=True,
                waveform_options=gr.WaveformOptions(sample_rate=16000))
            audio_output = gr.Audio(
                label="Output Audio",
                value=None,
                format="wav",
                autoplay=True,
                streaming=True,
                interactive=False,
                show_label=True,
                waveform_options=gr.WaveformOptions(sample_rate=24000, ),
            )

    add_text_button.click(add_text, [chatbot, task_history, query],
                          [chatbot, task_history]).then(
                              reset_user_input, [],
                              [query]).then(predict, [chatbot, task_history],
                                            [chatbot],
                                            show_progress=True).then(
                                                stream_audio_output,
                                                [chatbot, task_history],
                                                [audio_output],
                                            )

    # video_input.upload(add_video, [chatbot, task_history, video_input],
    #                    [chatbot, task_history])
    video_input.stop_recording(add_video, [chatbot, task_history, video_input], [chatbot, task_history])
    empty_bin.click(reset_state, [task_history], [chatbot])
    addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn],
                       [chatbot, task_history])

    add_audio_button.click(add_audio, [chatbot, task_history, record_btn],
                           [chatbot, task_history]).then(
                               predict, [chatbot, task_history], [chatbot],
                               show_progress=True).then(
                                   stream_audio_output,
                                   [chatbot, task_history],
                                   [audio_output],
                               )
    add_audio_ana_button.click(add_audio_ana, [chatbot, task_history, record_btn],
                           [chatbot, task_history]).then(
                                   stream_audio_output,
                                   [chatbot, task_history],
                                   [audio_output],
                               )

# Init ixc
print('Init ixc model')
ixc_prompt = """你是一个多模态人工智能助手，名字叫浦语·灵笔。
- 浦语·灵笔是由上海人工智能实验室开发的一个多模态对话模型，是一个有用，真实且无害的模型。
- 浦语·灵笔可以根据看到和听到的内容，流利的同用户进行交流，并使用用户使用的语言（中文或英文）进行回复。
"""

hf_model = f'internlm-xcomposer2d5-ol-7b/merge_lora'
pipe = pipeline(hf_model, vision_config=VisionConfig(thread_safe=True))
pipe.chat_template.meta_instruction = ixc_prompt
gen_config = GenerationConfig(top_k=50, top_p=0.8, temperature=0.1)

# Init Audio
print('Init audio model')
model_id_or_path = 'internlm-xcomposer2d5-ol-7b/audio'
engine = PtEngine(model_id_or_path,
                  model_type='qwen2_audio',
                  device_map='cuda:0')

# Init TTS
t2s_model = TTS(language="ZH", device="auto")
speaker_ids = t2s_model.hps.data.spk2id
t2s_model.tts_to_file('123', speaker_ids['ZH'], None, speed=1.0, quiet=True)

demo.launch(
    share=False,
    show_api=False,
    show_error=False,
    max_threads=1,
    server_name='172.30.56.42',
)
