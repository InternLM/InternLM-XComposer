import json
import os
from glob import glob
import wave
from tqdm import tqdm
from asr import WebsocketClient


def asr_st(data):
    ws = WebsocketClient(punctuation=1)
    results = {}
    for i in tqdm(range(len(data))):
        with wave.open(data[i], 'rb') as reader:
            n = reader.getnframes()
            frames = reader.readframes(n)
        ws.set_audio(frames)
        ws.run()
        text = ws.txt
        results[data[i]] = text
    return results


def load_data():
    folder = 'NQ_audio'
    data = glob(os.path.join(folder, '*', "*.wav"), recursive=True)
    exists = json.load(open('data.json'))
    data = [it for it in data if it not in exists]
    return data


if __name__ == '__main__':
    data = load_data()
    print(len(data))
    results = asr_st(data)
    json.dump(results, open('data2.json', 'w'))