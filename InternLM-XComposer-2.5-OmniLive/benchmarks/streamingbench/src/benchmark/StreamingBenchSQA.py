import tqdm
import os
import json

from utils.data_execution import get_model_response
from utils.video_execution import split_video

from benchmark.Benchmark import Benchmark

PROMPT_TEMPLATE = '''You are an advanced video question-answering AI assistant. You have been provided with a video and a multiple-choice question related to the video. Your task is to carefully analyze the video and the provided context to answer the question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

{}

Here is the question. Answer it and don't confuse it with the previous conversation.
Question: {}

Options:
{}
{}
{}
{}

The best option is:'''

class StreamingBenchSQA(Benchmark):
    def __init__(self, data):
        StreamingBenchSQAInit(data)

    def eval(self, data, model, output_path):
        StreamingBenchSQAEval(data, model, output_path)

def StreamingBenchSQAInit(data):
    pass

def StreamingBenchSQAEval(data, MODEL, output_path):
    for video_data in tqdm.tqdm(data):  
        context = ""
        for subset in video_data:  
            for question in subset["questions"]:
                if MODEL.name() in question and question[MODEL.name()]:
                    continue

                video_path = subset["video_path"]
                timestamp = question["time_stamp"]
                # convert timestamps like "00:03:10" to seconds
                timestamp = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))

                file = split_video(video_path, 0, timestamp, 'sqa')

                ques = question["question"]
                options = question["options"]

                if not options[0].startswith("A."):
                    options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]

                inp = PROMPT_TEMPLATE.format(context, ques, *options)

                print(f"input: {inp}")

                response = get_model_response(MODEL, file, inp, 0)
                question[MODEL.name()] = response
                
                if not context:
                    context += "Here are the contextual information related to the video. Please answer the questions based on the contextual information: "
                context += f"At timestamp {question['time_stamp']}, the following question and answer occurred: Question: {ques}; Options: {options[0]}, {options[1]}, {options[2]}, {options[3]}; Answer: {question['answer']}; "

                with open(output_path, "w") as f:
                    json.dump(data, f, indent=4)

                # remove the clip file
                # os.remove(file)
