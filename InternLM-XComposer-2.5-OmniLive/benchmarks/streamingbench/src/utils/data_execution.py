import json

def get_timestamps(question_set):
    """
    """
    timestamps = []
    
    for question in question_set["questions"]:
        timestamps.append(question['time_stamp'])

    return timestamps

def load_data(EVAL_DATA_FILE):
    with open(EVAL_DATA_FILE, "r") as f:
        data = json.load(f)
    
    return data

def get_model_response(model, file, inp, ques, timestamp):
    """
    Get the model response for the given input
    Model: Model name
    file: Video file path
    inp: Input prompt
    """
    return model.Run(file, inp, ques, timestamp)