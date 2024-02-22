import json


def load_json_lines(file):
    with open(file, 'rb') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    return lines


def dump_json_lines(json_list, file):
    with open(file, 'w') as f:
        for obj in json_list:
            f.write(json.dumps(obj) + '\n')
        f.flush()
