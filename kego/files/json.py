import json


def load_json(filepath):
    with open(filepath) as file:
        data = json.load(file)
    return data
