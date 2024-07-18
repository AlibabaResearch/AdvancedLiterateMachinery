import json
import os

class JsonHelper:
    @staticmethod
    def load_json(data_path):
        with open(data_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_json(data_path, data):
        file_dir = os.path.dirname(data_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(data_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)