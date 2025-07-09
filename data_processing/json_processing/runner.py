import os
import re
from json_processing.converter import json_to_csv

def process_json_directory(path):
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for json_file in os.listdir(subfolder_path):
            if not json_file.endswith(".json"):
                continue

            json_file_path = os.path.join(subfolder_path, json_file)
            json_to_csv(json_file_path)
