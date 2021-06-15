import os
import json



def dump_PRM(context_config, model, directory_path="./"):
    """
    """
    # Create a directory if it does not exist
    # Dump model configuration as JSON
    # Dump GraphML 

    data = {}
    data["samples"] = [list(sample) for sample in model.samples]
    data["config"] = context_config

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    file_path = os.path.join(directory_path, "data.json")
    with open(file_path, "w") as file:
        json.dump(data, file)

def load_PRM(directory_path):
    file_path = os.path.join(directory_path, "data.json")
    with open(file_path, "r") as file:
        return json.load(file)

