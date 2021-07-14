import os
import json
import datetime
import igraph

def dump_PRM(context_config, model, directory_path="./", filename="data.json"):
    """
    """
    # Create a directory if it does not exist
    # Dump model configuration as JSON
    # Dump GraphML 

    data = {}
    data["samples"] = [[float(val) for val in sample] for sample in model.samples]
    data["config"] = context_config

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    file_path = os.path.join(directory_path, filename)
    with open(file_path, "w") as file:
        json.dump(data, file)

def load_PRM(directory_path, filename="data.json"):
    file_path = os.path.join(directory_path, filename)
    with open(file_path, "r") as file:
        return json.load(file)


def dump_model(context_config, model, parent_directory_path="./", directory_name=None):
    """
    """
    # Create a directory if it does not exist
    # Dump model configuration as JSON
    # Dump sampples as JSON
    # Dump GraphML 

    now = datetime.datetime.today()
    nTime = now.strftime('%Y-%m-%dT%H-%M-%S') if directory_name is None else directory_name
    directory_path = os.path.join(parent_directory_path, nTime) 
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Dump Planning Configruration
    file_path = os.path.join(directory_path, './config.json')
    with open(file_path, "w") as file:
        data = {}
        data["config"] = context_config
        json.dump(data, file)
    
    # Dump Samples
    file_path = os.path.join(directory_path, './samples.json')
    with open(file_path, "w") as file:
        data = {}
        data["samples"] = [[float(val) for val in sample] for sample in model.samples]
        json.dump(data, file)

    # Dump Graph as a graph ML
    file_path = os.path.join(directory_path, './graph.graphml')
    with open(file_path, "w") as file:
        model.graph.write_graphml(file)

def load_model(directory_path):
    # Load Planning Configruration
    file_path = os.path.join(directory_path, 'config.json')
    with open(file_path, "r") as file:
        config = json.load(file)['config']
    
    # Load Samples
    file_path = os.path.join(directory_path, 'samples.json')
    with open(file_path, "r") as file:
        samples = json.load(file)['samples']

    # Load Graph as a graph ML
    file_path = os.path.join(directory_path, 'graph.graphml')
    with open(file_path, "r") as file:
        ipython_graph = igraph.Graph.Read_GraphML(file)

    return config, samples, ipython_graph
