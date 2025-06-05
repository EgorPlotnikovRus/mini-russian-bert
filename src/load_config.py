import yaml

def get_config(path):
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    return params