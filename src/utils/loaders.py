import yaml

from utils.constants import CONFIG_DIR


def getConfigResourcePath(file_name):
    config_path = CONFIG_DIR / f"{file_name}.yaml"
    if config_path.exists():
        return config_path
    else:
        raise FileNotFoundError(f"Missing {file_name} at {config_path}")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
