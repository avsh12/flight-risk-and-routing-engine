from pathlib import Path

import pandas as pd
import yaml
from utils.constants import PROJECT_ROOT
from utils.loaders import getConfigResourcePath, load_yaml


def extractFlightData(filepath: str, columns: list = None, dtype=None) -> pd.DataFrame:
    if columns is None:
        df = pd.read_csv(filepath, engine="c", dtype=dtype)
    else:
        df = pd.read_csv(filepath, usecols=columns, engine="c", dtype=dtype)
    return df


def extract():
    config_path = getConfigResourcePath("config")
    schema_path = getConfigResourcePath("schema")

    config = load_yaml(config_path)
    schema = load_yaml(schema_path)

    filepath = PROJECT_ROOT / config["data"]["raw_path"]
    load_columns = config["pipeline"]["load_columns"]

    flight_data = extractFlightData(
        filepath=filepath,
        columns=load_columns,
        dtype=schema["silver_schema"],
    )

    return flight_data
