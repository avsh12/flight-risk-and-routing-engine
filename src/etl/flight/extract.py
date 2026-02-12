from pathlib import Path

import pandas as pd
import yaml
from utils.constants import PROJECT_ROOT
from utils.loaders import getConfigResourcePath, load_yaml
from utils.logger import log_progress


def extract_flight_data(
    filepath: str, columns: list = None, dtype=None
) -> pd.DataFrame:
    df = pd.read_csv(
        filepath, usecols=columns, engine="c", dtype=dtype, low_memory=False
    )
    return df


def extract():
    log_progress("Extracting data")
    config_path = getConfigResourcePath("config")
    schema_path = getConfigResourcePath("schema")

    config = load_yaml(config_path)
    schema = load_yaml(schema_path)

    filepath = PROJECT_ROOT / config["data"]["raw_path"]
    load_columns = config["pipeline"]["load_columns"]

    flight_data = extract_flight_data(
        filepath=filepath,
        columns=load_columns,
        dtype=schema["bronze_schema"],
    )
    log_progress("Data Extraction Done!\n")

    return flight_data
