import pandas as pd
from utils.loaders import getConfigResourcePath, load_yaml


def dropCancelled(df: pd.DataFrame) -> pd.DataFrame:
    len_df = len(df)
    """
    We will drop rows for cancelled flights
    """

    df = df.query("CANCELLED == 0").drop(columns=["CANCELLED"])
    # reset the index column for the new data size
    df = df.reset_index(drop=True)

    print(f"Percentage of cancelled flights: {100*(len_df-len(df))/len_df:.4f} %")
    return df


def dropDuplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(ignore_index=True)
    return df


def selectColumns(df: pd.DataFrame, columns):
    df = df[columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    schema_path = getConfigResourcePath("schema")
    schema = load_yaml(schema_path)

    df = df.pipe(dropCancelled).pipe(dropDuplicates)
    return df
