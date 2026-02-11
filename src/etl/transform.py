import asyncio
import gc
import itertools
import math
import os
import re
import sys
import time
import timeit
from datetime import datetime, timezone
from itertools import tee
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import openmeteo_requests
import pandas as pd
import requests
import requests_cache
import yaml
from airports import airport_data
from retry_requests import retry

from etl.flight_extract import extract

"""Create features?
1. Previous_Flight_Delay (How late was the plane coming in?).
2. Origin_Airport_Congestion (Average delay of all flights leaving that airport in the last hour).
"""


def createNewFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. First feature set
        The function creates the following features in the dataframe,
        1. SCH_DEP_HOUR
        2. SCH_DEP_MIN
        3. DEP_HOUR
        4. DEP_MIN
        5. SCH_ARI_HOUR
        6. SCH_ARI_MIN
        7. ARI_HOUR
        8. ARI_MIN
        using the times in the columns SCHEDULED_DEPARTURE, DEPARTURE_TIME, SCHEDULED_ARRIVAL, ARRIVAL_TIME.

        The time of departure and arrival are stored as float values in SCHEDULED_DEPARTURE, DEPARTURE_TIME, SCHEDULED_ARRIVAL, ARRIVAL_TIME. The float value is interpreted in four-digit format where the first two digits are for hours and the next two digits for minutes.

    2. Second feature set
        1. Recalculate SCH_ARI_MIN and ARI_MIN using the departure date and time, and the time of travel.
        2. Calculate SCH_DEP_DATE, SCH_ARI_DATE, and SCH_DEP_DAY.
    """

    # Separate the hour and minute values.
    h, m = np.divmod(df["SCHEDULED_DEPARTURE"], 100)
    df["SCH_DEP_HOUR"], df["SCH_DEP_MIN"] = h, m
    h, m = np.divmod(df["DEPARTURE_TIME"], 100)
    df["DEP_HOUR"], df["DEP_MIN"] = h, m

    h, m = np.divmod(df["SCHEDULED_ARRIVAL"], 100)
    df["SCH_ARI_HOUR"], df["SCH_ARI_MIN"] = h, m
    h, m = np.divmod(df["ARRIVAL_TIME"], 100)
    df["ARI_HOUR"], df["ARI_MIN"] = h, m

    # Reorder the columns. Move the created columns after the YEAR, MONTH, and DAY columns.
    cols = df.columns.to_list()
    cols = (
        cols[:3]
        + [
            "SCH_DEP_HOUR",
            "SCH_DEP_MIN",
            "DEP_HOUR",
            "DEP_MIN",
            "SCH_ARI_HOUR",
            "SCH_ARI_MIN",
            "ARI_HOUR",
            "ARI_MIN",
        ]
        + cols[3:-8]
    )

    # Reorder the columns using the rearranged column names.
    df = df.reindex(columns=cols)

    # Drop the time columns after the hour and minute values are extracted.
    df.drop(
        ["SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "SCHEDULED_ARRIVAL", "ARRIVAL_TIME"],
        axis=1,
        inplace=True,
    )

    # Convert the scheduled departure date and time values to pandas datetime object
    sch_departure_datetime = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY", "SCH_DEP_HOUR", "SCH_DEP_MIN"]].rename(
            columns={
                "YEAR": "year",
                "MONTH": "month",
                "DAY": "day",
                "SCH_DEP_HOUR": "hour",
                "SCH_DEP_MIN": "minute",
            }
        )
    )

    # Convert the departure date and time values to pandas datetime object
    departure_datetime = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY", "DEP_HOUR", "DEP_MIN"]].rename(
            columns={
                "YEAR": "year",
                "MONTH": "month",
                "DAY": "day",
                "DEP_HOUR": "hour",
                "DEP_MIN": "minute",
            }
        )
    )

    # Convert the scheduled time of travel values to pandas datetime object
    sch_time = pd.to_timedelta(df["SCHEDULED_TIME"], unit="m")
    # Convert the actual time of travel to pandas datetime object
    elapsed_time = pd.to_timedelta(df["ELAPSED_TIME"], unit="m")

    # The hour column of the arrival time is not the sum of departure and time of travel.
    # Time zone difference needs to be taken into account.
    # Assuming the timezone difference is integral number of hours, we can calculate the ARI_MIN using the departure datetime and the time of travel. We recalculate the feature as there can be errors in the precomputed values.
    df["SCH_ARI_MIN"] = (sch_departure_datetime + sch_time).dt.minute
    df["ARI_MIN"] = (departure_datetime + elapsed_time).dt.minute

    """
    sch_time = pd.to_timedelta(df["SCHEDULED_TIME"], unit="m")
    We have ignored the timezone difference for the present and calculate the scheduled arrival date as sum of sch_departure_datetime and 
    """
    sch_arrival_datetime = sch_departure_datetime + sch_time
    df.insert(0, "SCH_DEP_DATE", sch_departure_datetime.dt.date.astype("string"))
    df.insert(1, "SCH_ARI_DATE", sch_arrival_datetime.dt.date.astype("string"))
    df.insert(
        df.columns.get_loc("SCH_ARI_HOUR"), "SCH_ARI_DAY", sch_arrival_datetime.dt.day
    )

    return df


def dropNull(df: pd.DataFrame) -> pd.DataFrame:
    null_containing_columns = []
    null_counts = []

    # loop through the columns and store the indices of null rows and the number of null values for the each column.
    for col in df.columns:
        null_indices = df[col].isna()
        null_count = null_indices.sum()

        if null_count != 0:
            # Get the indices of null fields if there is any null values in the column.
            null_indices = null_indices[null_indices].index
            # Drop the null values.
            df[col].drop(null_indices, inplace=True)
            null_containing_columns.append(col)
            null_counts.append(null_count)

    print(
        f"""Columns containg null values: \
        {list(zip(zip(df.columns[null_containing_columns], null_containing_columns),
                    null_counts))}"""
    )

    return df


def combineWeatherAndFlightData(
    df: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    # Get the weather data for the origin airport.
    source_weather = weather.loc[
        (
            df["ORIGIN_AIRPORT"].values,
            df["SCH_DEP_DATE"].values,
            df["SCH_DEP_HOUR"].values,
        ),
    ]

    # Get the weather data for the destination airport.
    destination_weather = weather.loc[
        (
            df["DESTINATION_AIRPORT"].values,
            df["SCH_ARI_DATE"].values,
            df["SCH_ARI_HOUR"].values,
        ),
    ]

    # Concatenate the origin and destination weather data to the flight dataframe.
    df_flight_weather = pd.concat(
        [
            df.reset_index(drop=True),
            source_weather.reset_index(drop=True).add_prefix("SOURCE_"),
            destination_weather.reset_index(drop=True).add_prefix("DEST_"),
        ],
        axis=1,
    )
    # Clean up unnecessary memory
    del source_weather, destination_weather
    gc.collect()

    # Drop the date and time columns that are not needed.
    df_flight_weather.drop(
        [
            "SCH_DEP_DATE",
            "SCH_ARI_DATE",
            "SCHEDULED_TIME",
            "ELAPSED_TIME",
            "AIR_TIME",
            "DISTANCE",
        ],
        axis=1,
        inplace=True,
    )

    return df_flight_weather


def indexCategorical(
    df: pd.DataFrame, airports: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Categorical Columns:
    AIRLINE
    FLIGHT_NUMBER
    TAIL_NUMBER
    ORIGIN_AIRPORT
    DESTINATION_AIRPORT
    """

    # Create Categorical data types for the categorical columns.
    airline_categories = pd.CategoricalDtype(df["AIRLINE"].unique())
    flight_no_categories = pd.CategoricalDtype(df["FLIGHT_NUMBER"].unique())
    tail_no_categories = pd.CategoricalDtype(df["TAIL_NUMBER"].unique())
    airport_categories = pd.CategoricalDtype(airports["IATA"].unique())

    # Map between categorical columns and the respective data types.
    categories = {
        "AIRLINE": airline_categories,
        "FLIGHT_NUMBER": flight_no_categories,
        "TAIL_NUMBER": tail_no_categories,
        "ORIGIN_AIRPORT": airport_categories,
        "DESTINATION_AIRPORT": airport_categories,
    }

    # Change the data types to categorical data types.
    for category in categories:
        df[category] = df[category].astype(categories[category]).cat.codes

    return df, categories


def runETLPipeline(
    flightdata_filepath: str = "./../data/flights.csv",
    start_date: str = "2015-01-01",
    end_date: str = "2015-01-03",
) -> pd.DataFrame:

    flight_data = extractFlightData(flightdata_filepath)
    flight_data = flight_data.sample(3)

    # Data cleaning pipeline
    flight_data = (
        flight_data.pipe(dropCancelled)
        .pipe(keepRelevantColumns)
        .pipe(dropTypeMismatchRows)
    )

    # Feature Engineering and data cleaning
    flight_data = flight_data.pipe(createNewFeatures).pipe(dropNull)

    # Data Extraction
    weather_data = flight_data.pipe(getAirportData).pipe(
        getWeatherData, start_date, end_date
    )

    # Data Transformation
    weather_data = weatherDataProcessing(weather_data)
    # Data Aggregation
    flight_weather_data = combineWeatherAndFlightData(flight_data, weather_data)

    del flight_data, weather_data

    flight_weather_data = indexCategorical(flight_weather_data)

    return flight_weather_data


def triggerETLPipeline(
    flightdata_filepath: str = "./../data/flights.csv",
    start_date: str = "2015-01-01",
    end_date: str = "2015-01-03",
):
    flight_weather_data = runETLPipeline(flightdata_filepath, start_date, end_date)
    flight_weather_data.to_parquet("flight_weather_data.parquet")
