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
from etl.clean import clean
from etl.flight_extract import extract
from retry_requests import retry

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


def getAirportData(df: pd.DataFrame) -> pd.DataFrame:
    # Get unique airport IATA codes from the origin and destination airports.
    origin_airports = pd.Index(df["ORIGIN_AIRPORT"].unique())
    destination_airports = pd.Index(df["DESTINATION_AIRPORT"].unique())
    airports = origin_airports.union(destination_airports)

    # Fetch the details of airports using their IATA codes.
    airports = airport_data.get_multiple_airports(airports)

    num_airports = len(airports)
    iata_codes = np.zeros(num_airports, dtype="U3")
    latitudes = np.zeros(num_airports)
    longitudes = np.zeros(num_airports)

    # Store IATA code, Latitude, and Longitude of the airports as DataFrame.
    for ix, airport in enumerate(airports):
        iata_codes[ix] = airport["iata"]
        latitudes[ix] = airport["latitude"]
        longitudes[ix] = airport["longitude"]
    airport_column_names = ["IATA", "LATITUDE", "LONGITUDE"]
    airport_columns = [iata_codes, latitudes, longitudes]

    airports = pd.DataFrame(dict(zip(airport_column_names, airport_columns)))
    return airports


# features is a newline-separated string of the data needed
# Weight of API call is calculated as
# weight = nLocations * (nDays / 14) * (nVariables / 10)
def weatherRequest(url, latitude, longitude, start_date, end_date, features):
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": features,
    }

    return openmeteo.weather_api(url, params=params)


# Actual limit is 600/min, 5000/hour and 10000/day
def apiCallsPerMin(
    num_days,
    num_features,
    num_values,
    max_api_calls_per_day=10000,
    max_api_calls_per_hour=5000,
    max_api_calls_per_min=600,
):
    num_api_calls_required = num_values * (num_days / 14) * (num_features / 10)
    api_calls_per_min = 0

    if num_api_calls_required >= max_api_calls_per_day:
        api_calls_per_min = max_api_calls_per_day // (24 * 60)
    elif num_api_calls_required >= max_api_calls_per_hour:
        api_calls_per_min = max_api_calls_per_hour // 60
    else:
        api_calls_per_min = max_api_calls_per_min

    print(f"API call rate: {api_calls_per_min} /min")
    print(f"Number of API calls: {num_api_calls_required:.0f}")

    return api_calls_per_min


def numDataSplits(start_date, end_date, num_features, num_values):
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    api_calls_per_min = apiCallsPerMin(num_days, num_features, num_values)

    num_splits = num_values // (api_calls_per_min * 140 / (num_days * num_features))

    print(f"Total time: {num_splits} min")

    return num_splits


def getWeatherData(airports, start_date="2015-01-01", end_date="2015-01-03"):
    url = "https://archive-api.open-meteo.com/v1/archive"

    features = """temperature_2m
    rain
    snowfall
    cloud_cover_low
    cloud_cover_high
    wind_speed_10m
    wind_speed_100m
    wind_gusts_10m""".split()

    # max_api_calls_per_day = 10000
    # max_api_calls_per_hour = 5000
    # max_api_calls_per_min = 600

    # num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    num_features = len(features)
    num_values = len(airports)
    # num_api_calls_required = len(airports) * (num_days / 14) * (num_features / 10)

    # if num_api_calls_required >= max_api_calls_per_day:
    #     api_calls_per_min = max_api_calls_per_day // (24 * 60)
    # elif num_api_calls_required >= max_api_calls_per_hour:
    #     api_calls_per_min = max_api_calls_per_hour // 60
    # else:
    #     api_calls_per_min = max_api_calls_per_min

    # api_calls_per_min = apiCallsPerMin(start_date, end_date, num_features, num_values)

    # num_splits = len(airports) // (api_calls_per_min * 140 / (num_days * num_features))
    num_splits = numDataSplits(start_date, end_date, num_features, num_values)

    responses = []
    logs = []

    # latlong_splitted = np.array_split(airports[['LATITUDE', 'LONGITUDE']].values.T, num_splits, axis=1)
    # For sample run, consider two airports.
    latlong_splitted = [airports.loc[:2, ["LATITUDE", "LONGITUDE"]].values.T]
    del_t = 0

    for ix, (latitude, longitude) in enumerate(latlong_splitted):
        try:
            start = time.time()
            response = weatherRequest(
                url, latitude, longitude, start_date, end_date, features
            )
            del_t = time.time() - start

            responses.extend(response)

        except:
            error = f"Error occured in block {ix}"
            logs.append((ix, error))
            print(error)

        if (del_t < 60) & (ix < len(latlong_splitted) - 1):
            time.sleep(65 - del_t)

    return responses, logs


def weatherDataProcessing(responses: list) -> pd.DataFrame:
    # Create pandas DateTimeIndex for the range of date with a time step of 1 hour
    t = pd.date_range(
        start=pd.to_datetime(responses[0].Hourly().Time(), unit="s"),
        end=pd.to_datetime(responses[0].Hourly().TimeEnd(), unit="s"),
        freq=pd.Timedelta(responses[0].Hourly().Interval(), unit="s"),
        inclusive="left",
    )
    # Store the date and hour as a numpy array
    t = np.array([[str(t.date()), int(t.hour)] for t in t], dtype="object")

    # create repeated arrays for IATA codes and date to create cartesian product of
    # each iata code with the date array
    l_iata = 3  # len(airports)
    l_date = len(t)

    # Concatenate the IATA code, date, and hour columns
    # Each IATA code is to be repeated vertically upto the full range of date.
    # Each date is to be repeated for the full set of IATA codes
    index = np.hstack(
        [
            np.repeat(
                airports["IATA"].values.reshape(-1, 1)[:l_iata, :], l_date, axis=0
            ),
            np.tile(t, (l_iata, 1)),
        ]
    )

    # Create multiindex using the index array
    index = pd.MultiIndex.from_arrays(index.T, names=["IATA", "DATE", "HOUR"])
    # Create DataFrame from the features in the respnses and the multiindex as index of the DataFrame.
    weather = pd.DataFrame(
        np.concatenate(
            [
                np.stack(
                    [
                        responses[j].Hourly().Variables(i).ValuesAsNumpy()
                        for i in range(len(features))
                    ],
                    axis=1,
                )
                for j in range(len(responses))
            ],
            axis=0,
        ),
        columns=features,
        index=index,
    )
    # Uppercase all the column names
    weather.columns = weather.columns.str.upper()

    return weather


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


flight_data = extract()
flight_data = flight_data.sample(10)

discarded_list = []

flight_data = clean(flight_data)
# Data cleaning pipeline
# flight_data = (
#     flight_data.pipe(dropCancelled)
#     .pipe(dropDuplicates)
#     .pipe(dropTypeMismatchRows, schema["silver_schema"], discarded_list=discarded_list)
# )

# Feature Engineering and data cleaning
# flight_data = flight_data.pipe(createNewFeatures).pipe(dropNull)
# discarded_flight_data = pd.DataFrame({}, columns=load_columns)
# discarded_flight_data = pd.concat(discarded_list, axis=0, join="outer")

print(flight_data)
