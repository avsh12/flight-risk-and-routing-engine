import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


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
