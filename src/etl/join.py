import gc

import pandas as pd


def combine_flight_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
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
