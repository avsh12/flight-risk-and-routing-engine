import pandas as pd
from etl import airports, date_encoding, flight, index_categories, join, scale, weather
from utils.loaders import getConfigResourcePath, load_yaml


def flight_etl() -> pd.DataFrame:
    schema_path = getConfigResourcePath("schema")
    schema = load_yaml(schema_path)

    # Flight Extract
    flight_data = flight.extract.extract()

    # Schema Validation
    flight_data = flight.schema_validation.schema_validaton(
        flight_data, schema["bronze_schema"]
    )

    # Clean & Deduplicate
    flight_data = flight.clean.clean(flight_data)

    # Flight Transform & Feature Engineering
    flight_data = (
        flight_data.pipe(flight.clean.drop_null)
        .pipe(flight.transform.create_new_features)
        .pipe(flight.clean.drop_null)
    )

    return flight_data


def airport_etl(flight_data: pd.DataFrame) -> pd.DataFrame:
    airport_codes = airports.get_airports(flight_data)
    return airports.get_airports_location(airport_codes)


def weather_etl(
    airport_locations: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    # Weather Extract
    weather_data = weather.extract.extract(airport_locations, start_date, end_date)

    # Weather Clean and Transform
    weather_data = weather.clean.clean(weather_data)
    weather_data = weather.transform.transform(weather_data)

    return weather_data


def pipeline(
    start_date: str = "2015-01-01",
    end_date: str = "2015-01-03",
) -> pd.DataFrame:

    # Flight ETL
    flight_data = flight_etl()

    # Airports Location
    airport_locations = airport_etl(flight_data)

    # Weather ETL
    weather_data = weather_etl(airport_locations, start_date, end_date)

    # Combine and Join
    flight_weather_data = join.combine_flight_weather(flight_data, weather_data)

    del flight_data, weather_data

    # Index Categories
    flight_weather_data = index_categories.index_categorical(flight_weather_data)

    # Scaling and Cyclical Encoding
    flight_weather_data = scale.scale(flight_weather_data)
    flight_weather_data = date_encoding.encode_date(flight_weather_data)

    return flight_weather_data
