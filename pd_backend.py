from datetime import datetime
from mmap import mmap

import pandas as pd
import typing

TAXI_DATA_URL = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata"


def download_data(
    start: typing.Union[str, datetime],
    end: typing.Union[str, datetime],
    backend: str = "pandas",
    mmap: bool = False,
) -> pd.DataFrame:
    """
    Downloads data from the API and returns a pandas dataframe.
    Start inclusive, end not inclusive.
    """
    if isinstance(start, str):
        # Convert to datetime
        start = datetime.strptime(start, "%Y-%m-%d")
    if isinstance(end, str):
        # Convert to datetime
        end = datetime.strptime(end, "%Y-%m-%d")

    # Extract months and years from start and end
    start_month = start.month
    start_year = start.year
    end_month = end.month if end.month >= start_month else end.month + 12
    end_year = end.year

    # Put all data in pandas dataframe
    dfs = []
    for month in range(start_month, end_month + 1):
        month = month % 12 if month > 12 else month
        for year in range(start_year, end_year + 1):
            print(f"Downloading {month}/{year}")
            df = pd.read_csv(
                f"{TAXI_DATA_URL}_{year}-{month:02d}.csv",
                memory_map=mmap,
            )
            dfs.append(df)

    df = pd.concat(dfs)
    df = df[
        (df.tpep_pickup_datetime >= start) & (df.tpep_pickup_datetime < end)
    ].reset_index(drop=True)
    return df


if __name__ == "__main__":
    import time

    start_time = time.time()
    df = download_data(
        start="2015-01-01",
        end="2015-01-31",
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

    start_time = time.time()
    df = download_data(
        start="2015-01-01",
        end="2015-01-31",
        mmap=True,
    )
    end_time = time.time()
    print(f"Time taken with mmap: {end_time - start_time}")
