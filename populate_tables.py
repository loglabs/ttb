import numpy as np
import os
import pandas as pd
from pandas.core.dtypes import dtypes
import psycopg2
import psycopg2.extras as extras

from io import StringIO
from dotenv import load_dotenv

data_dir = os.path.join(os.getcwd(), "data")
processed_fname = f"{data_dir}/processed_files.txt"
load_dotenv()


conn = psycopg2.connect(
    f"host={os.getenv('HOSTNAME')} user={os.getenv('USER')} port={os.getenv('PORT')} password={os.getenv('SECRET')}"
)
conn.set_isolation_level(0)

cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS taxi_data (
        VendorID text,
        tpep_pickup_datetime timestamp without time zone,
        tpep_dropoff_datetime timestamp without time zone,
        passenger_count numeric,
        trip_distance numeric,
        RateCodeID numeric,
        store_and_fwd_flag text,
        PULocationID numeric,
        DOLocationID numeric,
        payment_type text,
        fare_amount numeric,
        extra numeric,
        mta_tax numeric,
        tip_amount numeric,
        tolls_amount numeric,
        improvement_surcharge numeric,
        total_amount numeric,
        congestion_surcharge numeric
    )
"""
)

cur.execute("DROP TABLE IF EXISTS taxi_data_copy;")
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS taxi_data_copy (
        VendorID text,
        tpep_pickup_datetime timestamp without time zone,
        tpep_dropoff_datetime timestamp without time zone,
        passenger_count numeric,
        trip_distance numeric,
        RateCodeID numeric,
        store_and_fwd_flag text,
        PULocationID numeric,
        DOLocationID numeric,
        payment_type text,
        fare_amount numeric,
        extra numeric,
        mta_tax numeric,
        tip_amount numeric,
        tolls_amount numeric,
        improvement_surcharge numeric,
        total_amount numeric,
        congestion_surcharge numeric
    )
"""
)
cur.execute("INSERT INTO taxi_data_copy SELECT * from taxi_data;")


def execute_batch(conn, df, table, page_size=1000):
    """
    Using cursor.execute_batch() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ",".join(list(df.columns))
    # SQL quert to execute
    query = "INSERT INTO %s(%s) " % (table, cols)
    placeholders = ",".join(["%s" for _ in range(len(list(df.columns)))])
    query += f"VALUES({placeholders})"
    print(query)
    cursor = conn.cursor()
    try:
        extras.execute_batch(cursor, query, tuples, page_size)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_many() done")
    cursor.close()


# List all files in directory
data_files = [fname for fname in os.listdir(data_dir) if ".csv" in fname]

# Remove processed files
processed_files = open(processed_fname, "r").readlines()
processed_files = [fname.strip() for fname in processed_files]
data_files = sorted(list(set(data_files) - set(processed_files)))

# Specify dtypes
dtypes = {
    "VendorID": "string",
    "passenger_count": "int",
    "trip_distance": "float",
    "RateCodeID": "int",
    "store_and_fwd_flag": "string",
    "PULocationID": "int",
    "DOLocationID": "int",
    "payment_type": "string",
    "fare_amount": "float",
    "extra": "float",
    "mta_tax": "float",
    "tip_amount": "float",
    "tolls_amount": "float",
    "improvement_surcharge": "float",
    "total_amount": "float",
    "congestion_surcharge": "float",
}

for fname in data_files:
    print(f"Processing {fname}...")

    import time

    start_time = time.time()
    # Delete staging table
    cur.execute("DROP TABLE IF EXISTS taxi_data_staging;")
    cur.execute(
        """
            CREATE TABLE taxi_data_staging (
                VendorID text,
                tpep_pickup_datetime timestamp without time zone,
                tpep_dropoff_datetime timestamp without time zone,
                passenger_count numeric,
                trip_distance numeric,
                RateCodeID numeric,
                store_and_fwd_flag text,
                PULocationID numeric,
                DOLocationID numeric,
                payment_type text,
                fare_amount numeric,
                extra numeric,
                mta_tax numeric,
                tip_amount numeric,
                tolls_amount numeric,
                improvement_surcharge numeric,
                total_amount numeric,
                congestion_surcharge numeric
            )
        """
    )

    df = pd.read_csv(
        os.path.join(data_dir, fname),
        parse_dates=True,
        # dtype=dtypes,
        on_bad_lines="skip",
    )
    # df.replace("", np.nan, inplace=True)
    df = df.dropna()
    # df = df.where(df.notna(), None)

    # Convert datetime
    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"], errors="coerce"
    )
    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"], errors="coerce"
    )
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    print(df.head())

    df.to_csv(".temporary.csv", index=False)
    print(f"Wrote out csv with len {len(df)}")

    with open(".temporary.csv", "r") as f:
        next(f)  # Skip the header row.
        cur.copy_from(f, "taxi_data_staging", sep=",")
    os.remove(".temporary.csv")

    # execute_batch(conn, df, "taxi_data_staging")
    # # Copy staging table to existing table
    cur.execute("INSERT INTO taxi_data SELECT * from taxi_data_staging;")

    end_time = time.time()
    print(f"Took {end_time - start_time} to complete.")

    # Mark file as read
    open(processed_fname, "a").write(fname + "\n")
