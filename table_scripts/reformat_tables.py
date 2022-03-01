import numpy as np
import os
import psycopg2

from io import StringIO
from dotenv import load_dotenv

load_dotenv()


conn = psycopg2.connect(
    f"host={os.getenv('HOSTNAME')} user={os.getenv('USERNAME')} port={os.getenv('PORT')} password={os.getenv('SECRET')}"
)
conn.set_isolation_level(0)

cur = conn.cursor()

# Copy taxi_data table to taxi_data_copy
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
cur.execute(
    "INSERT INTO taxi_data_copy SELECT * from taxi_data ORDER BY tpep_pickup_datetime ASC;"
)

# Create new pkey column
cur.execute("ALTER TABLE taxi_data_copy ADD COLUMN id SERIAL PRIMARY KEY;")

# Migrate back to taxi_data
cur.execute("DROP TABLE IF EXISTS taxi_data;")
cur.execute("ALTER TABLE taxi_data_copy RENAME TO taxi_data;")
cur.execute("CREATE INDEX ON taxi_data (tpep_pickup_datetime);")
cur.execute("CREATE INDEX ON taxi_data (tpep_dropoff_datetime);")
