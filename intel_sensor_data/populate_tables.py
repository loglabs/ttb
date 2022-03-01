import numpy as np
import os
import pandas as pd
from pandas.core.dtypes import dtypes
import psycopg2
import psycopg2.extras as extras

from io import StringIO
from dotenv import load_dotenv

data_file = os.path.join(os.getcwd(), "data.txt")
load_dotenv()


conn = psycopg2.connect(
    f"host={os.getenv('HOSTNAME')} user={os.getenv('USERNAME')} port={os.getenv('PORT')} password={os.getenv('SECRET')}"
)
conn.set_isolation_level(0)

cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS intel_lab_data (
        reading_timestamp timestamp without time zone,
        epoch integer,
        moteid integer,
        temperature numeric,
        humidity numeric,
        light numeric,
        voltage numeric
    )
"""
)

cur.execute("DROP TABLE IF EXISTS intel_lab_data_copy;")
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS intel_lab_data_copy (
        reading_timestamp timestamp without time zone,
        epoch integer,
        moteid integer,
        temperature numeric,
        humidity numeric,
        light numeric,
        voltage numeric
    )
"""
)
cur.execute("INSERT INTO intel_lab_data_copy SELECT * from intel_lab_data;")


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


print(f"Processing {data_file}...")

import time

start_time = time.time()
# Delete staging table
cur.execute("DROP TABLE IF EXISTS intel_lab_data_staging;")
cur.execute(
    """
        CREATE TABLE intel_lab_data_staging (
            reading_timestamp timestamp without time zone,
            epoch integer,
            moteid integer,
            temperature numeric,
            humidity numeric,
            light numeric,
            voltage numeric
        )
    """
)

df = pd.read_csv(
    data_file,
    on_bad_lines="skip",
    sep=" ",
    header=None,
)
df.columns = [
    "reading_date",
    "reading_hour",
    "epoch",
    "moteid",
    "temperature",
    "humidity",
    "light",
    "voltage",
]
df = df.dropna()

# Merge reading date and reading hour
df["reading_timestamp"] = pd.to_datetime(
    df["reading_date"] + " " + df["reading_hour"]
)
df.drop(columns=["reading_date", "reading_hour"], inplace=True)
df = df.dropna(subset=["reading_timestamp"])
df["moteid"] = df["moteid"].astype(int)
cols = [df.columns[-1]] + list(df.columns[:-1])
df = df[cols]
print(df.head())

df.to_csv(".temporary.csv", index=False)
print(f"Wrote out csv with len {len(df)}")

with open(".temporary.csv", "r") as f:
    next(f)  # Skip the header row.
    cur.copy_from(f, "intel_lab_data_staging", sep=",")
os.remove(".temporary.csv")

# # Copy staging table to existing table
cur.execute("INSERT INTO intel_lab_data SELECT * from intel_lab_data_staging;")

end_time = time.time()
print(f"Took {end_time - start_time} to complete.")

# Create index on reading_timestamp
cur.execute("CREATE INDEX ON intel_lab_data (reading_timestamp);")
