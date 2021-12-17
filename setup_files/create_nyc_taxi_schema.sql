-- This file reads in the raw taxicab data and converts it
-- into postgres tables.

CREATE TABLE yellow_tripdata_staging (
  id bigserial primary key,
  vendor_id text,
  tpep_pickup_datetime text,
  tpep_dropoff_datetime text,
  passenger_count text,
  trip_distance text,
  pickup_longitude numeric,
  pickup_latitude numeric,
  rate_code_id text,
  store_and_fwd_flag text,
  dropoff_longitude numeric,
  dropoff_latitude numeric,
  payment_type text,
  fare_amount text,
  extra text,
  mta_tax text,
  tip_amount text,
  tolls_amount text,
  improvement_surcharge text,
  total_amount text,
  pickup_location_id text,
  dropoff_location_id text,
  congestion_surcharge text,
  junk1 text,
  junk2 text
)
WITH (
  autovacuum_enabled = false,
  toast.autovacuum_enabled = false
);

CREATE TABLE jan_2019 (
  VendorID text,
  tpep_pickup_datetime timestamp without time zone,
  tpep_dropoff_datetime timestamp without time zone,
  Passenger_count integer,
  Trip_distance numeric,
  PULocationID integer,
  DOLocationID integer,
  RateCodeID integer,
  Store_and_fwd_flag text,
  Payment_type text,
  Fare_amount numeric,
  Extra numeric,
  MTA_tax numeric,
  Improvement_surcharge numeric,
  Tip_amount numeric,
  Tolls_amount numeric,
  Total_amount numeric
);