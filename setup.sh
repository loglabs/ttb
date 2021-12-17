#!/bin/bash

createdb nyc-taxi-data

psql nyc-taxi-data -f setup_files/create_nyc_taxi_schema.sql

year_month_regex="tripdata_([0-9]{4})-([0-9]{2})"
schema_header="(vendor_id,tpep_pickup_datetime,tpep_dropoff_datetime,passenger_count,trip_distance,rate_code_id,store_and_fwd_flag,pickup_location_id,dropoff_location_id,payment_type,fare_amount,extra,mta_tax,tip_amount,tolls_amount,improvement_surcharge,total_amount,congestion_surcharge)"

for filename in data/yellow_tripdata*.csv; do
  [[ $filename =~ $year_month_regex ]]
  year=${BASH_REMATCH[1]}
  month=$((10#${BASH_REMATCH[2]}))

  echo "`date`: beginning load for ${filename}"
  sed $'s/\r$//' $filename | sed '/^$/d' | psql nyc-taxi-data -c "COPY yellow_tripdata_staging ${schema} FROM stdin CSV HEADER;"
  echo "`date`: finished raw load for ${filename}"
  psql nyc-taxi-data -f setup_files/populate_yellow_trips.sql
  echo "`date`: loaded trips for ${filename}"
done;

psql nyc-taxi-data -c "CREATE INDEX ON trips USING BRIN (pickup_datetime) WITH (pages_per_range = 32);"