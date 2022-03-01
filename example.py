from datetime import datetime, timedelta
from ttb import Dataset

if __name__ == "__main__":
    dataset = Dataset("taxi_data", cutoff_date="2021-01-01", backend="pandas")

    # Query dataset
    df = dataset.loadRecent(timedelta(days=5))
    print(df.head())

    intel_lab_data = Dataset(
        "intel_lab_data", cutoff_date="2004-03-04", backend="pandas"
    )

    # Query dataset
    df = intel_lab_data.loadRecent(timedelta(days=5))
    print(df.head())
