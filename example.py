from datetime import datetime, timedelta
from ttsb import Dataset

if __name__ == "__main__":
    dataset = Dataset("taxi_data", cutoff_date="2021-01-01", backend="pandas")

    # Query dataset
    df = dataset.loadRecent(timedelta(days=5))
    print(df.head())
