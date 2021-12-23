# TTSB: Tabular Time Series Benchmarks

Author: shreyashankar

This repository contains helper functions to read and deploy models on data from time series benchmarks. Work in progress.

## Data Criteria

To be included as a benchmark, the data source must:

* Have a corresponding tractable ML task
* Be time series & tabular in nature

## Current Data Sources

* [NYC Taxicab Yellow Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), ingested into a timestamp-indexed table hosted on AWS RDS

## How To Use

### Install

This package is currently not hosted on PyPI. To install, `git clone` this repo and run `pip install -e .` in the project root directory. You will also need keys to pull from the DB containing the data sources, so email me if you are interested in having the keys.

### DB Config

You can run all the following functions locally, but you will need to access the DB containing the task data. To do this, create a file named `.env` in the root directory with the following contents (which you will receive after emailing me for credentials):

```
HOSTNAME=...
USERNAME=...
PORT=...
SECRET=...
```

### API

The core `Dataset` abstraction accepts a name (currently only "taxi_data"), cutoff date, and backend (currently only `pandas`). The cutoff date exists to prevent accidental [data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/). It contains two fuctions:

* `load(start_date, end_date)`: Takes in string or datetime objects representing dates. Returns a dataframe containing data from [start_date, end_date).
* `load_recent(delta)`: Takes in a timedelta of recent data to load in a dataframe.

The code is as follows:

```python
class Dataset:
    def __init__(
        self,
        name: str,
        cutoff_date: typing.Union[str, datetime], # %Y-%m-%d
        cache_dir: str = None,
        backend: str = "pandas",
    ):
        ...

    def load(
        self,
        start_date: typing.Union[str, datetime], # %Y-%m-%d
        end_date: typing.Union[str, datetime], # %Y-%m-%d
    ) -> pd.DataFrame:
        """Method to load data for the dataset.

        Args:
            start_date (typing.Union[str, datetime]): Start date of the data (inclusive).
            end_date (typing.Union[str, datetime]): End date of the data (exclusive).

        Raises:
            ValueError: When end date is before start date.
            ValueError: When end date is after cutoff date.

        Returns:
            pd.DataFrame: Loaded data for the dataset.
        """
        ...

    def loadRecent(self, delta: timedelta) -> pd.DataFrame:
        """Method to load recent data for the dataset.

        Args:
            delta (timedelta): How far back in time to load data.

        Returns:
            pd.DataFrame: Loaded data for the dataset.
        """

        ...
```

## Ongoing Work

If you are interested in working on any of the following, please create / comment on an issue or email me!

* **Adding more data sources in this format:** email me if you have ideas and would like to contribute. This involves creating a migration from the source to the RDS instance.
* **Supporting backends other than Pandas:** We plan to support TF & PyTorch data objects.
* **Caching data once it has been loaded locally:** To prevent multiple unnecessary DB reads, we can add a cache layer to this project.