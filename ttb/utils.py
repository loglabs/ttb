"""Utility functions for caching data."""

from datetime import datetime

import joblib
import os


def generate_filename(
    start_date: datetime, end_date: datetime, backend: str, cache_dir: str
) -> str:
    """Generates a filename for the data.

    Args:
        start_date (datetime): Start date of the data (inclusive).
        end_date (datetime): End date of the data (exclusive).
        backend (str): Backend being used.
        cache_dir (str): Directory to store the data.

    Returns:
        str: Filename for the data.
    """
    filename = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y-%m-%d')}_{backend}.joblib"
    return os.path.join(cache_dir, filename)


def save_to_filename(data: object, filename: str):
    """Saves data to filename.

    Args:
        data (object): Data to save.
        filename (str): Filename to save data to.
    """
    joblib.dump(data, filename)
