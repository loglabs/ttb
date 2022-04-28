import typing

import pandas as pd


"""
    Returns df column with tip percentages. If the total amount 
    of a trip is 0, the tip percent is calculated as 0 by default.
"""
def tip_percent_column(total_amount_series: pd.Series,
                       tip_amount_series: pd.Series):
    zero_total_indices = total_amount_series == 0
    tip_percents_including_nans = tip_amount_series / total_amount_series
    tip_percents_including_nans[zero_total_indices] = 0
    return tip_percents_including_nans


def dollars_per_mile(total_amount_series: pd.Series,
                       trip_distance_series: pd.Series):
    zero_distance_indices = trip_distance_series == 0
    dollars_per_mile = total_amount_series / trip_distance_series
    dollars_per_mile[zero_distance_indices] = 0
    return dollars_per_mile


"""
    Returns df column with day picked up in month.
"""
def pickup_day_in_month(pickup_date_series: pd.Series):
    return pickup_date_series.dt.day

"""
    Returns pickup time in mins since midnight of the day.
"""
def pickup_time_sice_midnight(pickup_date_series: pd.Series):
    return pickup_date_series.dt.hour * 60 + pickup_date_series.dt.minute

"""
    Returns pickup day of week from 0-6.
"""
def pickup_day_in_week(pickup_date_series: pd.Series):
    return pickup_date_series.dt.weekday

"""
    Returns month from 0 BC to make it easier for plotting across months across years.
"""
def pickup_month_inc_year(pickup_date_series: pd.Series):
    return pickup_date_series.dt.year * 12 + pickup_date_series.dt.month

