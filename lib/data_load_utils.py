import typing

import numpy as np
import pandas as pd

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from ttb import Dataset
from typing import Optional
from tqdm.auto import tqdm


class TrainTestDataFrameAccessor:
    '''
        Supports querying for dataframes corresponding to the training label
        and nonlabel features of the original df.
    '''
    def __init__(self, df_data: pd.DataFrame,
        time_index_col: pd.Series,
        x_labels: tuple[str, ...],
        y_label: str,
        date_slice_fit: tuple[datetime, datetime],
        date_slices_infer: tuple[tuple[datetime, datetime], ...]):

        self.x_accessor = SingleDataFrameAccessor(
            df_data[list(x_labels)], time_index_col,
            date_slice_fit, date_slices_infer)
        self.y_accessor = SingleDataFrameAccessor(
            df_data[[y_label]], time_index_col,
            date_slice_fit, date_slices_infer)

    def get_fit_dates(self) -> tuple[datetime, datetime]:
        return self.x_accessor.get_fit_dates()

    def get_dates_infer(self) -> tuple[tuple[datetime, datetime], ...]:
        return self.x_accessor.get_dates_infer()
    
    def get_x_fit(self) -> pd.DataFrame:
        return self.x_accessor.get_df_fit()

    def get_x_infer_iterable(self) -> Iterable[pd.DataFrame]:
        yield from self.x_accessor.get_df_infer_iterable()
    
    def get_y_fit(self) -> pd.DataFrame:
        return self.y_accessor.get_df_fit()

    def get_y_infer_iterable(self) -> Iterable[pd.DataFrame]:
        yield from self.y_accessor.get_df_infer_iterable()

    def get_xy_fit_np(self) -> tuple[np.array, np.array]:
        return (self.get_x_fit().values.astype(np.float32),
            self.get_y_fit().values.astype(np.float32))

    def get_xy_infer_iterable_np(self) -> Iterable[np.array, np.array]:
        for x_df, y_df in zip(self.get_x_infer_iterable(),
            self.get_y_infer_iterable()):
            yield (x_df.values.astype(np.float32),
                y_df.values.astype(np.float32))


class AnnotatedTrainTestDataFrameAccessor(TrainTestDataFrameAccessor):
    def __init__(self, df_data: pd.DataFrame,
        time_index_col: pd.Series,
        x_labels: tuple[str, ...],
        y_label: str,
        date_slice_fit: tuple[datetime, datetime],
        date_slices_infer: tuple[tuple[datetime, datetime], ...],
        is_continuous_annotations: dict):
        TrainTestDataFrameAccessor.__init__(
            self, df_data, time_index_col, x_labels, y_label,
            date_slice_fit, date_slices_infer)
        self.x_accessor.is_continuous_annotations = is_continuous_annotations
        self.y_accessor.is_continuous_annotations = is_continuous_annotations

    def get_xy_fit_np(self) -> tuple[
        tuple[np.array, np.array], tuple[np.array, np.array]]:
        x_np, y_np = super().get_xy_fit_np()
        return ((x_np, self.x_accessor.get_annotations()),
            (y_np, self.y_accessor.get_annotations()))

    def get_xy_infer_iterable_np(self) -> Iterable[
        tuple[np.array, np.array], tuple[np.array, np.array]]:
        '''
            TODO: properly propogate the logic of annotations into the inner class
            because clearly the annotations are tied to each of x and y values
            and can be set as an attribute during init
        '''
        for xy_tup in super().get_xy_infer_iterable_np():
            x_np, y_np = xy_tup
            yield ((x_np, self.x_accessor.get_annotations()),
                (y_np, self.y_accessor.get_annotations()))

@dataclass
class SingleDataFrameAccessor:
    '''
        Supports iterating through a df by date_slices.
    '''
    df_data: pd.DataFrame
    time_index_col: pd.Series
    date_slice_fit: tuple[datetime, datetime]
    date_slices_infer: tuple[tuple[datetime, datetime], ...]
    is_continuous_annotations: Optional[dict] = None

    def get_fit_dates(self) -> tuple[datetime, datetime]:
        return self.date_slice_fit

    def get_annotations(self) -> np.array:
        return np.array(list(
            map(self.is_continuous_annotations.get,
            list(self.df_data.columns))))
    '''
        Unpacks date_slice into inclusive, exclusive start_date, end_date
        for slicing df_data.
    '''
    def get_date_indices_(self, date_slice: tuple[datetime, datetime]):
        start_date, end_date = date_slice
        return (self.time_index_col >= start_date) & \
            (self.time_index_col < end_date)

    def get_df_fit(self) -> pd.DataFrame:
        return self.df_data[self.get_date_indices_(self.date_slice_fit)]
    
    def get_dates_infer(self) -> tuple[tuple[datetime, datetime], ...]:
        return self.date_slices_infer

    def get_df_infer_iterable(self) -> Iterable[pd.DataFrame]:
        for date_slice in self.date_slices_infer:
            yield self.df_data[self.get_date_indices_(date_slice)]
    
    '''
        Yields the inclusive start date for each df slice for display
        purposes. Yield datetime instead of Tuple[datetime, datetime]
        for ease of display.
    '''
    def get_curr_date_iterable(self) -> Iterable[datetime]:
        for date_slice in self.date_slices_infer:
            yield date_slice[0]

    def get_future_date_iterable(self) -> Iterable[datetime]:
        for date_slice in self.date_slices_infer:
            yield date_slice[1]

@dataclass
class AccessorFactory:
    '''
        Dependent on whether is_continuous_annotations
        is set or not, returns TrainTestDataFrameAccessor
        or AnnotatedTrainTestDataFrameAccessor
    '''
    df_data: pd.DataFrame
    time_index_col: pd.Series
    date_slice_fit: tuple[datetime, datetime]
    date_slices_infer: tuple[tuple[datetime, datetime], ...]
    is_continuous_annotations: Optional[dict] = None

    def get(self, x_labels: tuple[str, ...], y_label: str):
        if self.is_continuous_annotations is None:
            return TrainTestDataFrameAccessor(
                self.df_data,
                self.time_index_col,
                x_labels,
                y_label,
                self.date_slice_fit,
                self.date_slices_infer)
        else:
            return AnnotatedTrainTestDataFrameAccessor(
                self.df_data,
                self.time_index_col,
                x_labels,
                y_label,
                self.date_slice_fit,
                self.date_slices_infer,
                self.is_continuous_annotations)

'''
    Batches and appends pandas dataframes. Useful for loading
    large datasets because memory swaps kill performance.
    Args:
        progress_format: either 'print' or 'tqdm'
'''
def batch_load_df(dataset: Dataset, load_start: datetime, 
    load_end: datetime, no_batches: int,
    progress_format: str) -> pd.DataFrame:
    
    date_iter = date_iter_step_no(load_start, load_end,
        step_no = no_batches)
    if progress_format == 'tqdm':
        date_iter = tqdm(date_iter, total=no_batches)
    
    df_list = []
    for curr_date, future_date in date_iter:
        start_time = datetime.now()
        df_list.append(dataset.load(curr_date, future_date))
        if progress_format == 'print':
            print(f"Loaded { curr_date } to { future_date } in "\
                f"{ datetime.now() - start_time }", end="\r")
        
    df_intermonth = pd.concat(df_list)
    del df_list[:] # free space since loading is so expensive
    return df_intermonth


'''
    Iterate between start_date (inc) and end_date(inc) for chunking
    data for loading and graphing.

    args:
        step_no: subdivisions of delta to advance by for memory
        management in batch loading
    
    yields:
        (curr_date, future_date), inclusive exclusive tuple whose
        union covers entire date range.
'''
def date_iter_step_no(
    start_date: typing.Union[str, datetime],
    end_date: typing.Union[str, datetime],
    step_no: int = 100,
    ):
    delta = (end_date - start_date) / step_no
    curr_date = start_date
    while curr_date < end_date:
        yield (curr_date, curr_date + delta)
        curr_date += delta

def date_iter_step_size(
    start_date: typing.Union[str, datetime],
    end_date: typing.Union[str, datetime],
    step_size: timedelta = timedelta(days=7),
    ):
    curr_date = start_date
    while curr_date < end_date:
        yield (curr_date, curr_date + step_size)
        curr_date += step_size