import typing

import pandas as pd

"""
    Cleans up all strings from each column of the NYC Taxi dataset.
"""
def hard_coded_cleaning_steps_mutation(df: pd.DataFrame):
    # Correct string type confusion in vendorid, payment_type
    df['vendorid'] = df['vendorid'].astype(float).astype(int)
    df['payment_type'] = df['payment_type'].astype(float).astype(int)

    # Y/N more useful as 0/1 for graphing and analysis
    df.loc[df['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0
    df.loc[df['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1